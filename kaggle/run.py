"""Kaggle kernel entrypoint for AgentLight.

Self-contained, mirroring the GPTlight GitHub->Kaggle workflow: install deps,
sync the repo, run the training pipeline. Resumes automatically across the
weekly-quota session cuts (pipeline_state.json + adapter checkpoints).

Usage: push with the Kaggle CLI from the repo root:
    kaggle kernels push -p kaggle
Attach a checkpoints dataset to resume a previous session (see
checkpoints/README.md), and set the accelerator to **GPU T4 x2** in the
notebook settings (the code asserts the GPU so a wrong assignment fails loud).

Run a smoke test first (flip SMOKE = True below) on any fresh image or
dependency bump: ~10-20 min end-to-end (install -> data -> train -> save
checkpoint) instead of hours, to catch breakage before burning real quota.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# --- 1. Point this at your pushed repo (after `git push`) --------------------
REPO_URL = "https://github.com/Say43/AgentLight.git"
REPO_DIR = "/kaggle/working/AgentLight"
BRANCH = "main"
PHASES = ["reasoning_sft", "general_sft", "grpo"]  # trim to fit a session
USE_DDP = False  # single-GPU (project decision): Unsloth OSS multi-GPU is
# unstable (hangs / silent gradient corruption vs. its patched graph). Flip to
# True only to *experiment* with torchrun DDP AFTER a working single-GPU pilot.

# --- 2. Smoke-test toggle -----------------------------------------------------
# Flip to True to validate the whole pipeline cheaply before a real run: this
# exports AGENTLIGHT_SMOKE=1 to the training process, which config/config.py
# reads to shrink the three phases to 13 total steps and cap datasets to a tiny
# sample. Always smoke-test first on a fresh Kaggle image or dependency pin
# bump (see requirements.txt) — it's ~10-20 min vs. hours for a real phase.
SMOKE = True

# --- 3. Pinned dependency versions --------------------------------------------
# Candidate training stack for the mandatory smoke run. This dict is what gets
# installed on Kaggle; Unsloth's CUDA stack does not belong in a plain local
# `pip install -r requirements.txt` used for chat/eval. Replaces the
# old unpinned `pip install --upgrade`, which was a P0 risk: a drifted
# Unsloth/TRL/vLLM release could silently change trainer behavior mid
# quota-burn. src/train.py feature-detects the handful of API shapes that
# have moved across Unsloth/TRL releases. These pins become the validated
# baseline only after the two-GPU smoke kernel succeeds; any bump requires the
# same smoke test again.
PINNED = {
    "unsloth": "2026.7.4",
    "unsloth_zoo": "2026.7.3",
    "trl": "1.8.0",
}


def sh(cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def install():
    pkgs = [f"{name}=={version}" for name, version in PINNED.items()]
    sh([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs)
    sh([sys.executable, "-m", "pip", "check"])


def sync_repo():
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        sh(["git", "-C", REPO_DIR, "fetch", "origin"])
        sh(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"])
    else:
        sh(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR])


def restore_checkpoints(out_dir):
    """Copy a previously-attached checkpoints dataset into AGENTLIGHT_OUT so
    src/train.py's pipeline_state.json resume logic actually picks it up.

    A prior session's `checkpoints/` output, re-attached as a Kaggle
    dataset_source (see checkpoints/README.md), is mounted read-only
    somewhere under /kaggle/input/ — the exact subpath depends on the
    dataset's slug, so we don't hardcode it. Instead we walk /kaggle/input
    looking for the marker file (pipeline_state.json) and copy that whole
    directory's contents into the writable output dir. Without this step,
    resume was a no-op: the attached dataset just sat unused in
    /kaggle/input and every session silently restarted from phase 1 (the P0
    bug this fixes).
    """
    marker = "pipeline_state.json"
    found = None
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if marker in files:
                found = root
                break
    if found is None:
        print("[resume] no prior checkpoints — fresh run", flush=True)
        return
    print(f"[resume] found prior checkpoints at {found}, "
          f"copying into {out_dir}", flush=True)
    shutil.copytree(found, out_dir, dirs_exist_ok=True)
    print(f"[resume] restored: {sorted(os.listdir(out_dir))}", flush=True)


# ---------------------------------------------------------------------------
# GPU assignment
# ---------------------------------------------------------------------------
# Current Unsloth supports DDP through torchrun. The smoke run uses this exact
# two-rank path so a Kaggle image/API incompatibility is caught before quota is
# committed. Config halves gradient accumulation under WORLD_SIZE=2, preserving
# the intended global batch rather than silently doubling it.
def training_command():
    """Build the exact single- or two-GPU launch command."""
    train_script = os.path.join(REPO_DIR, "src", "train.py")
    if not USE_DDP:
        return [sys.executable, train_script, "--phases", *PHASES]

    import torch
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(
            f"USE_DDP=True requires 2 GPUs, but torch sees {gpu_count}. "
            "Select the Kaggle T4 x2 accelerator or set USE_DDP=False.")
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=2",
        train_script, "--phases", *PHASES,
    ]


def main():
    if SMOKE:
        os.environ["AGENTLIGHT_SMOKE"] = "1"
        print("[smoke] AGENTLIGHT_SMOKE=1 — shrunk smoke-test run", flush=True)

    install()
    sync_repo()

    # Persist checkpoints to the writable working dir so Kaggle saves them as
    # output (attach that output as a dataset next session to resume).
    default_out = ("/kaggle/working/AgentLight/smoke_checkpoints" if SMOKE
                   else "/kaggle/working/AgentLight/checkpoints")
    out_dir = os.environ.setdefault("AGENTLIGHT_OUT", default_out)
    os.makedirs(out_dir, exist_ok=True)
    # Never restore smoke state: its completed markers would make the real
    # run skip phases if someone accidentally attached the smoke output.
    if not SMOKE:
        restore_checkpoints(out_dir)

    sys.path.insert(0, REPO_DIR)
    sh(training_command(), env=os.environ)


if __name__ == "__main__":
    main()
