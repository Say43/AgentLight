"""Kaggle kernel entrypoint for AgentLight.

Self-contained, mirroring the GPTlight GitHub->Kaggle workflow: install deps,
sync the repo, run the training pipeline. Resumes automatically across the
weekly-quota session cuts (pipeline_state.json + adapter checkpoints).

Usage: push with the Kaggle CLI from the repo root:
    kaggle kernels push -p kaggle
Attach a checkpoints dataset to resume a previous session (see
checkpoints/README.md). GPU selection is forced via kernel-metadata.json's
"machine_shape": "NvidiaTeslaT4" -- Kaggle has no API field to request T4x2
specifically (only T4/P100/TPU), so a *single* T4 is what gets reserved; a
lucky pool assignment may still hand out 2, but don't rely on it. Plain
"enable_gpu": true left Kaggle free to hand out a P100 (compute capability
6.0, unsupported by Unsloth) instead -- assert_gpu() catches that loudly if
it ever happens again, but forcing machine_shape avoids wasting the queue
wait on a doomed run in the first place.

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
PHASES = ["reasoning_sft", "repair_sft", "general_sft", "grpo"]  # trim to fit a session
USE_DDP = False  # single-GPU (project decision): Unsloth OSS multi-GPU is
# unstable (hangs / silent gradient corruption vs. its patched graph). Flip to
# True only to *experiment* with torchrun DDP AFTER a working single-GPU pilot.

# --- 2. Smoke-test toggle -----------------------------------------------------
# Flip to True to validate the whole pipeline cheaply before a real run: this
# exports AGENTLIGHT_SMOKE=1 to the training process, which config/config.py
# reads to shrink the three phases to 13 total steps and cap datasets to a tiny
# sample. Always smoke-test first on a fresh Kaggle image or dependency pin
# bump (see requirements.txt) — it's ~10-20 min vs. hours for a real phase.
SMOKE = False  # REAL run (full phase steps + sample sizes). Smoke validated in v11.

# --- Mode: "train" runs the phase pipeline; "eval" runs HumanEval pass@1 on a
# single model and prints the number (no training). Use "eval" with
# EVAL_ADAPTER pointing at the base model for the "before" baseline, or at a
# trained checkpoint dir for the "after". EVAL_N caps the number of HumanEval
# tasks (164 = full benchmark).
MODE = "train"  # "train" runs the phase pipeline; "eval" = HumanEval only
EVAL_ADAPTER = "unsloth/Llama-3.2-3B-Instruct"  # base model => baseline number
EVAL_N = 40

# --- 3. Dependency install ----------------------------------------------------
# Unsloth + unsloth_zoo + trl are tightly version-coupled, and jointly pinning
# exact versions of all three (`pip install X==a Y==b Z==c`) forces pip's
# resolver to solve them simultaneously against each other's declared ranges —
# that failed on the first smoke run (unsloth==2026.7.4 vs
# unsloth_zoo==2026.7.3: "ResolutionImpossible"), before any GPU code even ran.
# Unsloth's own Kaggle/Colab docs sidestep this with --no-deps: install the
# tightly-coupled trio without asking pip to jointly resolve them, and let
# their already-declared transitive deps (torch, transformers, etc., present
# on the Kaggle base image) stay as-is.
#
# Top-level versions PINNED to the exact set that produced the first fully
# green end-to-end smoke run (v11): unsloth 2026.7.4 + unsloth_zoo 2026.7.4 +
# trl 1.9.0 (torch 2.10.0+cu128, transformers 5.0.0, datasets 5.0.0 come from
# the Kaggle base image). --no-deps means these exact pins do NOT re-trigger
# the resolver conflict that killed v1 (unsloth 2026.7.4 vs unsloth_zoo
# 2026.7.3). peft/triton/cut_cross_entropy/xformers are left unpinned (lower
# drift risk, and -q hid their exact versions); install() logs a full
# `pip freeze` of the key packages so a later run can complete the lockfile.
NO_DEPS_PKGS = ["unsloth==2026.7.4", "unsloth_zoo==2026.7.4", "trl==1.9.0",
                "peft==0.19.1", "triton==3.6.0",
                "cut_cross_entropy==25.1.1", "xformers==0.0.35"]
DEPS_PKGS = ["bitsandbytes==0.49.2", "accelerate==1.13.0", "datasets==5.0.0",
             "sentencepiece", "protobuf", "huggingface_hub", "hf_transfer",
             # The Kaggle base image ships torchao 0.10.0. We don't use
             # torchao quantization (bitsandbytes 4-bit is our QLoRA path),
             # but peft's adapter-injection step probes every quantization
             # backend it knows about, including torchao, and raises instead
             # of skipping when a too-old version is merely present (v4 smoke
             # run: "Found version 0.10.0, but only versions above 0.16.0 are
             # supported"). Pinned to the exact version from the green v11/v12
             # runs; forces pip to upgrade the preinstalled 0.10.0 copy.
             "torchao==0.17.0"]


def sh(cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def install():
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + NO_DEPS_PKGS)
    sh([sys.executable, "-m", "pip", "install", "-q"] + DEPS_PKGS)
    # Informational only: --no-deps intentionally skips resolver validation,
    # so `pip check` may report benign gaps. Don't fail the run over it, but
    # log it in case it points at a real incompatibility.
    subprocess.run([sys.executable, "-m", "pip", "check"])
    # Record exact versions of the key packages so the log is a complete,
    # reproducible manifest (the -q installs above hide them). This is how we
    # finish pinning the transitive deps we couldn't read from the v11 log.
    print("[versions] exact installed set:", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=subprocess.PIPE, text=True, check=False)
    import importlib.metadata as im
    for pkg in ["unsloth", "unsloth_zoo", "trl", "peft", "triton",
                "cut_cross_entropy", "xformers", "torch", "torchao",
                "bitsandbytes", "accelerate", "datasets", "transformers",
                "vllm"]:
        try:
            print(f"[versions] {pkg}=={im.version(pkg)}", flush=True)
        except im.PackageNotFoundError:
            print(f"[versions] {pkg} (not installed)", flush=True)


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

    if MODE == "eval":
        # Baseline / checkpoint HumanEval pass@1. Single GPU (T4), no training.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        sh([sys.executable, os.path.join(REPO_DIR, "src", "eval_code.py"),
            "--adapter", EVAL_ADAPTER, "--n", str(EVAL_N)], env=env)
        return

    sh(training_command(), env=os.environ)


if __name__ == "__main__":
    main()
