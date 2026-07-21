"""Kaggle kernel entrypoint for AgentLight.

Self-contained, mirroring the GPTlight GitHub->Kaggle workflow: install deps,
sync the repo, run the training pipeline. Resumes automatically across the
weekly-quota session cuts (pipeline_state.json + adapter checkpoints).

Usage: push with the Kaggle CLI from the repo root:
    kaggle kernels push -p kaggle
Attach a checkpoints dataset to resume a previous session (see
checkpoints/README.md), and set the accelerator to **GPU T4 x2** in the
notebook settings (the code asserts the GPU so a wrong assignment fails loud).
"""

import os
import subprocess
import sys

# --- 1. Point this at your pushed repo (after `git push`) --------------------
REPO_URL = "https://github.com/Say43/AgentLight.git"
REPO_DIR = "/kaggle/working/AgentLight"
BRANCH = "main"
PHASES = ["reasoning_sft", "general_sft", "grpo"]  # trim to fit a session


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def install():
    # Unsloth pulls a matching trl/peft/bitsandbytes stack. Pin-free install;
    # if the Kaggle base image drifts, adjust here (this is the one brittle
    # spot — see docs/PLAN.md "Kaggle gotchas").
    sh([sys.executable, "-m", "pip", "install", "-q", "unsloth", "unsloth_zoo"])
    sh([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "trl", "peft", "accelerate", "bitsandbytes", "datasets"])


def sync_repo():
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        sh(["git", "-C", REPO_DIR, "fetch", "origin"])
        sh(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"])
    else:
        sh(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR])


def main():
    install()
    sync_repo()
    # Persist checkpoints to the writable working dir so Kaggle saves them as
    # output (attach that output as a dataset next session to resume).
    os.environ.setdefault("AGENTLIGHT_OUT", "/kaggle/working/AgentLight/checkpoints")
    sys.path.insert(0, REPO_DIR)
    sh([sys.executable, os.path.join(REPO_DIR, "src", "train.py"),
        "--phases", *PHASES])


if __name__ == "__main__":
    main()
