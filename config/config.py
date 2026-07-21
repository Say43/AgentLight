"""Central configuration for AgentLight.

Everything the training/eval pipeline needs is defined here so a Kaggle run
is reproducible from a single file. Switching the base model between 3B and 8B
is a one-line change (MODEL.name).

Design notes / lessons carried over from the GPTlight project:
- No globbing for dataset paths. Every dataset path is explicit and asserted
  to exist, so a wrong mount order can never silently train on the wrong data.
- One config object is logged verbatim at the start of every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Default: vanilla (NOT reasoning-pretrained) Llama 3.2 3B Instruct.
    # We teach reasoning ourselves — that is the point of the project.
    # Switch to "unsloth/Meta-Llama-3.1-8B-Instruct" when more GPU quota is
    # available (8B GRPO is not realistic in a single 16h Kaggle budget).
    name: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True          # QLoRA
    dtype: str | None = None            # None -> auto (bf16 on Ampere, fp16 on T4)

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0           # 0.0 is Unsloth-optimized
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    use_gradient_checkpointing: str = "unsloth"  # long-context, low VRAM


# ---------------------------------------------------------------------------
# Data — all sources are documented in THIRD_PARTY_NOTICES.md with licenses.
# Kaggle attaches datasets read-only under /kaggle/input/<slug>. When running
# locally the HF hub names are used instead.
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    # SHOWCASE DOMAIN: code generation. Verifiable rewards (run unit tests),
    # which is exactly what makes GRPO work on a small budget.

    # Reasoning SFT: R1-distilled long-CoT (code + math). DeepSeek-R1 is MIT
    # and explicitly permits distillation; this HF dataset is Apache-2.0.
    # Teaches the <think>...</think> reasoning format.
    reasoning_sft_hf: str = "open-thoughts/OpenThoughts-114k"
    reasoning_sft_split: str = "train"

    # General SFT. smoltalk is Apache-2.0 (broad instruction/chat) — keeps the
    # model a usable assistant, not only a code emitter.
    general_sft_hf: str = "HuggingFaceTB/smoltalk"
    general_sft_config: str = "all"

    # GRPO: verifiable-reward CODE. MBPP = Python tasks with executable
    # test_list (CC-BY-4.0 — attribution required, see THIRD_PARTY_NOTICES).
    grpo_hf: str = "google-research-datasets/mbpp"
    grpo_config: str = "full"

    # Held-out eval benchmark for the before/after showcase (HumanEval, MIT).
    eval_hf: str = "openai/openai_humaneval"

    # How many examples to actually use per phase (budget control, not the
    # full corpora — 16h does not fit millions of rows).
    reasoning_sft_max_samples: int = 18_000
    general_sft_max_samples: int = 12_000
    grpo_max_samples: int = 600          # MBPP "full" train is ~374 tasks


# ---------------------------------------------------------------------------
# Per-phase training hyperparameters, tuned around a single 16GB T4.
# ---------------------------------------------------------------------------
@dataclass
class PhaseConfig:
    name: str
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    save_steps: int = 200
    logging_steps: int = 10


@dataclass
class TrainConfig:
    seed: int = 3407
    # Kaggle run.py points this at /kaggle/working so checkpoints are saved as
    # kernel output (attach as a dataset next session to resume).
    output_root: str = field(
        default_factory=lambda: os.environ.get("AGENTLIGHT_OUT", "checkpoints"))

    # Ordered pipeline. reasoning -> sft -> grpo (curriculum: first learn to
    # think in long CoT, then general instruction/tool behavior, then sharpen
    # reasoning with RL on verifiable rewards).
    reasoning_sft: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        name="reasoning_sft", max_steps=1200,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4))
    general_sft: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        name="general_sft", max_steps=900,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4))
    grpo: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        name="grpo", max_steps=400,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=5e-6, warmup_ratio=0.1, save_steps=50))

    # GRPO-specific
    grpo_num_generations: int = 4       # rollouts per prompt (T4-friendly)
    grpo_max_prompt_length: int = 640
    grpo_max_completion_length: int = 1024   # code + reasoning needs room


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def dump(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/input")


CONFIG = Config()


if __name__ == "__main__":
    print(CONFIG.dump())
