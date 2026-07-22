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
    reasoning_sft_config: str = "metadata"
    reasoning_sft_split: str = "train"
    reasoning_sft_domain: str = "code"
    shuffle_seed: int = 3407

    # General SFT. smoltalk is Apache-2.0 (broad instruction/chat) — keeps the
    # model a usable assistant, not only a code emitter.
    general_sft_hf: str = "HuggingFaceTB/smoltalk"
    general_sft_config: str = "all"

    # GRPO: verifiable-reward CODE. MBPP = Python tasks with executable
    # test_list (CC-BY-4.0 — attribution required, see THIRD_PARTY_NOTICES).
    grpo_hf: str = "google-research-datasets/mbpp"
    grpo_config: str = "full"

    # Repair SFT (Kimi-Dev-style skill-then-repair): same license-clean MBPP
    # source as GRPO, reused to build buggy-attempt -> test-failure ->
    # corrected-code trajectories. See data/prepare_data.py:load_repair_sft.
    repair_sft_max_samples: int = 300    # MBPP "full" train has only 374 rows

    # Held-out eval benchmark for the before/after showcase (HumanEval, MIT).
    eval_hf: str = "openai/openai_humaneval"

    # How many examples to actually use per phase (budget control, not the
    # full corpora — 16h does not fit millions of rows).
    reasoning_sft_max_samples: int = 12_000
    general_sft_max_samples: int = 4_000
    grpo_max_samples: int = 374          # MBPP "full" train has 374 tasks


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

    # Ordered pipeline. reasoning -> repair -> sft -> grpo (curriculum: first
    # learn to think in long CoT, then practice repairing a failing attempt
    # from a test trace (Kimi-Dev-style skill-then-repair), then general
    # instruction/tool behavior, then sharpen reasoning with RL on verifiable
    # rewards).
    reasoning_sft: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        name="reasoning_sft", max_steps=600,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4))
    repair_sft: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        name="repair_sft", max_steps=200,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=1.5e-4))
    general_sft: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        name="general_sft", max_steps=250,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=1e-4))
    grpo: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        name="grpo", max_steps=200,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=5e-6, warmup_ratio=0.1, save_steps=50))

    # GRPO-specific
    grpo_num_generations: int = 4       # rollouts per prompt (T4-friendly)
    grpo_max_prompt_length: int = 640
    grpo_max_completion_length: int = 512

    # GRPO curriculum filter (DeepSWE-style): before training, cheaply probe
    # a subsample of tasks and drop the ones that are trivially all-pass or
    # all-fail for the current policy — those give zero reward variance and
    # therefore zero GRPO gradient, so probing them out saves rollout budget
    # for tasks that can actually teach the policy something.
    grpo_curriculum: bool = True
    grpo_curriculum_sample_size: int = 48   # max tasks probed (cheap subsample)
    grpo_curriculum_num_generations: int = 2  # generations/task for the probe
    grpo_curriculum_max_new_tokens: int = 224  # short probe completions
    grpo_curriculum_timeout: float = 3.0    # per-test sandbox timeout (probe)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def dump(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/input")


def _apply_smoke_overrides(cfg: Config) -> None:
    """Shrink every phase while preserving the real code paths and APIs."""
    if os.environ.get("AGENTLIGHT_SMOKE", "").lower() not in {"1", "true", "yes"}:
        return
    cfg.data.reasoning_sft_max_samples = 24
    cfg.data.repair_sft_max_samples = 24
    cfg.data.general_sft_max_samples = 24
    cfg.data.grpo_max_samples = 16
    cfg.train.reasoning_sft.max_steps = 5
    cfg.train.repair_sft.max_steps = 5
    cfg.train.general_sft.max_steps = 5
    cfg.train.grpo.max_steps = 3
    cfg.train.reasoning_sft.save_steps = 5
    cfg.train.repair_sft.save_steps = 5
    cfg.train.general_sft.save_steps = 5
    cfg.train.grpo.save_steps = 3
    cfg.train.grpo_max_completion_length = 128
    # Keep the curriculum pre-pass near-trivial in smoke mode: a couple of
    # tasks, a couple of short generations each.
    cfg.train.grpo_curriculum_sample_size = 4
    cfg.train.grpo_curriculum_num_generations = 2
    cfg.train.grpo_curriculum_max_new_tokens = 48


def _apply_distributed_overrides(cfg: Config) -> None:
    """Preserve the configured global batch when torchrun uses two GPUs."""
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    if world_size == 1:
        return
    for phase in (
        cfg.train.reasoning_sft,
        cfg.train.repair_sft,
        cfg.train.general_sft,
        cfg.train.grpo,
    ):
        phase.gradient_accumulation_steps = max(
            1, phase.gradient_accumulation_steps // world_size)


CONFIG = Config()
_apply_smoke_overrides(CONFIG)
_apply_distributed_overrides(CONFIG)


if __name__ == "__main__":
    print(CONFIG.dump())
