"""AgentLight training pipeline: reasoning-SFT -> general-SFT -> GRPO.

Run one, several, or all phases:

    python src/train.py --phases reasoning_sft general_sft grpo

The pipeline is resumable: each phase saves its LoRA adapter and a
`pipeline_state.json`. Re-running skips completed phases and continues from the
last saved adapter — needed because a full run does not fit one Kaggle session.

LESSONS BAKED IN FROM THE GPTlight PROJECT
------------------------------------------
1. No silent CPU fallback. We assert a CUDA GPU is present and log its name;
   a P100-vs-2xT4 mix-up or a driver fault fails LOUDLY, not after hours.
2. No dataset globbing. Datasets are referenced by explicit HF name (see
   config), never by "first *.bin found", so the wrong corpus can't sneak in.
3. Full state is checkpointed. HF Trainer saves optimizer/scheduler/scaler;
   we additionally persist an atomic pipeline_state.json so a resume never
   re-runs a finished phase or loses which phase it was on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# Make sibling packages importable when run as a script on Kaggle.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG, on_kaggle           # noqa: E402
from data.prepare_data import LOADERS, CODE_SYSTEM_PROMPT  # noqa: E402


# ---------------------------------------------------------------------------
# GPTlight lesson #1: never train on CPU by accident.
# ---------------------------------------------------------------------------
def assert_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU visible. Refusing to fall back to CPU (that wasted "
            "hours on GPTlight). Check the Kaggle accelerator setting.")
    name = torch.cuda.get_device_name(0)
    n = torch.cuda.device_count()
    print(f"[gpu] {n}x {name} | torch {torch.__version__}", flush=True)
    return name


# ---------------------------------------------------------------------------
# Pipeline state (GPTlight lesson #3)
# ---------------------------------------------------------------------------
def state_path(cfg):
    return os.path.join(cfg.train.output_root, "pipeline_state.json")


def load_state(cfg):
    p = state_path(cfg)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"completed": [], "model_name": cfg.model.name}


def save_state(cfg, state):
    os.makedirs(cfg.train.output_root, exist_ok=True)
    p = state_path(cfg)
    # Atomic write so an interrupted save can't corrupt the state file.
    fd, tmp = tempfile.mkstemp(dir=cfg.train.output_root)
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, p)


def phase_dir(cfg, phase_name):
    return os.path.join(cfg.train.output_root, phase_name)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(cfg, resume_from=None):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    load_target = resume_from or cfg.model.name
    print(f"[model] loading {load_target}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_target,
        max_seq_length=cfg.model.max_seq_length,
        dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    if resume_from is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=list(cfg.model.target_modules),
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
            random_state=cfg.train.seed,
        )
    return model, tokenizer


# ---------------------------------------------------------------------------
# SFT phases (reasoning_sft, general_sft)
# ---------------------------------------------------------------------------
def run_sft(cfg, model, tokenizer, phase, dataset):
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    def fmt(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=False)
                 for m in batch["messages"]]
        return {"text": texts}

    dataset = dataset.map(fmt, batched=True, remove_columns=dataset.column_names)

    out = phase_dir(cfg, phase.name)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        packing=False,
        args=SFTConfig(
            output_dir=out,
            per_device_train_batch_size=phase.per_device_train_batch_size,
            gradient_accumulation_steps=phase.gradient_accumulation_steps,
            warmup_ratio=phase.warmup_ratio,
            max_steps=phase.max_steps,
            learning_rate=phase.learning_rate,
            lr_scheduler_type=phase.lr_scheduler_type,
            logging_steps=phase.logging_steps,
            save_steps=phase.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=cfg.train.seed,
            report_to="none",
        ),
    )
    # Assistant-only loss (GPTlight SFT lesson: don't train on the user turns).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    resume = os.path.isdir(out) and any(
        d.startswith("checkpoint-") for d in os.listdir(out))
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[sft] {phase.name} done -> {out}", flush=True)


# ---------------------------------------------------------------------------
# GRPO phase — verifiable code reward
# ---------------------------------------------------------------------------
def _completion_text(c):
    return c[0]["content"] if isinstance(c, list) else c


def correctness_reward(prompts, completions, tests=None, setup=None, **kw):
    """Fraction of MBPP unit tests the generated code passes, scaled to [0,3]."""
    from agent.executor import extract_code, score_solution
    tests = tests or [[] for _ in completions]
    setup = setup or ["" for _ in completions]
    rewards = []
    for comp, t, s in zip(completions, tests, setup):
        code = extract_code(_completion_text(comp))
        rewards.append(3.0 * score_solution(code, t, s))
    return rewards


def format_reward(prompts, completions, **kw):
    """Small shaping bonus for using <think> and exactly one code block."""
    import re
    out = []
    for comp in completions:
        txt = _completion_text(comp)
        r = 0.0
        if "<think>" in txt and "</think>" in txt:
            r += 0.3
        if len(re.findall(r"```python", txt)) == 1:
            r += 0.2
        out.append(r)
    return out


def run_grpo(cfg, model, tokenizer, phase, dataset):
    from trl import GRPOConfig, GRPOTrainer

    out = phase_dir(cfg, phase.name)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=out,
            per_device_train_batch_size=phase.per_device_train_batch_size,
            gradient_accumulation_steps=phase.gradient_accumulation_steps,
            num_generations=cfg.train.grpo_num_generations,
            max_prompt_length=cfg.train.grpo_max_prompt_length,
            max_completion_length=cfg.train.grpo_max_completion_length,
            max_steps=phase.max_steps,
            learning_rate=phase.learning_rate,
            lr_scheduler_type=phase.lr_scheduler_type,
            warmup_ratio=phase.warmup_ratio,
            logging_steps=phase.logging_steps,
            save_steps=phase.save_steps,
            optim="adamw_8bit",
            seed=cfg.train.seed,
            report_to="none",
        ),
    )
    resume = os.path.isdir(out) and any(
        d.startswith("checkpoint-") for d in os.listdir(out))
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[grpo] {phase.name} done -> {out}", flush=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
PHASE_CFG = {
    "reasoning_sft": lambda c: c.train.reasoning_sft,
    "general_sft":   lambda c: c.train.general_sft,
    "grpo":          lambda c: c.train.grpo,
}
PHASE_MAX = {
    "reasoning_sft": lambda c: c.data.reasoning_sft_max_samples,
    "general_sft":   lambda c: c.data.general_sft_max_samples,
    "grpo":          lambda c: c.data.grpo_max_samples,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+",
                    default=["reasoning_sft", "general_sft", "grpo"])
    args = ap.parse_args()

    cfg = CONFIG
    print("[config]\n" + cfg.dump(), flush=True)
    assert_gpu()
    os.makedirs(cfg.train.output_root, exist_ok=True)

    state = load_state(cfg)
    completed = set(state.get("completed", []))

    # Resume from the most recently completed phase's adapter, if any.
    resume_from = None
    for ph in ["grpo", "general_sft", "reasoning_sft"]:
        d = phase_dir(cfg, ph)
        if ph in completed and os.path.isdir(d):
            resume_from = d
            break

    model, tokenizer = build_model(cfg, resume_from=resume_from)

    for ph in args.phases:
        if ph in completed:
            print(f"[skip] {ph} already completed", flush=True)
            continue
        print(f"\n===== PHASE: {ph} =====", flush=True)
        n = PHASE_MAX[ph](cfg)
        dataset = LOADERS[ph](cfg.data, n)
        print(f"[data] {ph}: {len(dataset)} examples", flush=True)
        phase = PHASE_CFG[ph](cfg)
        if ph == "grpo":
            run_grpo(cfg, model, tokenizer, phase, dataset)
        else:
            run_sft(cfg, model, tokenizer, phase, dataset)
        completed.add(ph)
        state["completed"] = sorted(completed)
        save_state(cfg, state)

    print("\n[done] pipeline finished:", sorted(completed), flush=True)


if __name__ == "__main__":
    main()
