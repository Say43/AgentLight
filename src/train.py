"""AgentLight training pipeline:
reasoning-SFT -> repair-SFT -> general-SFT -> GRPO.

Run one, several, or all phases:

    python src/train.py --phases reasoning_sft repair_sft general_sft grpo

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
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
import sys
import tempfile

# Make sibling packages importable when run as a script on Kaggle.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG, on_kaggle           # noqa: E402
from data.prepare_data import LOADERS, CODE_SYSTEM_PROMPT  # noqa: E402


def _supported_kwargs(callable_obj, kwargs):
    """Keep only kwargs supported by the installed TRL API shape."""
    params = inspect.signature(callable_obj).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def _distributed_barrier():
    import torch
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


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
    # GPTlight lesson #4 extended: a Kaggle P100 (Pascal, compute capability
    # 6.0) can be auto-assigned instead of the expected T4 (7.5). Unsloth
    # requires CC >= 7.0 (no bf16 / optimized 4-bit kernels below that), so a
    # P100 would fail deep inside model load after wasting the session. Fail
    # loudly and clearly at startup instead.
    major, minor = torch.cuda.get_device_capability(0)
    if major < 7:
        raise RuntimeError(
            f"GPU '{name}' has compute capability {major}.{minor} < 7.0, which "
            "Unsloth does not support (this is almost certainly a P100). Set the "
            "Kaggle accelerator to 'GPU T4 x2' and rerun.")
    print(f"[gpu] {n}x {name} (cc {major}.{minor}) | torch {torch.__version__}",
          flush=True)
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

    # Long reasoning traces are right-truncated by SFTTrainer. If the final
    # code is cut off, the label teaches exactly the wrong behaviour, so drop
    # overlong rows before training and report how much data was affected.
    def add_lengths(batch):
        encoded = tokenizer(
            batch["text"], add_special_tokens=False, truncation=False)
        return {"token_length": [len(ids) for ids in encoded["input_ids"]]}

    dataset = dataset.map(add_lengths, batched=True, batch_size=64)
    before = len(dataset)
    dataset = dataset.filter(
        lambda length: length <= cfg.model.max_seq_length,
        input_columns=["token_length"],
    )
    removed = before - len(dataset)
    print(f"[data] {phase.name}: removed {removed}/{before} overlong rows "
          f"(>{cfg.model.max_seq_length} tokens)", flush=True)
    if not len(dataset):
        raise RuntimeError(f"{phase.name}: no rows remain after length filtering")
    dataset = dataset.remove_columns(["token_length"])

    out = phase_dir(cfg, phase.name)
    import torch
    bf16 = bool(torch.cuda.is_bf16_supported())
    sft_config_kwargs = {
        "output_dir": out,
        "per_device_train_batch_size": phase.per_device_train_batch_size,
        "gradient_accumulation_steps": phase.gradient_accumulation_steps,
        "warmup_ratio": phase.warmup_ratio,
        "max_steps": phase.max_steps,
        "learning_rate": phase.learning_rate,
        "lr_scheduler_type": phase.lr_scheduler_type,
        "logging_steps": phase.logging_steps,
        "save_steps": phase.save_steps,
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "seed": cfg.train.seed,
        "report_to": "none",
        "fp16": not bf16,
        "bf16": bf16,
        # TRL renamed max_seq_length -> max_length and moved these fields
        # from SFTTrainer to SFTConfig. Feature detection keeps the pinned
        # baseline and future deliberate upgrades on the same code path.
        "dataset_text_field": "text",
        # max_length=None, not cfg.model.max_seq_length: the current TRL
        # defaults padding_free=True, and combining that with packing=False
        # plus a non-None max_length is a hard error ("max_length is not
        # enforced ... provide already truncated inputs, or set
        # max_length=None" -- v6 smoke run). We already drop every row over
        # cfg.model.max_seq_length tokens above (same tokenizer), so the
        # inputs reaching the trainer are already truncated to budget; no
        # further enforcement is needed here.
        "max_length": None,
        "max_seq_length": cfg.model.max_seq_length,
        "packing": False,
    }
    sft_args = SFTConfig(**_supported_kwargs(SFTConfig, sft_config_kwargs))
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": sft_args,
    }
    trainer_params = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    # Compatibility with older TRL where these were trainer kwargs.
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = False
    # No max_seq_length re-injection here (unlike dataset_text_field/packing,
    # which just duplicate an identical value already in sft_args): v7 smoke
    # run reproduced the exact same padding_free/max_length ValueError even
    # after sft_args.max_length was set to None, because Unsloth's SFTTrainer
    # wrapper accepts this legacy direct kwarg and uses it to reconstruct a
    # non-None effective max_length internally, silently overriding the
    # SFTConfig-level setting above. Leave truncation control entirely to
    # sft_args (max_length=None there, backed by our own pre-filter).
    trainer = SFTTrainer(**trainer_kwargs)
    # Assistant-only loss (GPTlight SFT lesson: don't train on the user turns).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    resume = os.path.isdir(out) and any(
        d.startswith("checkpoint-") for d in os.listdir(out))
    trainer.train(resume_from_checkpoint=resume)
    if _is_main_process():
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"[sft] {phase.name} done -> {out}", flush=True)
    _distributed_barrier()


# ---------------------------------------------------------------------------
# GRPO phase — verifiable code reward
# ---------------------------------------------------------------------------
def _completion_text(c):
    return c[0]["content"] if isinstance(c, list) else c


# Set by run_grpo() for the duration of trainer.train(), read by
# correctness_reward(). TRL's reward-func contract does not pass the trainer
# or step count into reward functions, so this is how the reward function
# (which sees raw completions + test scores) and the rollout/metrics logging
# (which needs those same scores) share data without changing the reward
# function's call signature.
_ACTIVE_GRPO_METRICS = None


class _GRPOMetrics:
    """Rollout log (SWE-Gym-style reuse) + per-step training-health summary.

    correctness_reward() calls record() with the raw (unscaled, 0..1) test
    scores it already computed -- no duplicate test execution. The
    RolloutMetricsCallback nested in run_grpo() calls flush() once per step
    with timing/VRAM it alone can observe.
    """

    def __init__(self, rollout_path, metrics_path, num_generations,
                 max_completion_length):
        self.rollout_path = rollout_path
        self.metrics_path = metrics_path
        self.num_generations = max(1, num_generations)
        self.max_completion_length = max(1, max_completion_length)
        self.reset()

    def reset(self):
        self.scores = []
        self.lengths = []

    def record(self, completions, scores):
        if not _is_main_process():
            return
        lines = []
        for comp, s in zip(completions, scores):
            text = _completion_text(comp)
            self.scores.append(s)
            self.lengths.append(len(text))
            lines.append(json.dumps({"completion": text, "test_score": s}))
        if self.rollout_path and lines:
            with open(self.rollout_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    def flush(self, step, elapsed_sec, peak_vram_gb):
        if not _is_main_process():
            self.reset()
            return
        if not self.scores:
            return
        import statistics
        n = self.num_generations
        groups = [self.scores[i:i + n] for i in range(0, len(self.scores), n)]
        zero_variance = sum(
            1 for g in groups if len(g) > 1 and max(g) - min(g) < 1e-6)
        # Chars, not tokens (no tokenizer available here) -- a rough /4
        # chars-per-token estimate is used only to flag likely truncation,
        # not to report an exact token count.
        approx_tokens = [c / 4.0 for c in self.lengths]
        truncated = sum(
            1 for t in approx_tokens
            if t >= self.max_completion_length * 0.98)
        record = {
            "step": step,
            "reward_mean": statistics.fmean(self.scores),
            "reward_std": (statistics.pstdev(self.scores)
                           if len(self.scores) > 1 else 0.0),
            "zero_variance_groups": zero_variance,
            "num_groups": len(groups),
            "completion_chars_mean": statistics.fmean(self.lengths),
            "completion_chars_max": max(self.lengths),
            "truncation_rate_approx": truncated / len(self.lengths),
            "sec_per_step": round(elapsed_sec, 3),
            "peak_vram_gb": round(peak_vram_gb, 3),
        }
        if self.metrics_path:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        print(f"[grpo-metrics] {json.dumps(record)}", flush=True)
        self.reset()


def correctness_reward(prompts, completions, tests=None, setup=None, **kw):
    """Fraction of MBPP unit tests the generated code passes, scaled to [0,3]."""
    from agent.executor import extract_code, score_solution
    tests = tests or [[] for _ in completions]
    setup = setup or ["" for _ in completions]
    jobs = []
    for comp, t, s in zip(completions, tests, setup):
        code = extract_code(_completion_text(comp))
        jobs.append((code, t, s))

    # Each score launches isolated test subprocesses. Run completion groups in
    # parallel so slow/timeout candidates do not serialize an entire GRPO step.
    def score(job):
        code, test_list, test_setup = job
        return score_solution(code, test_list, test_setup, timeout=3.0)

    workers = max(1, min(4, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fractions = list(pool.map(score, jobs))

    if _ACTIVE_GRPO_METRICS is not None:
        _ACTIVE_GRPO_METRICS.record(completions, fractions)

    return [3.0 * f for f in fractions]


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


def _curriculum_filter(cfg, model, tokenizer, dataset):
    """DeepSWE-style curriculum filter: drop zero-variance MBPP tasks.

    Cheap pre-pass over a small subsample of tasks: sample a few short
    completions per task and drop the ones that are trivially all-pass or
    all-fail for the current policy -- those give zero reward variance and
    therefore zero GRPO gradient, so this is rollout budget better spent on
    tasks that can actually teach the policy something. Tasks beyond the
    probed subsample are left untouched (this is a budget-cheap heuristic,
    not an exhaustive re-grading of the whole dataset).
    """
    if not cfg.train.grpo_curriculum:
        return dataset
    sample_n = min(len(dataset), cfg.train.grpo_curriculum_sample_size)
    if sample_n == 0:
        return dataset

    from agent.executor import extract_code, score_solution
    from unsloth import FastLanguageModel
    from datasets import concatenate_datasets
    import torch

    n_gen = max(1, cfg.train.grpo_curriculum_num_generations)
    probe = dataset.select(range(sample_n))
    keep_idx = []

    FastLanguageModel.for_inference(model)
    try:
        for i in range(sample_n):
            ex = probe[i]
            prompt_text = tokenizer.apply_chat_template(
                ex["prompt"], tokenize=False, add_generation_prompt=True)
            try:
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=cfg.train.grpo_curriculum_max_new_tokens,
                        do_sample=True, temperature=1.0,
                        num_return_sequences=n_gen,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                scores = []
                for row in gen:
                    text = tokenizer.decode(
                        row[prompt_len:], skip_special_tokens=True)
                    code = extract_code(text)
                    scores.append(score_solution(
                        code, ex["tests"], ex.get("setup", ""),
                        timeout=cfg.train.grpo_curriculum_timeout))
            except Exception as e:  # pragma: no cover - defensive, keep on error
                print(f"[grpo-curriculum] probe {i} failed ({e}); keeping task",
                      flush=True)
                keep_idx.append(i)
                continue
            all_pass = all(s >= 1.0 - 1e-6 for s in scores)
            all_fail = all(s <= 1e-6 for s in scores)
            if not (all_pass or all_fail):
                keep_idx.append(i)
    finally:
        FastLanguageModel.for_training(model)

    dropped = sample_n - len(keep_idx)
    kept_probe = probe.select(keep_idx)
    rest = (dataset.select(range(sample_n, len(dataset)))
            if len(dataset) > sample_n else None)
    filtered = concatenate_datasets([kept_probe, rest]) if rest is not None \
        else kept_probe
    print(f"[grpo-curriculum] probed {sample_n} tasks, dropped {dropped} "
          f"zero-variance, kept {len(kept_probe)}/{sample_n} probed "
          f"(+{len(rest) if rest is not None else 0} unprobed) -> "
          f"{len(filtered)} tasks total", flush=True)
    return filtered


def run_grpo(cfg, model, tokenizer, phase, dataset):
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
    import torch

    class RolloutMetricsCallback(TrainerCallback):
        """Lightweight HF TrainerCallback: times each step and reads peak
        VRAM, then asks the shared _GRPOMetrics (fed by correctness_reward)
        to flush one JSON metrics line per step. report_to stays "none" --
        this is a self-contained substitute, not a W&B/Comet integration.
        """

        def __init__(self, metrics):
            self.metrics = metrics
            self._t0 = None

        def on_step_begin(self, args, state, control, **kwargs):
            import time
            self._t0 = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            import time
            elapsed = (time.monotonic() - self._t0) if self._t0 else 0.0
            peak_vram_gb = 0.0
            if torch.cuda.is_available():
                peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats()
            self.metrics.flush(state.global_step, elapsed, peak_vram_gb)

    out = phase_dir(cfg, phase.name)
    bf16 = bool(torch.cuda.is_bf16_supported())

    if cfg.train.grpo_curriculum:
        dataset = _curriculum_filter(cfg, model, tokenizer, dataset)

    os.makedirs(cfg.train.output_root, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    global _ACTIVE_GRPO_METRICS
    _ACTIVE_GRPO_METRICS = _GRPOMetrics(
        rollout_path=os.path.join(cfg.train.output_root, "rollouts.jsonl"),
        metrics_path=os.path.join(out, "grpo_metrics.jsonl"),
        num_generations=cfg.train.grpo_num_generations,
        max_completion_length=cfg.train.grpo_max_completion_length,
    )
    grpo_config_kwargs = {
        "output_dir": out,
        "per_device_train_batch_size": phase.per_device_train_batch_size,
        "gradient_accumulation_steps": phase.gradient_accumulation_steps,
        "num_generations": cfg.train.grpo_num_generations,
        "max_prompt_length": cfg.train.grpo_max_prompt_length,
        "max_completion_length": cfg.train.grpo_max_completion_length,
        "max_steps": phase.max_steps,
        "learning_rate": phase.learning_rate,
        "lr_scheduler_type": phase.lr_scheduler_type,
        "warmup_ratio": phase.warmup_ratio,
        "logging_steps": phase.logging_steps,
        "save_steps": phase.save_steps,
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "max_grad_norm": 0.1,
        "seed": cfg.train.seed,
        "report_to": "none",
        "fp16": not bf16,
        "bf16": bf16,
        # DAPO stability/sample-efficiency components supported by recent TRL.
        "loss_type": "dapo",
        "epsilon_high": 0.28,
        "mask_truncated_completions": True,
        "log_completions": True,
    }
    grpo_args = GRPOConfig(**_supported_kwargs(GRPOConfig, grpo_config_kwargs))
    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "reward_funcs": [correctness_reward, format_reward],
        "train_dataset": dataset,
        "args": grpo_args,
        "callbacks": [RolloutMetricsCallback(_ACTIVE_GRPO_METRICS)],
    }
    trainer = GRPOTrainer(**_supported_kwargs(GRPOTrainer, trainer_kwargs))
    resume = os.path.isdir(out) and any(
        d.startswith("checkpoint-") for d in os.listdir(out))
    try:
        trainer.train(resume_from_checkpoint=resume)
    finally:
        _ACTIVE_GRPO_METRICS = None
    if _is_main_process():
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"[grpo] {phase.name} done -> {out}", flush=True)
    _distributed_barrier()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
PHASE_CFG = {
    "reasoning_sft": lambda c: c.train.reasoning_sft,
    "repair_sft":    lambda c: c.train.repair_sft,
    "general_sft":   lambda c: c.train.general_sft,
    "grpo":          lambda c: c.train.grpo,
}
PHASE_MAX = {
    "reasoning_sft": lambda c: c.data.reasoning_sft_max_samples,
    "repair_sft":    lambda c: c.data.repair_sft_max_samples,
    "general_sft":   lambda c: c.data.general_sft_max_samples,
    "grpo":          lambda c: c.data.grpo_max_samples,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+",
                    default=["reasoning_sft", "repair_sft", "general_sft", "grpo"])
    args = ap.parse_args()

    cfg = CONFIG
    print("[config]\n" + cfg.dump(), flush=True)
    assert_gpu()
    os.makedirs(cfg.train.output_root, exist_ok=True)

    state = load_state(cfg)
    state_model = state.get("model_name")
    if state_model and state_model != cfg.model.name:
        raise RuntimeError(
            f"Checkpoint model mismatch: state has {state_model!r}, "
            f"config requests {cfg.model.name!r}")
    completed = set(state.get("completed", []))

    # Resume from the most recently completed phase's adapter, if any.
    resume_from = None
    for ph in ["grpo", "general_sft", "repair_sft", "reasoning_sft"]:
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
        if _is_main_process():
            save_state(cfg, state)
        _distributed_barrier()

    print("\n[done] pipeline finished:", sorted(completed), flush=True)


if __name__ == "__main__":
    main()
