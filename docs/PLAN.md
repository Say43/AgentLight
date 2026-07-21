# AgentLight — the plan

## Goal

Turn a **vanilla** Llama 3.2 3B Instruct (deliberately *not* a model already
trained for reasoning) into a **coding agent**: it reasons in the open, writes
Python, executes it against tests, and revises until they pass. The whole thing
must fit a hobbyist Kaggle budget and stay legally clean.

## Showcase task & how we prove it

**Task:** solve programming problems with visible reasoning + tool use (code
execution). **Proof:** HumanEval pass@1, measured identically on each
checkpoint — the same before/after methodology GPTlight used.

| Checkpoint | What it shows |
|---|---|
| base (Llama 3.2 3B Instruct) | starting point |
| + reasoning-SFT | does it now reason before coding? |
| + general-SFT | still a usable assistant? |
| + GRPO | **the headline number** — RL on verifiable rewards |
| GRPO + ReAct (`--agentic`) | lift from *acting* (execute/retry) vs. single-shot |

Honest expectation: measurable, demoable gains on basic-to-medium Python
(MBPP/HumanEval level). Not a general agent — a focused, verifiable one.

## The 16h Kaggle budget

Kaggle caps a single GPU **session at ~12h** and the **weekly quota** is what
you have left (~16h). So the run is designed to **span ≥2 sessions** and resume
automatically. `max_steps` per phase is the knob to fit a session.

| Phase | max_steps | eff. batch | rough T4 time | why this size |
|---|---|---|---|---|
| reasoning-SFT | 1200 | 16 | ~2.5–4h | enough to lock in the `<think>` format |
| general-SFT | 900 | 16 | ~1.5–3h | keep assistant ability, avoid over-specializing |
| GRPO | 400 | 4 | ~4–7h | rollouts dominate; this is the expensive part |
| eval + buffer | — | — | ~1–2h | HumanEval before/after, install overhead |

**Total ≈ 9–16h.** GRPO is intentionally the biggest slice because it's where
the verifiable-reward payoff is. If a session gets cut, the next one resumes at
the unfinished phase. First session target: finish both SFT phases + start
GRPO. Second session: finish GRPO + eval.

Fit-the-session tactics (all in `config/config.py`):
- Lower a phase's `max_steps` or `*_max_samples` to finish sooner.
- GRPO `num_generations` (4) and `max_completion_length` (1024) are the main
  GRPO cost levers — halve them if GRPO is too slow.

## Lessons carried over from GPTlight (and how AgentLight fixes them)

GPTlight's `RESULTS.md` documented four Kaggle failures that cost real quota.
Each is designed out here:

1. **Silent data mix-up** (a glob picked the wrong `.bin`). →
   `data/prepare_data.py` references every dataset by **explicit HF name**;
   no globbing. `src/train.py` logs the example count per phase.
2. **AMP scaler state not checkpointed** (rough transitions after resume). →
   we use HF `Trainer` checkpointing (`resume_from_checkpoint`), which saves
   optimizer + scheduler + scaler together; no hand-rolled scaler handling.
3. **Stale iteration counter across phases** (a resume could re-run a finished
   phase). → an **atomic** `pipeline_state.json` records completed phases;
   `src/train.py` skips them on resume.
4. **Silent CPU fallback on a wrong GPU assignment** (hours wasted). →
   `assert_gpu()` refuses to run without CUDA and logs the device name; a
   P100-vs-2xT4 surprise fails **loudly at startup**.

## Legal / license posture (hard requirement)

- Base weights are **downloaded at runtime**, never redistributed. We only ship
  our own LoRA adapter, labeled "Built with Llama".
- **No OpenAI/Anthropic/Google-model-distilled data** in training — only
  DeepSeek-R1-distilled (MIT, distillation explicitly allowed) and
  openly-licensed sources. See `THIRD_PARTY_NOTICES.md`.
- MBPP requires attribution; HumanEval is eval-only (MIT). Both documented.

## Roadmap beyond the first 16h

In rough priority order once the pipeline is validated:

1. **Scale the base to Llama 3.1 8B** (one line: `MODEL.name`) for the SFT
   phases when more quota is available — 8B GRPO still needs a bigger GPU.
2. **More GRPO steps / harder problems** (add `open-r1/codeforces-cots` or
   LiveCodeBench-style tasks) once basic MBPP is saturated.
3. **Tool-use SFT phase** — train the ReAct execute/observe format explicitly,
   not just at inference, for more reliable multi-step agent behavior.
4. **Better eval** — MBPP test split + pass@k with multiple samples, seed
   averaging (GPTlight noted single-run numbers are a snapshot, not a CI).
