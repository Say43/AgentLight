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

## The 16h Kaggle budget (measured plan — released in stages, not up front)

Kaggle caps a single GPU **session at ~12h** and the **weekly quota** is what
you have left (~16h). So the run is designed to **span ≥2 sessions** and resume
automatically. Unlike the original all-at-once step table, GRPO's step count
is **not fixed in advance** — it's set from a measured pilot, because GRPO
rollout cost on a T4 is too variable to guess correctly up front (a bad guess
either burns quota on unproductive extra steps or leaves easy gains on the
table).

| Stage | Budget | What happens |
|---|---|---|
| Baseline eval + smoke | 30–60 min | HumanEval pass@1 on the vanilla base model; then a **10–20-step full smoke test** (`SMOKE = True` in `kaggle/run.py`) exercising every phase end-to-end to catch install/API/data breakage before it costs real quota |
| Code reasoning + light general replay | 4–5h | 600 reasoning-SFT steps followed by 250 lower-LR general replay steps; overlong rows are removed before training |
| GRPO (pilot-gated) | 5–6h | **only 150–250 steps**, 4 rollouts (`num_generations`), **max 384–512 completion tokens** (`max_completion_length`) — deliberately smaller than earlier drafts assumed, because rollouts dominate T4 GRPO cost |
| Full eval + export + reserve | 2h | HumanEval before/after on every checkpoint, adapter export, and slack for session overhead / retries |

**Total ≈ 12–14h**, leaving headroom inside the ~16h weekly quota.

Both T4s are used through `torchrun --nproc_per_node=2`. The configuration
halves gradient accumulation under `WORLD_SIZE=2`, keeping the intended global
batch unchanged. This exact DDP path must pass the smoke kernel before the
full run; falling back to one GPU changes the wall-time estimate materially.

**The GRPO step count is released only after the pilot.** Concretely: run
GRPO for a short measured slice first (e.g. 20–30 steps) and check two
things — reward variance is actually decreasing, and held-out eval
(`src/eval_code.py`) actually improves over the SFT checkpoint. Only if
**both** hold do you release the remaining 150–250-step budget; if either is
flat or noisy, stop and debug the reward function / data mix rather than
spending more rollout-hours hoping it self-corrects. Any time left over after
full eval + export goes back into more GRPO steps **only** under that same
condition, not by default.

Fit-the-session tactics (all in `config/config.py`):
- Lower a phase's `max_steps` or `*_max_samples` to finish sooner.
- GRPO `num_generations` (4) and `max_completion_length` (384–512) are the
  main GRPO cost levers — this is where to cut first if a pilot slice runs
  slower than expected.

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
