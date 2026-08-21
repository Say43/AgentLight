# AgentLight — handoff

Updated: 2026-08-21

The training-state ambiguity is resolved. The private Kaggle kernel
`says43/agentlight-train` completed successfully, including GRPO at the configured
200 steps. Do **not** launch another GRPO run merely because the old local
`pipeline_state.json` omitted `grpo`; that marker was stale.

## 1. Verified training state

The latest Kaggle output was downloaded and inspected directly. It contains:

- `checkpoints/grpo/checkpoint-150/` and `checkpoint-200/`, including optimizer,
  scheduler, RNG, scaler, trainer state, tokenizer and adapter files;
- a final top-level `checkpoints/grpo/` adapter and tokenizer;
- 200 rows in `grpo_metrics.jsonl` and completion snapshots through
  `completions_00200.parquet`;
- `pipeline_state.json` listing all four phases as complete: `reasoning_sft`,
  `repair_sft`, `general_sft`, and `grpo`.

`checkpoint-200/trainer_state.json` is conclusive:

```text
global_step: 200
max_steps:   200
epoch:       0.5847953216374269
```

The final adapter was not actually missing locally. Its SHA-256 is identical in
all of these locations:

```text
C8E997297A3DA6159A4C346C0FDC02678BA5B9DA9A515CCA28B135CB7AE7DE59
  local checkpoints/grpo/adapter_model.safetensors
  local checkpoints/grpo_v14/adapter_model.safetensors
  Kaggle checkpoints/grpo/adapter_model.safetensors
  Kaggle checkpoints/grpo/checkpoint-200/adapter_model.safetensors
```

Therefore `grpo_v14` is not an alternative model and `grpo/` is not a 100-step
adapter. Both local adapter files were already the final step-200 export; only
the local evidence/resume files and completion marker were outdated. The latest
Kaggle artifacts have now been merged additively into local `checkpoints/`.
Older checkpoint-50/100 directories are retained as history.

## 2. Measured GRPO run

The run took **7,607.89 logged seconds = 126.80 minutes = 2.113 hours** across
200 steps, averaging 38.04 seconds per step. This supersedes the old planning
estimate of 5–6 hours.

Basic telemetry:

| Signal | Result |
|---|---:|
| Logged steps | 200 (1 through 200) |
| Mean reward, first 20 steps | 0.24375 |
| Mean reward, last 20 steps | 0.35000 |
| Mean reward, full run | 0.44188 |
| All-zero-variance steps | 89 / 200 |
| Non-finite reward rows | 0 |
| Mean truncation, last 20 steps | 0 |
| Peak logged VRAM | 3.776 GB |

This proves that GRPO ran and updated the adapter between checkpoint 150 and
200 (their hashes differ). It does **not** prove that the final adapter improves
held-out HumanEval performance. The reward trend is mildly positive, but 89
zero-variance steps are inefficient and the previously observed “Think”
repetition remains a checkpoint-quality warning until evaluation.

## 3. Current working-tree changes

Preserve and review the existing uncommitted edits:

- `agent/executor.py` — code extraction now prefers the first fenced block that
  actually defines a function;
- `chat/chat_ui.py` — no longer prompts the known-degenerate adapter to emit
  literal `<think>` text;
- `data/prepare_data.py` — lazy-imports `datasets` for faster inference startup;
- `kaggle/kernel-metadata.json` — attaches the private checkpoint dataset;
- `HANDOFF.md` and `checkpoints/README.md` — record the verified completed run.

Do not discard unrelated user changes. Run the tests before committing these
files, and commit only after reviewing the final diff.

## 4. Remaining work for a publishable portfolio project

Training is complete. The critical path is now evaluation and reporting:

1. Sanity-test `checkpoints/grpo/` interactively for repetition or malformed
   answers. `grpo_v14` need not be tested separately because it is byte-identical.
2. Run `src/eval_code.py` with the same full 164-task HumanEval setup for the
   base model, `reasoning_sft`, `general_sft`, and final `grpo` adapter.
3. Evaluate the final adapter both single-shot and agentically; also run the
   configured test-time-scaling variant (`--tts --k 4`).
4. Fill the placeholder table in `MODEL_CARD.md` with real results and update
   the README status. Do not claim improvement before these held-out numbers
   exist.
5. If the adapter is published, follow `checkpoints/README.md`: name it
   `Llama-AgentLight`, ship the model card, `NOTICE`, and the Llama license, and
   never redistribute Meta base weights.

The repository remote is `https://github.com/Say43/AgentLight.git`. Model
artifacts under `checkpoints/` are intentionally gitignored, so the tracked
handoff and checkpoint README are the durable record of this verification.
