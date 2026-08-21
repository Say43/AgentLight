# AgentLight

**Built with Llama**

AgentLight is an experimental training and evaluation project for adapting
Llama 3.2 3B Instruct to short Python code-generation tasks. The repository
implements a resumable four-phase QLoRA pipeline and a small ReAct-style
execute/observe/revise loop for inference.

The training pipeline has completed. Model quality has not yet been established
with the planned held-out HumanEval comparison, so this repository currently
documents a completed training run—not a demonstrated benchmark improvement.

## Project status

| Area | Status |
|---|---|
| Reasoning-SFT | Complete |
| Repair-SFT | Complete |
| General-SFT replay | Complete |
| GRPO | Complete, 200/200 steps |
| Final adapter | Available locally at `checkpoints/grpo/` |
| HumanEval comparison | Pending |
| Public adapter release | Not published |

Completion was verified from the Kaggle kernel output, the final
`trainer_state.json`, 200 GRPO metric records, rollout snapshots through step
200, and cryptographic hashes of the saved adapters. The pipeline state now
marks all four phases as complete.

## Scope

The project tests a specific engineering question: can a small instruction
model be adapted into a more reliable function-level Python assistant using
supervised reasoning examples, explicit repair trajectories, and executable
rewards within a single-T4 budget?

It is not intended to be a general-purpose coding model or a production agent.
The target workload is short, self-contained Python functions with executable
tests. Repository-scale editing, long-context reasoning, and secure arbitrary
code execution are outside the current scope.

## Training pipeline

The phases run sequentially and reuse the same LoRA adapter:

| Phase | Data | Configured budget | Purpose |
|---|---|---:|---|
| Reasoning-SFT | `open-thoughts/OpenThoughts-114k`, code subset | 12,000 rows / 600 steps | Learn structured solution traces and code output |
| Repair-SFT | `google-research-datasets/mbpp` | 300 rows / 200 steps | Correct a buggy attempt from a failing test trace |
| General-SFT | `HuggingFaceTB/smoltalk` | 4,000 rows / 250 steps | Replay general instruction-following behavior |
| GRPO | `google-research-datasets/mbpp` | 374 tasks / 200 steps | Optimize executable correctness and output format |

The base model is loaded in 4-bit and trained with LoRA (`r=16`, `alpha=16`)
over the attention and MLP projection layers. GRPO generates four candidates
per prompt and scores generated Python against tests not exposed in the prompt.
A curriculum pre-pass removes trivially all-pass or all-fail examples from a
small probe subset before the main GRPO run.

The complete configuration is defined in [config/config.py](config/config.py).
Training and phase-resume logic live in [src/train.py](src/train.py).

## Observed GRPO training results

The final GRPO run produced the following telemetry:

| Metric | Observed value |
|---|---:|
| Completed steps | 200 / 200 |
| Logged step time | 7,607.89 s (2.113 h) |
| Mean time per step | 38.04 s |
| Mean raw correctness reward, first 20 steps | 0.24375 |
| Mean raw correctness reward, last 20 steps | 0.35000 |
| Mean raw correctness reward, full run | 0.44188 |
| Steps with zero reward variance | 89 / 200 |
| Non-finite reward rows | 0 |
| Mean truncation rate, last 20 steps | 0 |
| Peak logged VRAM allocation | 3.776 GB |

The late-run mean reward is higher than the first-20-step mean, and the adapter
changed between checkpoints 150 and 200. This confirms that optimization
continued through the final checkpoint. It is not sufficient evidence that the
adapter generalizes better: rewards are noisy, 44.5% of steps had no within-group
reward variance, and the metric is measured on the training objective.

The canonical final adapter has SHA-256:

```text
C8E997297A3DA6159A4C346C0FDC02678BA5B9DA9A515CCA28B135CB7AE7DE59
```

Run provenance, checkpoint comparisons, and the handoff state are documented in
[HANDOFF.md](HANDOFF.md). Model weights and trainer state are intentionally
gitignored and are not stored in this repository.

## Evaluation status

The planned benchmark is HumanEval pass@1 under identical settings for the base
model and trained adapter. Final numbers have not been generated yet. Until that
comparison is complete, no claim is made that AgentLight improves on Llama 3.2
3B Instruct.

Run the full 164-task evaluation with:

```bash
# Base-model baseline
python src/eval_code.py \
  --adapter unsloth/Llama-3.2-3B-Instruct \
  --n 164

# Final adapter, single shot
python src/eval_code.py \
  --adapter checkpoints/grpo \
  --n 164

# Final adapter with execute/observe/revise
python src/eval_code.py \
  --adapter checkpoints/grpo \
  --n 164 \
  --agentic

# Test-time scaling with four candidates
python src/eval_code.py \
  --adapter checkpoints/grpo \
  --n 164 \
  --tts \
  --k 4
```

The resulting base-versus-adapter table should be added to
[MODEL_CARD.md](MODEL_CARD.md) before publishing the adapter.

## Local inference

An NVIDIA CUDA environment is recommended because the adapter references a
4-bit base model. Install the inference dependencies and start either client:

```bash
pip install -r requirements.txt

# Minimal terminal chat
python chat/local_chat.py --adapter checkpoints/grpo

# Rich terminal UI
python chat/chat_ui.py --adapter checkpoints/grpo
```

For a single coding task with the ReAct loop:

```bash
python agent/react_agent.py \
  --adapter checkpoints/grpo \
  --problem "Implement a function that returns the prime factors of an integer."
```

## Training and resume behavior

The Kaggle entry point installs the pinned CUDA training stack, restores a prior
checkpoint dataset when present, and writes checkpoints under Kaggle's working
directory:

```bash
kaggle kernels push -p kaggle
```

The checked-in kernel is private and configured for an NVIDIA T4. Its dataset
source (`says43/agentlight-sft-ckpt`) is also private, so other users must remove
that source and train from scratch or attach their own compatible checkpoint
dataset.

Each completed phase is recorded atomically in `pipeline_state.json`. Re-running
the trainer skips completed phases and resumes from the newest completed adapter:

```bash
python src/train.py --phases reasoning_sft repair_sft general_sft grpo
```

Set `AGENTLIGHT_SMOKE=1` for a short end-to-end code-path check before allocating
a full training run.

## Repository structure

```text
config/config.py        Model, data, phase, and GRPO configuration
data/prepare_data.py    Dataset loading, filtering, and prompt formatting
src/train.py            Resumable SFT and GRPO training pipeline
src/eval_code.py        HumanEval single-shot, agentic, and TTS evaluation
agent/executor.py       Code extraction, subprocess execution, and scoring
agent/react_agent.py    Execute/observe/revise inference loop
chat/local_chat.py      Minimal terminal chat
chat/chat_ui.py         Rich terminal chat interface
kaggle/run.py           Kaggle environment setup and checkpoint restoration
checkpoints/README.md   Artifact layout, resume process, and release rules
MODEL_CARD.md           Adapter card; benchmark table is still pending
HANDOFF.md              Verified run state and remaining work
```

## Known limitations

- No held-out HumanEval result has been published yet.
- The 3B base model is limited on long or multi-file tasks.
- GRPO used only 200 steps and 89 steps had zero reward variance.
- A previous chat prompt triggered repetitive literal “Think” output. The prompt
  was removed, but the final adapter still requires systematic generation QA.
- The ReAct loop improves retry behavior but does not provide a security
  boundary.
- Generated code is executed during reward calculation and evaluation. Use an
  isolated disposable environment; do not run untrusted generations on a
  machine containing sensitive data.
- Adapter weights are not yet distributed publicly.

## License and attribution

The repository code is MIT-licensed. The trained adapter is a derivative of
Llama 3.2 and is subject to the Llama 3.2 Community License. If distributed, it
must be named **Llama-AgentLight**, include the required Llama license and
notices, and contain only the adapter—not Meta's base weights.

MBPP is licensed under CC BY 4.0 and requires attribution. Dataset, model, and
redistribution details are documented in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[checkpoints/README.md](checkpoints/README.md).

> Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright ©
> Meta Platforms, Inc. All Rights Reserved.
