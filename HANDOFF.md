# AgentLight run handoff

Updated: 2026-07-21

## Current status

The local code-path blockers found in the pre-run review are addressed in the
working tree, but the GPU stack has **not yet been validated on Kaggle**. Do not
start the full 16-hour run before the smoke kernel succeeds.

Locally verified:

- all Python files compile;
- all four offline regression tests in `tests/test_core.py` pass;
- the reward self-test gives 1.0 for correct code and 0.0 for wrong code and a
  forged `SCORE=...`/early-exit attempt;
- `AGENTLIGHT_SMOKE=1` reduces all datasets, steps and completion length;
- `WORLD_SIZE=2` halves gradient accumulation and preserves the global batch;
- Kaggle metadata now points to `run.py`;
- smoke outputs use `smoke_checkpoints/`, separate from real resumable state.

Still requires external validation:

- `https://github.com/Say43/AgentLight.git` currently returns "Repository not
  found" without credentials, and this local repository has no remote. Kaggle
  cannot clone the project until a reachable repository is created/pushed.
- The pinned Kaggle candidate stack (`unsloth==2026.7.4`,
  `unsloth_zoo==2026.7.3`, `trl==1.8.0`) must pass the exact T4 x2 smoke run.
- No full dataset download, CUDA model load, SFT step or GRPO step has been run
  in the local Windows environment.

## Required order before the real run

1. Create/push the repository at the URL in `kaggle/run.py`, branch `main`, or
   change `REPO_URL` to the actual reachable repository.
2. Keep the already-safe `SMOKE = True` setting in `kaggle/run.py`, commit and
   push the current working tree.
3. Push the Kaggle kernel with `kaggle kernels push -p kaggle`, select **T4 x2**
   and keep internet enabled.
4. Require all three phases to finish. Check the log for `pip check`, two
   torchrun ranks, non-empty post-length-filter datasets, finite losses, GRPO
   reward metrics, and saved adapters under `smoke_checkpoints/`.
5. If smoke succeeds, set `SMOKE = False`, commit/push, then start the real run.
6. Do not attach smoke output as resume data. For a second real session, attach
   only the previous real `checkpoints/` kernel output; `restore_checkpoints()`
   locates and restores its `pipeline_state.json` automatically.
7. Run full final evaluation with all 164 HumanEval tasks. Treat `--agentic`
   and `--tts` as public-doctest-assisted modes; hidden checks are final grading
   only.

## Stop conditions during GRPO

Stop rather than spending the remaining quota when rewards are non-finite,
almost every group has zero reward variance, completion truncation stays high,
or a held-out evaluation after the pilot slice does not improve over the SFT
checkpoint.
