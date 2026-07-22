# checkpoints/

Trained artifacts live here. **Weights are gitignored** — LoRA adapters,
optimizer state, and any merged/GGUF exports are never committed (large, and
the base weights they derive from are license-encumbered; see
`../THIRD_PARTY_NOTICES.md`).

Layout after a run:

```text
checkpoints/
  pipeline_state.json     Which phases have completed (drives resume)
  reasoning_sft/          Adapter + HF Trainer checkpoints for phase 1
  general_sft/            ... phase 2
  grpo/                   ... phase 3 (the final model to evaluate / ship)
```

## Resuming across Kaggle sessions

A full run doesn't fit one ~12h session. To continue next session:

1. The previous kernel saved `checkpoints/` under `/kaggle/working` — Kaggle
   keeps that as the kernel **output**.
2. Add that output as a **dataset source** on the next run (edit
   `kaggle/kernel-metadata.json` `dataset_sources`, or attach it in the UI).
   Kaggle mounts it read-only under `/kaggle/input/<dataset-slug>/...` — the
   exact subpath depends on the dataset's slug, which varies run to run.
3. Re-push. Before launching training, `kaggle/run.py`'s `restore_checkpoints()`
   walks `/kaggle/input/` looking for the marker file `pipeline_state.json`
   (rather than assuming a fixed path), and `shutil.copytree`s that whole
   directory's contents into `AGENTLIGHT_OUT` (`checkpoints/` under
   `/kaggle/working`, same layout as above), logging what it restored. If no
   such dataset is attached, it logs "no prior checkpoints — fresh run" and
   proceeds normally.
4. `src/train.py` then reads `pipeline_state.json`, skips completed phases,
   and loads the latest adapter to continue.

## Shipping the adapter

Only the `grpo/` adapter is the shippable artifact. If you publish it, include:
the Llama Community License, a "Built with Llama" notice, and attribution for
MBPP (CC-BY-4.0). Never upload the base weights.
