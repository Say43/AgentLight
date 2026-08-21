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

## Verified local snapshot (2026-08-21)

The private Kaggle kernel `says43/agentlight-train` completed GRPO at the full
configured budget (`global_step = max_steps = 200`). Its final artifacts were
merged into this local directory, including `checkpoint-150`, `checkpoint-200`,
`grpo_metrics.jsonl`, completion parquets through step 200, `rollouts.jsonl`,
and the corrected `pipeline_state.json` marking every phase complete.

The local top-level `grpo/adapter_model.safetensors` was already byte-identical
to the Kaggle step-200 adapter (SHA-256
`C8E997297A3DA6159A4C346C0FDC02678BA5B9DA9A515CCA28B135CB7AE7DE59`).
The former `grpo_v14` directory contains that same adapter, not a separate model.
Use `grpo/` as the one canonical evaluation and shipping path.

The 200 logged GRPO steps took 7,607.89 seconds (2.113 hours). These training
metrics confirm completion, but they are not a substitute for held-out
HumanEval evaluation; no performance claim should be published until the model
card's evaluation table is filled with real results.

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

Only the `grpo/` adapter is the shippable artifact. Publishing it is a
*distribution* of a Llama derivative, which triggers obligations that do not
apply while the weights stay local. Before uploading anywhere — Hugging Face,
Kaggle, or a release asset — do all of the following:

1. **Name it `Llama-AgentLight`.** The Llama 3.2 Community License requires a
   distributed derivative's name to begin with "Llama". This is not optional
   and not cosmetic.
2. **Publish [`../MODEL_CARD.md`](../MODEL_CARD.md) as the model card.** It
   already carries "Built with Llama", the Meta attribution string, and the
   MBPP CC BY 4.0 attribution. Fill in the evaluation numbers first.
3. **Include [`../NOTICE`](../NOTICE) and
   [`../licenses/LLAMA_3.2_COMMUNITY_LICENSE.txt`](../licenses/) alongside the
   weights.** The licence requires a copy to travel with the derivative.
4. **Never upload Meta's base weights** — only your adapter.

The reasoning behind each point is in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Note that a *private* Kaggle dataset used to resume training is not a
distribution. These obligations attach when you make the adapter available to
others.
