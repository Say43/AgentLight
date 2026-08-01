---
base_model: meta-llama/Llama-3.2-3B-Instruct
license: llama3.2
library_name: peft
tags:
  - lora
  - grpo
  - code
  - reasoning
datasets:
  - open-thoughts/OpenThoughts-114k
  - HuggingFaceTB/smoltalk
  - google-research-datasets/mbpp
---

# Llama-AgentLight

**Built with Llama**

A LoRA adapter that turns Llama 3.2 3B Instruct into a reasoning, tool-using
coding agent, trained on a single free-tier Kaggle GPU budget. Produced by the
[AgentLight](https://github.com/Say43/AgentLight) pipeline:
reasoning-SFT → repair-SFT → general-SFT → GRPO with verifiable rewards.

> **Use this file as the model card whenever the adapter is published.** The
> name must begin with "Llama" and the notices below must stay intact — both
> are conditions of the Llama 3.2 Community License, not stylistic choices.

## Model details

- **Base model:** `meta-llama/Llama-3.2-3B-Instruct` (loaded via
  `unsloth/Llama-3.2-3B-Instruct`, 4-bit QLoRA)
- **Adapter:** LoRA, r=16, alpha=16, applied to the attention and MLP
  projections
- **Context length:** 2048 tokens
- **What is distributed here:** the LoRA adapter only. Meta's base weights are
  not included; they are downloaded from their official source.

## Intended use

Generating and repairing short Python programs, with step-by-step reasoning
and an execute-observe-revise loop. It is a focused code assistant, not a
general-purpose one.

## Limitations

- 3B parameters: it fails on problems requiring long multi-step reasoning or
  large-context understanding.
- Trained and evaluated on short, self-contained function-level tasks (MBPP,
  HumanEval). Behaviour on real codebases is untested.
- **Generated code is executed during evaluation and in the agent loop.** Run
  it in a sandbox. Model-written code should never be executed unreviewed on a
  machine you care about.
- Inherits the biases and failure modes of the base model and its training
  data.

## Evaluation

<!-- Fill in from src/eval_code.py before publishing; do not ship placeholder
     numbers. Report base vs. adapter under identical settings. -->

| Setting | pass@1 |
|---|---|
| Base model, non-agentic | _to be filled in_ |
| Llama-AgentLight, non-agentic | _to be filled in_ |
| Llama-AgentLight, agentic (ReAct) | _to be filled in_ |

Measured with `src/eval_code.py` on HumanEval.

## Training data

| Dataset | Phase | Licence |
|---|---|---|
| `open-thoughts/OpenThoughts-114k` (code subset) | Reasoning SFT | Apache-2.0 |
| `HuggingFaceTB/smoltalk` | General SFT | Apache-2.0 |
| `google-research-datasets/mbpp` | Repair SFT + GRPO reward | CC BY 4.0 |

No data distilled from OpenAI, Anthropic, or Google models was used for
training.

## Licence and attribution

This adapter is a derivative work of Meta Llama 3.2 and is licensed under the
**Llama 3.2 Community License Agreement**. A copy accompanies the
[AgentLight repository](https://github.com/Say43/AgentLight); the Acceptable
Use Policy applies to all use.

> Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright ©
> Meta Platforms, Inc. All Rights Reserved.

**Built with Llama.**

This model was trained using MBPP, © Google Research, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), from "Program
Synthesis with Large Language Models", Austin et al., 2021
([arXiv:2108.07732](https://arxiv.org/abs/2108.07732)).

The AgentLight training code is separately MIT-licensed. Full third-party
attribution: `THIRD_PARTY_NOTICES.md` and `NOTICE` in the repository.

## Citation

```bibtex
@misc{llama-agentlight,
  title  = {Llama-AgentLight: a reasoning code agent trained on a free-tier GPU budget},
  author = {Say43},
  year   = {2026},
  url    = {https://github.com/Say43/AgentLight}
}
```
