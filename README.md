# AgentLight

A **coding agent** built by fine-tuning a vanilla open-weight Llama into a
reasoning + tool-using model — on a hobbyist Kaggle GPU budget.

Unlike a from-scratch project, AgentLight starts from a strong pretrained base
(Llama 3.2 3B Instruct) and *teaches it to reason and act* through a three-stage
pipeline. The showcase task is verifiable by construction: **given a
programming problem, the model reasons step by step, writes Python, runs it
against unit tests, and revises until the tests pass.**

It's a standalone sibling to the earlier `GPTlight` project (which trained a
97M model from scratch). AgentLight reuses GPTlight's hard-won Kaggle lessons
(see [docs/PLAN.md](docs/PLAN.md)) but shares no code — it's self-contained.

> **Why a small model can still be a real showcase:** the base 3B model already
> far outperforms a from-scratch model on every benchmark. Our contribution is
> the *pipeline* — reasoning-SFT → SFT → GRPO — and a measurable pass@1 lift on
> HumanEval from acting on verifiable rewards. It is a focused code agent, not a
> general far-reaching assistant; that honesty is the point.

## Pipeline

1. **Reasoning-SFT** — teach the `<think>…</think>` long-chain-of-thought
   format on R1-distilled reasoning traces (`open-thoughts/OpenThoughts-114k`,
   Apache-2.0).
2. **General-SFT** — keep it a usable assistant (`HuggingFaceTB/smoltalk`,
   Apache-2.0), assistant-only loss.
3. **GRPO** — reinforcement learning with a **verifiable reward**: generate
   code, execute it against MBPP unit tests, reward = fraction of tests passed.
   This is the RLVR recipe that made reasoning models strong at code.

Then an inference-time **ReAct loop** ([agent/react_agent.py](agent/react_agent.py))
turns generation into agency: execute → observe failure → revise.

## Repository layout

```text
config/config.py        Single source of truth (model, data, hyperparameters)
data/prepare_data.py    License-clean dataset loading & formatting
src/train.py            Pipeline: reasoning-SFT → general-SFT → GRPO (resumable)
src/eval_code.py        HumanEval pass@1 — the before/after showcase metric
agent/executor.py       Sandboxed code execution (GRPO reward + agent tool)
agent/react_agent.py    ReAct execute/retry coding agent
chat/local_chat.py      Local CLI chat against a trained adapter
kaggle/run.py           Kaggle entrypoint (install + sync repo + train)
kaggle/kernel-metadata.json
docs/PLAN.md            The 16h Kaggle plan, phase budget, and GPTlight lessons
THIRD_PARTY_NOTICES.md  Model + dataset licenses and compliance obligations
```

## Running it

**Train on Kaggle** (push from the repo root with the Kaggle CLI):
```
kaggle kernels push -p kaggle
```
Set the accelerator to **GPU T4 x2**. The run resumes automatically across
sessions — see [docs/PLAN.md](docs/PLAN.md) and [checkpoints/README.md](checkpoints/README.md).

**Evaluate** (the RESULTS story):
```
python src/eval_code.py --adapter checkpoints/grpo --n 60
python src/eval_code.py --adapter checkpoints/grpo --n 60 --agentic
```

**Chat / solve locally:**
```
pip install -r requirements.txt
python chat/local_chat.py --adapter checkpoints/grpo
```

## License

Code is MIT ([LICENSE](LICENSE)). The base model (Llama Community License) and
datasets carry their own terms and obligations — all documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Built with Llama.
