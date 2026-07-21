# Third-Party Licenses & Compliance

AgentLight's own code is MIT-licensed (see `LICENSE`). The **base model** and
**training datasets** are third-party works under their own licenses. This
file documents them and the obligations they impose, so the project stays
legal, freely usable, and license-compliant.

> **Re-verify before use.** Licenses on Hugging Face can change. Open each
> dataset/model page and confirm the license *at the time you run training*.
> The notes below reflect the state at project setup.

---

## Base model — Llama 3.2 (default) / Llama 3.1 (8B option)

- **License:** Llama 3.2 Community License Agreement (and Llama 3.1 for the
  8B option). Free to use, including commercially, **subject to its terms.**
- **Obligations we must meet (and do):**
  1. **Attribution / naming.** Any distributed derivative must be named
     beginning with **"Llama"** and must display **"Built with Llama"**.
     → AgentLight is a *fine-tuned adapter on top of Llama*; distributed
       artifacts are labeled "Llama-AgentLight" and carry "Built with Llama".
  2. **Include the license.** Ship a copy of the Llama Community License with
     any distributed model. → `licenses/LLAMA_LICENSE.txt` (add before you
     distribute weights).
  3. **Acceptable Use Policy** must be followed.
  4. **>700M MAU clause:** if your product exceeds 700M monthly active users,
     you must request a separate license from Meta. (Not a concern here.)
- **Weights are NOT redistributed in this repo** — the notebook downloads them
  from the official Unsloth/Meta repositories at runtime. We only ever
  distribute the LoRA adapter (small, our own training), never Meta's weights.

> If you want a base model with a *simpler* license (Apache-2.0, no naming or
> MAU clauses), Qwen2.5-7B or an OLMo model are drop-in alternatives — set
> `MODEL.name` in `config/config.py`. We use Llama per project decision.

---

## Datasets

| Dataset | Used for | Stated license | Notes |
|---|---|---|---|
| `open-thoughts/OpenThoughts-114k` | Reasoning SFT | Apache-2.0 | R1-distilled long-CoT (code + math). DeepSeek-R1 is MIT and **explicitly permits distillation**, so training on its outputs is allowed. |
| `HuggingFaceTB/smoltalk` | General SFT | Apache-2.0 | Broad instruction/chat — keeps the model a usable assistant. |
| `google-research-datasets/mbpp` | GRPO (verifiable reward) | CC-BY-4.0 | Python tasks with executable `test_list`. **Attribution required** on any distributed derivative. Reward = fraction of unit tests passed. |
| `openai/openai_humaneval` | Evaluation only | MIT | Human-written code benchmark (NOT model outputs). Used to measure pass@1 before/after — not for training. |

### The one rule we deliberately follow

**No data distilled from OpenAI/Anthropic/Google models is used for training.**
Those providers' terms of service prohibit using their outputs to train
competing models. Our reasoning traces come from **DeepSeek-R1 (MIT, distill
explicitly allowed)** or human/openly-licensed sources only. This is the
single most common license trap in reasoning-model fine-tuning, and we avoid
it by construction.

---

## What this repository distributes

- ✅ Original code (MIT).
- ✅ Our trained **LoRA adapter** (our own copyrightable work; a derivative of
  Llama, so labeled "Built with Llama").
- ❌ Never: Meta's base weights, raw copies of the datasets.

If you publish the adapter, include: this file, the Llama license, and a
"Built with Llama" notice on the model card.
