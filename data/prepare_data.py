"""Dataset loading & formatting for AgentLight (coding-agent showcase).

Every SFT phase gets data in a uniform conversational schema:

    {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}

GRPO needs a prompt + a *verifiable* target (the unit tests to run):

    {"prompt": [<messages>], "tests": [<assert strings>], "setup": str}

All sources and their licenses are documented in THIRD_PARTY_NOTICES.md.
No OpenAI/Anthropic/Google-model-distilled data is used for TRAINING.
(HumanEval is used for EVAL only and is a human-written MIT benchmark.)
"""

from __future__ import annotations

from datasets import load_dataset

# Shared format contract. Every phase reinforces the same behaviour: reason
# inside <think>, then emit exactly one ```python code block.
CODE_SYSTEM_PROMPT = (
    "You are AgentLight, a careful coding assistant. Think step by step inside "
    "<think> </think> tags, then output the final solution as a single Python "
    "code block:\n```python\n# your code here\n```\n"
    "Define the requested function(s) at module top level."
)


def _cap(ds, n):
    return ds.select(range(min(n, len(ds)))) if n and n < len(ds) else ds


def _sharegpt_to_messages(conv):
    """Normalize a sharegpt-style [{from,value}] list to {role,content}."""
    role_map = {"human": "user", "user": "user", "gpt": "assistant",
                "assistant": "assistant", "system": "system"}
    out = []
    for turn in conv:
        role = role_map.get(turn.get("from") or turn.get("role"), "user")
        out.append({"role": role, "content": turn.get("value") or turn.get("content", "")})
    return out


# ---------------------------------------------------------------------------
# Reasoning SFT — R1-distilled long-CoT (open-thoughts/OpenThoughts-114k)
# ---------------------------------------------------------------------------
def load_reasoning_sft(cfg, n=None):
    ds = load_dataset(cfg.reasoning_sft_hf, split=cfg.reasoning_sft_split)
    ds = _cap(ds, n)
    cols = set(ds.column_names)

    def to_messages(ex):
        # Handle the common OpenThoughts schemas robustly.
        if "conversations" in ex and ex["conversations"]:
            msgs = _sharegpt_to_messages(ex["conversations"])
        elif "messages" in ex and ex["messages"]:
            msgs = ex["messages"]
        else:
            problem = ex.get("problem") or ex.get("question") or ex.get("prompt", "")
            sol = ex.get("solution") or ex.get("response") or ex.get("answer", "")
            msgs = [{"role": "user", "content": problem},
                    {"role": "assistant", "content": sol}]
        # Ensure a system prompt is present and consistent.
        if not msgs or msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": CODE_SYSTEM_PROMPT}] + msgs
        return {"messages": msgs}

    return ds.map(to_messages, remove_columns=list(cols))


# ---------------------------------------------------------------------------
# General SFT — smoltalk (Apache-2.0); already in {messages} schema.
# ---------------------------------------------------------------------------
def load_general_sft(cfg, n=None):
    ds = load_dataset(cfg.general_sft_hf, cfg.general_sft_config, split="train")
    ds = _cap(ds, n)
    keep = [c for c in ds.column_names if c != "messages"]
    return ds.remove_columns(keep) if keep else ds


# ---------------------------------------------------------------------------
# GRPO — MBPP: Python tasks with executable test_list (verifiable reward).
# ---------------------------------------------------------------------------
def load_grpo(cfg, n=None):
    ds = load_dataset(cfg.grpo_hf, cfg.grpo_config, split="train")
    ds = _cap(ds, n)

    def to_prompt(ex):
        problem = ex.get("text") or ex.get("prompt", "")
        tests = ex.get("test_list", []) or []
        setup = ex.get("test_setup_code", "") or ""
        user = (problem.strip() +
                "\n\nYour solution must pass these tests:\n" +
                "\n".join(tests))
        return {
            "prompt": [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "tests": tests,
            "setup": setup,
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


LOADERS = {
    "reasoning_sft": load_reasoning_sft,
    "general_sft": load_general_sft,
    "grpo": load_grpo,
}
