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

import re
import os

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


def _split_for_budget(split: str, n) -> str:
    """Avoid downloading full multi-GB corpora for a smoke run."""
    smoke = os.environ.get("AGENTLIGHT_SMOKE", "").lower() in {"1", "true", "yes"}
    if smoke and n:
        return f"{split}[:{max(100, int(n) * 4)}]"
    return split


def _code_only(text: str) -> str:
    """Normalize a teacher answer to the contents of one Python code fence."""
    blocks = re.findall(
        r"```(?:python)?\s*\n(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    return (blocks[-1] if blocks else (text or "")).strip()


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
    ds = load_dataset(
        cfg.reasoning_sft_hf,
        cfg.reasoning_sft_config,
        split=_split_for_budget(cfg.reasoning_sft_split, n),
    )
    # The source mixes code, math, science and puzzles. AgentLight's system
    # contract always asks for Python, so training on the non-code rows creates
    # contradictory labels. Filter explicitly, then shuffle before capping so
    # the budget is not an accidental slice of one upstream source.
    if "domain" in ds.column_names:
        wanted = cfg.reasoning_sft_domain.lower()
        ds = ds.filter(lambda ex: str(ex.get("domain", "")).lower() == wanted)
    ds = ds.shuffle(seed=cfg.shuffle_seed)
    ds = _cap(ds, n)
    cols = set(ds.column_names)

    def to_messages(ex):
        # Prefer the metadata schema: it keeps reasoning and final solution
        # separate, allowing us to enforce exactly the format used at inference.
        if ex.get("problem") and ex.get("deepseek_solution"):
            problem = ex["problem"].strip()
            starter = (ex.get("starter_code") or "").strip()
            if starter:
                problem += "\n\nStarter code:\n```python\n" + starter + "\n```"
            reasoning = (ex.get("deepseek_reasoning") or "").strip()
            solution = _code_only(ex["deepseek_solution"])
            answer = (
                "<think>\n" + reasoning + "\n</think>\n\n"
                "```python\n" + solution + "\n```"
            )
            msgs = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer},
            ]
        elif "conversations" in ex and ex["conversations"]:
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
    ds = load_dataset(
        cfg.general_sft_hf,
        cfg.general_sft_config,
        split=_split_for_budget("train", n),
    )
    ds = ds.shuffle(seed=cfg.shuffle_seed)
    ds = _cap(ds, n)
    keep = [c for c in ds.column_names if c != "messages"]
    return ds.remove_columns(keep) if keep else ds


# ---------------------------------------------------------------------------
# GRPO — MBPP: Python tasks with executable test_list (verifiable reward).
# ---------------------------------------------------------------------------
def load_grpo(cfg, n=None):
    ds = load_dataset(
        cfg.grpo_hf,
        cfg.grpo_config,
        split=_split_for_budget("train", n),
    )
    ds = ds.shuffle(seed=cfg.shuffle_seed)
    ds = _cap(ds, n)

    def to_prompt(ex):
        problem = ex.get("text") or ex.get("prompt", "")
        all_tests = ex.get("test_list", []) or []
        # MBPP often omits the requested function name from the prose. Show one
        # public example so the interface is knowable, but keep the remaining
        # tests out of the prompt and use only those for the RL reward.
        public_tests = all_tests[:1]
        hidden_tests = all_tests[1:] or all_tests
        setup = ex.get("test_setup_code", "") or ""
        user = (problem.strip() +
                "\n\nYour solution must satisfy this public example:\n" +
                "\n".join(public_tests))
        return {
            "prompt": [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "tests": hidden_tests,
            "public_tests": public_tests,
            "setup": setup,
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


LOADERS = {
    "reasoning_sft": load_reasoning_sft,
    "general_sft": load_general_sft,
    "grpo": load_grpo,
}
