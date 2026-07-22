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

import random
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
    # No _split_for_budget() pre-slice here, unlike the other loaders: this
    # dataset is grouped by domain in its stored row order (verified via HF's
    # datasets-server statistics endpoint: 89,120 "math" rows before the
    # 19,904 "code" rows out of 113,957 total), so a smoke-mode index slice
    # like "train[:100]" lands entirely inside the math block and the domain
    # filter below zeroes out every row (confirmed: smoke run v5 crashed with
    # exactly this). The full "metadata" config is small regardless (~16s to
    # generate on Kaggle, confirmed in the same run's log), so loading it
    # unsliced and filtering+shuffling+capping afterward is cheap and correct
    # in both smoke and real mode.
    ds = load_dataset(
        cfg.reasoning_sft_hf,
        cfg.reasoning_sft_config,
        split=cfg.reasoning_sft_split,
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
    if len(ds) == 0:
        raise RuntimeError(
            f"reasoning_sft: domain filter '{cfg.reasoning_sft_domain}' matched "
            "0 rows. Check that the domain value is still valid for "
            f"{cfg.reasoning_sft_hf}/{cfg.reasoning_sft_config}.")
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



# ---------------------------------------------------------------------------
# Repair SFT (Kimi-Dev-style skill-then-repair) — same license-clean MBPP
# source as GRPO, turned into buggy-attempt -> test-failure -> corrected-code
# trajectories instead of RL rollouts.
# ---------------------------------------------------------------------------
# Each entry mutates exactly one common bug pattern into otherwise-correct
# MBPP gold code, e.g. a wrong comparison or an off-by-one. Fixed-string
# replacements only (no backreferences needed).
_BUG_PATTERNS = [
    (re.compile(r"<="), "<"),
    (re.compile(r">="), ">"),
    (re.compile(r"(?<![=!<>])==(?!=)"), "!="),
    (re.compile(r"\breturn True\b"), "return False"),
    (re.compile(r"\breturn False\b"), "return True"),
    (re.compile(r"\bmin\("), "max("),
    (re.compile(r"\bmax\("), "min("),
    (re.compile(r"\bsorted\("), "list("),
    (re.compile(r"\+ 1\b"), "- 1"),
    (re.compile(r"- 1\b"), "+ 1"),
]


def _mutate_code(code: str, rng: random.Random) -> str | None:
    """Turn correct MBPP code into a plausible one-bug WRONG attempt.

    Deterministic given `rng`: shuffles the candidate-pattern order so the
    corpus doesn't all fail the same way, then mutates exactly one randomly
    chosen occurrence of the first pattern that actually matches. Returns
    None if no candidate pattern matches (short/simple solutions) — the
    caller skips that MBPP row rather than emitting a "buggy" attempt that
    isn't actually buggy.
    """
    order = list(range(len(_BUG_PATTERNS)))
    rng.shuffle(order)
    for idx in order:
        pattern, repl = _BUG_PATTERNS[idx]
        matches = list(pattern.finditer(code))
        if matches:
            m = rng.choice(matches)
            return code[:m.start()] + repl + code[m.end():]
    return None


def _synthesize_failure(public_test: str) -> str:
    """Build a realistic-looking AssertionError trace for a failed test.

    The mutated code is never actually executed here — data prep stays a
    pure, fast, non-sandboxed step (only src/train.py's GRPO reward and
    agent/executor.py touch the sandboxed runner) — so this mirrors the
    shape of a genuine pytest/assert failure without inventing a specific
    wrong right-hand-side value the mutation didn't really produce.
    """
    return (
        "Ran the public test:\n"
        f"    {public_test}\n\n"
        "Traceback (most recent call last):\n"
        "  File \"solution.py\", line 1, in <module>\n"
        f"    {public_test}\n"
        "AssertionError"
    )


def load_repair_sft(cfg, n=None):
    ds = load_dataset(
        cfg.grpo_hf,
        cfg.grpo_config,
        split=_split_for_budget("train", n),
    )
    # Shuffle with the config seed BEFORE capping, same discipline as the
    # other loaders, so the budget is not an accidental slice of the corpus.
    ds = ds.shuffle(seed=cfg.shuffle_seed)
    cols = set(ds.column_names)

    def to_repair(ex, idx):
        gold = (ex.get("code") or "").strip()
        all_tests = ex.get("test_list", []) or []
        if not gold or not all_tests:
            return {"messages": []}
        rng = random.Random(cfg.shuffle_seed + idx)
        buggy = _mutate_code(gold, rng)
        if buggy is None:
            return {"messages": []}
        public_test = all_tests[0]
        problem = (ex.get("text") or ex.get("prompt") or "").strip()
        user_problem = (problem +
                        "\n\nYour solution must satisfy this public example:\n" +
                        public_test)
        failure = _synthesize_failure(public_test)
        think = (
            "The previous attempt fails the public test above. Re-reading "
            "the problem and comparing against the example, the bug is a "
            "small logic error (a comparison/boundary/helper-function "
            "mistake). Fixing that and re-checking against the example "
            "gives the corrected solution below."
        )
        msgs = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_problem},
            {"role": "assistant", "content": "```python\n" + buggy + "\n```"},
            {"role": "user", "content": failure},
            {"role": "assistant", "content": (
                "<think>\n" + think + "\n</think>\n\n"
                "```python\n" + gold + "\n```"
            )},
        ]
        return {"messages": msgs}

    ds = ds.map(to_repair, with_indices=True, remove_columns=list(cols))
    ds = ds.filter(lambda ex: bool(ex["messages"]))
    ds = _cap(ds, n)
    if len(ds) == 0:
        raise RuntimeError(
            "repair_sft: 0 rows survived bug-mutation + filtering. Check the "
            f"smoke-mode split slice against {cfg.grpo_hf}/{cfg.grpo_config}.")
    return ds


LOADERS = {
    "reasoning_sft": load_reasoning_sft,
    "repair_sft": load_repair_sft,
    "general_sft": load_general_sft,
    "grpo": load_grpo,
}
