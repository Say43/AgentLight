"""HumanEval pass@1 — the before/after showcase metric.

Run the SAME eval on each checkpoint (base, reasoning_sft, general_sft, grpo)
to produce the RESULTS.md table, mirroring how GPTlight benchmarked itself.

    python src/eval_code.py --adapter checkpoints/grpo --n 60
    python src/eval_code.py --adapter checkpoints/grpo --n 60 --agentic

`--agentic` runs the ReAct execute/retry loop instead of single-shot, to show
the additional lift from acting vs. just generating.

    python src/eval_code.py --adapter checkpoints/grpo --n 60 --tts --k 4

`--tts` is test-time scaling: sample --k candidates and select the one that
passes the most public doctest examples from the prompt. HumanEval's hidden
`check()` suite is used only once for final grading.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset                       # noqa: E402
from config.config import CONFIG                         # noqa: E402
from agent.executor import extract_code, run_code, score_solution  # noqa: E402
from agent.react_agent import load_model, generate, solve  # noqa: E402
from data.prepare_data import CODE_SYSTEM_PROMPT          # noqa: E402

def _public_tests_from_prompt(prompt: str):
    """Convert simple one-line doctest examples already visible in the
    HumanEval prompt into asserts. The hidden `check()` suite is deliberately
    never returned or shown to the model.
    """
    lines = prompt.splitlines()
    tests = []
    for index, line in enumerate(lines[:-1]):
        stripped = line.strip()
        if not stripped.startswith(">>> "):
            continue
        expression = stripped[4:].strip()
        expected = lines[index + 1].strip()
        if (not expected or expected.startswith(">>> ") or
                expression.startswith("print(")):
            continue
        candidate = f"assert ({expression}) == ({expected})"
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        tests.append(candidate)
    return tests


def humaneval_pass1(model, tok, n, agentic=False):
    ds = load_dataset(CONFIG.data.eval_hf, split="test")
    ds = ds.select(range(min(n, len(ds))))
    passed = 0
    for ex in ds:
        prompt, test, entry = ex["prompt"], ex["test"], ex["entry_point"]
        if agentic:
            # Execute only examples already present in the public prompt.
            # The hidden HumanEval check remains exclusively for final grading.
            tests = _public_tests_from_prompt(prompt)
            code, _, _ = solve(model, tok,
                               "Complete this function:\n" + prompt,
                               tests=tests, setup="", max_iters=3, verbose=False)
        else:
            reply = generate(model, tok, [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": "Complete this function:\n" + prompt},
            ], temperature=0.0)
            code = extract_code(reply)
        # Same final grading check regardless of path -- single-shot and
        # agentic are judged on identical hidden tests, a fair comparison.
        script = code + "\n" + test + f"\ncheck({entry})\n"
        ok, _, _ = run_code(script, timeout=15)
        passed += int(ok)
    return passed / len(ds), len(ds)


def humaneval_tts(model, tok, n, k=4, temperature=0.8):
    """Test-time-scaling pass@1: sample k candidates, keep the one that
    passes the most public doctest examples, then grade that pick once against
    the hidden check suite. Prompts without parseable examples fall back to
    selecting the first candidate that executes without an exception.
    """
    ds = load_dataset(CONFIG.data.eval_hf, split="test")
    ds = ds.select(range(min(n, len(ds))))
    passed = 0
    for ex in ds:
        prompt, test, entry = ex["prompt"], ex["test"], ex["entry_point"]
        tests = _public_tests_from_prompt(prompt)
        best_code, best_score = "", -1.0
        for _ in range(k):
            reply = generate(model, tok, [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": "Complete this function:\n" + prompt},
            ], temperature=temperature)
            code = extract_code(reply)
            if tests:
                score = score_solution(code, tests, timeout=3.0)
            else:
                ok, _, _ = run_code(code, timeout=3.0)
                score = float(ok)
            if score > best_score:
                best_code, best_score = code, score
            if score >= 1.0:
                break  # already found a fully-passing candidate
        # Same final grading check as the other modes.
        script = best_code + "\n" + test + f"\ncheck({entry})\n"
        ok, _, _ = run_code(script, timeout=15)
        passed += int(ok)
    return passed / len(ds), len(ds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--agentic", action="store_true")
    ap.add_argument("--tts", action="store_true",
                    help="Test-time scaling: sample --k candidates and select "
                         "the one passing the most example asserts.")
    ap.add_argument("--k", type=int, default=4,
                    help="Number of candidates to sample under --tts.")
    args = ap.parse_args()

    model, tok = load_model(args.adapter)
    if args.tts:
        acc, total = humaneval_tts(model, tok, args.n, k=args.k)
        print(f"HumanEval pass@1 (test-time scaling, k={args.k}, n={total}): {acc:.1%}")
    else:
        acc, total = humaneval_pass1(model, tok, args.n, args.agentic)
        mode = "agentic" if args.agentic else "single-shot"
        print(f"HumanEval pass@1 ({mode}, n={total}): {acc:.1%}")
