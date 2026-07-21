"""HumanEval pass@1 — the before/after showcase metric.

Run the SAME eval on each checkpoint (base, reasoning_sft, general_sft, grpo)
to produce the RESULTS.md table, mirroring how GPTlight benchmarked itself.

    python src/eval_code.py --adapter checkpoints/grpo --n 60
    python src/eval_code.py --adapter checkpoints/grpo --n 60 --agentic

`--agentic` runs the ReAct execute/retry loop instead of single-shot, to show
the additional lift from acting vs. just generating.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset                       # noqa: E402
from config.config import CONFIG                         # noqa: E402
from agent.executor import extract_code, run_code        # noqa: E402
from agent.react_agent import load_model, generate, solve  # noqa: E402
from data.prepare_data import CODE_SYSTEM_PROMPT          # noqa: E402


def humaneval_pass1(model, tok, n, agentic=False):
    ds = load_dataset(CONFIG.data.eval_hf, split="test")
    ds = ds.select(range(min(n, len(ds))))
    passed = 0
    for ex in ds:
        prompt, test, entry = ex["prompt"], ex["test"], ex["entry_point"]
        if agentic:
            # Let the agent iterate; give it the visible signature as problem.
            code, _, _ = solve(model, tok,
                               "Complete this function:\n" + prompt,
                               tests=[], setup="", max_iters=3, verbose=False)
        else:
            reply = generate(model, tok, [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": "Complete this function:\n" + prompt},
            ], temperature=0.0)
            code = extract_code(reply)
        script = code + "\n" + test + f"\ncheck({entry})\n"
        ok, _, _ = run_code(script, timeout=15)
        passed += int(ok)
    return passed / len(ds), len(ds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--agentic", action="store_true")
    args = ap.parse_args()

    model, tok = load_model(args.adapter)
    acc, total = humaneval_pass1(model, tok, args.n, args.agentic)
    mode = "agentic" if args.agentic else "single-shot"
    print(f"HumanEval pass@1 ({mode}, n={total}): {acc:.1%}")
