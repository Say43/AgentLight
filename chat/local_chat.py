"""Minimal local CLI to chat with / test a trained AgentLight adapter.

Each message is a fresh single-turn request (like GPTlight's local_chat: at this
model size, dragging prior turns into context hurt more than it helped). Use
`agent/react_agent.py` for the execute/retry coding loop.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react_agent import load_model, generate   # noqa: E402
from data.prepare_data import CODE_SYSTEM_PROMPT       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/grpo")
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    model, tok = load_model(args.adapter)
    print(f"AgentLight [{args.adapter}] — Ctrl-C to quit.\n")
    try:
        while True:
            user = input("you> ").strip()
            if not user:
                continue
            reply = generate(model, tok, [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ], temperature=args.temperature)
            print("\nagentlight>", reply, "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")


if __name__ == "__main__":
    main()
