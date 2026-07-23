"""AgentLight chat UI — a streaming terminal chat with a Claude-style
"thinking" display.

While the model generates, a spinner shows it is working; the reasoning it
emits inside <think> </think> is streamed live in a dimmed style (like Claude
showing what it is thinking about), and the final answer is streamed after in
normal style. Multi-turn: the conversation is kept across messages.

    python chat/chat_ui.py --adapter checkpoints/grpo
    python chat/chat_ui.py --adapter unsloth/Llama-3.2-3B-Instruct   # base model

Commands inside the chat: /reset (new conversation), /exit (quit).

Runs wherever the model can be loaded. On a GPU (Kaggle/Colab notebook, or a
local CUDA box) it streams in real time; on a CPU-only box a 3B model is slow
(tens of seconds per reply) but still works.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react_agent import load_model  # noqa: E402

# Keep the <think> habit (so the reasoning panel has something to show) but,
# unlike the training-time code prompt, don't force a Python code block — this
# is a general chat that can also answer in prose.
CHAT_SYSTEM_PROMPT = (
    "You are AgentLight, a careful assistant. Think step by step inside "
    "<think> </think> tags, then give your final answer. When the task is to "
    "write code, put the final solution in a Python code block."
)


# ---------------------------------------------------------------------------
# ANSI styling (enable VT mode on Windows so the escapes render)
# ---------------------------------------------------------------------------
def _enable_ansi():
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


RESET = "\033[0m"
DIM = "\033[2m"
GREY = "\033[90m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"


# ---------------------------------------------------------------------------
# Spinner shown while waiting for the first token
# ---------------------------------------------------------------------------
class Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label="AgentLight denkt nach"):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r{GREY}{frame} {self.label}…{RESET}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # Clear the spinner line.
        sys.stdout.write("\r" + " " * (len(self.label) + 6) + "\r")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Splits a streamed reply into a "think" channel and an "answer" channel,
# tolerant of the <think>/</think> markers straddling stream chunk boundaries.
# ---------------------------------------------------------------------------
class ThinkRenderer:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.buf = ""
        self.pos = 0
        self.in_think = False
        self.think = ""     # full accumulated reasoning
        self.answer = ""    # full accumulated answer

    def feed(self, delta, final=False):
        """Return a list of (channel, text) segments ready to print."""
        self.buf += delta
        hold = 0 if final else max(len(self.OPEN), len(self.CLOSE)) - 1
        limit = len(self.buf) - hold
        out = []
        while self.pos < limit:
            if not self.in_think:
                idx = self.buf.find(self.OPEN, self.pos)
                if idx != -1 and idx < limit:
                    pre = self.buf[self.pos:idx]
                    if pre:
                        self.answer += pre
                        out.append(("answer", pre))
                    self.pos = idx + len(self.OPEN)
                    self.in_think = True
                else:
                    text = self.buf[self.pos:limit]
                    if text:
                        self.answer += text
                        out.append(("answer", text))
                    self.pos = limit
            else:
                idx = self.buf.find(self.CLOSE, self.pos)
                if idx != -1 and idx < limit:
                    seg = self.buf[self.pos:idx]
                    if seg:
                        self.think += seg
                        out.append(("think", seg))
                    self.pos = idx + len(self.CLOSE)
                    self.in_think = False
                else:
                    seg = self.buf[self.pos:limit]
                    if seg:
                        self.think += seg
                        out.append(("think", seg))
                    self.pos = limit
        return out


class Printer:
    """Prints channel segments with the right style and one-time headers."""

    def __init__(self):
        self.think_header = False
        self.answer_header = False
        self.think_active = False

    def emit(self, channel, text):
        if channel == "think":
            if not self.think_header:
                sys.stdout.write(f"{GREY}🤔 Nachdenken{RESET}\n{GREY}")
                self.think_header = True
                self.think_active = True
            sys.stdout.write(text)
        else:  # answer
            if self.think_active:
                sys.stdout.write(RESET)          # close the dim reasoning block
                self.think_active = False
            if not self.answer_header:
                lead = "\n\n" if self.think_header else ""
                sys.stdout.write(f"{lead}{BOLD}{CYAN}💬 AgentLight:{RESET} ")
                self.answer_header = True
            sys.stdout.write(text)
        sys.stdout.flush()

    def finish(self):
        if self.think_active:
            sys.stdout.write(RESET)
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------
def stream_reply(model, tok, messages, max_new_tokens=1536, temperature=0.4):
    import torch
    from transformers import TextIteratorStreamer

    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tok.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    def _run():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    spinner = Spinner()
    spinner.start()
    renderer = ThinkRenderer()
    printer = Printer()
    first = True
    try:
        for chunk in streamer:
            if first:
                spinner.stop()
                first = False
            for channel, text in renderer.feed(chunk):
                printer.emit(channel, text)
    finally:
        if first:  # never produced a token
            spinner.stop()
    for channel, text in renderer.feed("", final=True):
        printer.emit(channel, text)
    printer.finish()
    worker.join()
    return renderer.answer.strip()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/grpo",
                    help="Trained adapter dir, or a base model name to test.")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-turns", type=int, default=8,
                    help="How many past turns to keep in context.")
    args = ap.parse_args()

    _enable_ansi()
    print(f"{DIM}Lade Modell aus {args.adapter} …{RESET}")
    try:
        model, tok = load_model(args.adapter)
    except Exception as e:
        print(f"\nModell konnte nicht geladen werden: {e}\n"
              "Braucht torch+transformers+peft (und die Adapter-Dateien). "
              "Auf einer GPU (Kaggle/Colab) läuft es flüssig; lokal auf CPU "
              "langsam.")
        return

    print(f"\n{BOLD}{GREEN}=== AgentLight Chat ==={RESET}")
    print(f"{DIM}Streaming-Antworten mit sichtbarem Reasoning. "
          f"Befehle: /reset (neues Gespräch), /exit (beenden).{RESET}\n")

    system = {"role": "system", "content": CHAT_SYSTEM_PROMPT}
    history = []

    while True:
        try:
            msg = input(f"{BOLD}Du:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/exit":
            break
        if msg == "/reset":
            history = []
            print(f"{DIM}— neues Gespräch —{RESET}\n")
            continue

        history.append({"role": "user", "content": msg})
        messages = [system] + history[-2 * args.max_turns:]
        print()
        answer = stream_reply(model, tok, messages,
                              max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature)
        print()
        # Store only the answer (not the <think> block) in history, so past
        # reasoning doesn't accumulate and drag the model off-topic.
        history.append({"role": "assistant", "content": answer})

    print(f"{DIM}Tschüss.{RESET}")


if __name__ == "__main__":
    main()
