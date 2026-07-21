"""Sandboxed Python execution — shared by the GRPO reward and the ReAct agent.

SECURITY: this runs model-generated code. It is safe to run inside an isolated
container (Kaggle kernels, Docker, a throwaway VM) — which is exactly where
training happens. Do NOT point this at a machine with data you care about.
Mitigations here: separate subprocess, hard wall-clock timeout, a fresh temp
dir as CWD, and (on POSIX) CPU/address-space rlimits. It is a sandbox by
convention, not a hardened jail.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the last ```python block; fall back to text after </think>."""
    blocks = _CODE_BLOCK.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return (text or "").strip()


# POSIX-only resource limiting; a no-op on Windows.
_PREAMBLE = textwrap.dedent("""
    import resource, sys
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    except Exception:
        pass
""") if os.name == "posix" else ""


def _run(script: str, timeout: float):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            p = subprocess.run(
                [sys.executable, "-I", path],
                cwd=tmp, capture_output=True, text=True, timeout=timeout,
            )
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "TIMEOUT"
        except Exception as e:  # pragma: no cover
            return -1, "", f"EXEC_ERROR: {e}"


def run_code(code: str, timeout: float = 10.0):
    """Execute code as-is. Returns (ok, stdout, stderr)."""
    rc, out, err = _run(_PREAMBLE + code, timeout)
    return rc == 0, out, err


def score_solution(code: str, tests: list[str], setup: str = "",
                   timeout: float = 12.0) -> float:
    """Fraction of unit tests the candidate code passes (0.0-1.0).

    Each assert is run in isolation so one failure doesn't hide the rest.
    A candidate that doesn't even import/define cleanly scores 0.
    """
    if not tests:
        return 0.0
    harness = _PREAMBLE + code + "\n" + (setup or "") + "\n"
    harness += "import json as _json\n_passed = 0\n_tests = [\n"
    for t in tests:
        harness += "    " + repr(t) + ",\n"
    harness += "]\n"
    harness += textwrap.dedent("""
        for _t in _tests:
            try:
                exec(_t)
                _passed += 1
            except Exception:
                pass
        print("SCORE=%d/%d" % (_passed, len(_tests)))
    """)
    rc, out, err = _run(harness, timeout)
    m = re.search(r"SCORE=(\d+)/(\d+)", out)
    if not m:
        return 0.0
    passed, total = int(m.group(1)), int(m.group(2))
    return passed / total if total else 0.0
