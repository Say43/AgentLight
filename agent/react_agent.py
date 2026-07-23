"""AgentLight ReAct coding agent.

This is what makes the model an *agent* rather than a chatbot: given a problem
(and optionally tests), it writes a solution, RUNS it via the code executor,
reads the failure, and revises — looping until tests pass or a budget is hit.

The model only has to emit reasoning + a code block; the agentic loop
(execute -> observe -> retry) lives here, in the harness, exactly as planned.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.executor import extract_code, run_code, score_solution  # noqa: E402
from data.prepare_data import CODE_SYSTEM_PROMPT                     # noqa: E402


def load_model(adapter_dir: str):
    """Load base + LoRA adapter. Prefers Unsloth; falls back to transformers."""
    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=adapter_dir, max_seq_length=2048, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
        return model, tok
    except Exception:
        import json
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg_path = os.path.join(adapter_dir, "adapter_config.json")
        if os.path.exists(cfg_path):
            # adapter_dir is a LoRA adapter (only adapter_config.json +
            # adapter_model.safetensors -- no tokenizer/base weights). The
            # tokenizer AND base weights come from the base model it references;
            # loading the tokenizer from adapter_dir fails (no tokenizer files).
            from peft import PeftModel
            with open(cfg_path, encoding="utf-8") as f:
                base_name = json.load(f)["base_model_name_or_path"]
            tok = AutoTokenizer.from_pretrained(base_name)
            base = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype=torch.float16, device_map="auto")
            model = PeftModel.from_pretrained(base, adapter_dir)
        else:
            # adapter_dir is actually a full model / HF repo name (e.g. the
            # base model itself, for a baseline chat).
            tok = AutoTokenizer.from_pretrained(adapter_dir)
            model = AutoModelForCausalLM.from_pretrained(
                adapter_dir, torch_dtype=torch.float16, device_map="auto")
        return model, tok


def generate(model, tok, messages, max_new_tokens=1024, temperature=0.4):
    import torch
    # return_dict=True: newer transformers returns a BatchEncoding (input_ids +
    # attention_mask), and generate() must get those as kwargs, not a bare
    # tensor as the first positional arg (that path calls .shape on the dict).
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                      pad_token_id=tok.eos_token_id)
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def solve(model, tok, problem: str, tests=None, setup: str = "",
          max_iters: int = 3, verbose: bool = True):
    """ReAct loop. Returns (code, passed_fraction, transcript)."""
    tests = tests or []
    messages = [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": problem +
         ("\n\nTests:\n" + "\n".join(tests) if tests else "")},
    ]
    transcript = []
    best_code, best_score = "", -1.0

    for it in range(max_iters):
        reply = generate(model, tok, messages)
        code = extract_code(reply)
        transcript.append(("assistant", reply))

        if tests:
            score = score_solution(code, tests, setup)
            if score > best_score:
                best_code, best_score = code, score
            if verbose:
                print(f"[iter {it}] tests passed: {score:.0%}", flush=True)
            if score >= 1.0:
                return code, 1.0, transcript
            # Observation: report failures back to the model (the ReAct step).
            ok, out, err = run_code(code + "\n" + setup + "\n" +
                                    "\n".join(tests))
            obs = f"Tests failed. Runtime output:\n{(err or out)[:800]}"
        else:
            ok, out, err = run_code(code)
            if ok:
                return code, 1.0, transcript
            obs = f"Execution error:\n{(err or out)[:800]}"

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": obs +
                         "\nFix the code and return the full corrected solution."})

    return best_code or code, max(best_score, 0.0), transcript


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/grpo")
    ap.add_argument("--problem", required=True)
    args = ap.parse_args()
    m, t = load_model(args.adapter)
    code, score, _ = solve(m, t, args.problem)
    print("\n=== SOLUTION ===\n" + code)
