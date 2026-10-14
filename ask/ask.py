#!/usr/bin/env python3
"""Send the prompt (or every prompt of PROMPTS_FILE) to a running llama-server
and write out/answers.jsonl, out/answer.md and the run summary.

Environment: MODEL_NAME QUANT PROMPT PROMPTS_FILE SYSTEM MAX_TOKENS TEMPERATURE THINKING PORT
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def load_prompts():
    env = os.environ
    path = env.get("PROMPTS_FILE", "").strip()
    default_system = env.get("SYSTEM", "")
    if not path:
        return [{"prompt": env.get("PROMPT", ""), "system": default_system}]
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if path.endswith(".jsonl"):
                obj = json.loads(line)
                items.append({"prompt": obj.get("prompt", ""), "system": obj.get("system", default_system)})
            else:
                items.append({"prompt": line, "system": default_system})
    return items


def ask(port, item, max_tokens, temperature, thinking):
    messages = []
    if item["system"]:
        messages.append({"role": "system", "content": item["system"]})
    messages.append({"role": "user", "content": item["prompt"]})
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.load(resp)
    elapsed = time.monotonic() - t0
    choice = data["choices"][0]
    msg = choice.get("message", {})
    timings = data.get("timings", {})
    return {
        "prompt": item["prompt"],
        "system": item["system"],
        "answer": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
        "pp_tps": timings.get("prompt_per_second"),
        "tg_tps": timings.get("predicted_per_second"),
        "elapsed_s": round(elapsed, 1),
    }


def render(label, results, max_tokens, temperature, thinking):
    lines = [f"## {label}", ""]
    lines.append(f"_max_tokens {max_tokens} · temperature {temperature} · thinking {'on' if thinking else 'off'}_")
    lines.append("")
    for r in results:
        if len(results) > 1:
            lines.append(f"### {r['n']}. {r['prompt'][:100]}")
        else:
            lines.append("### Prompt")
        lines.append("")
        if r["system"]:
            lines.append(f"**System:** {r['system']}")
            lines.append("")
        lines.append(f"**User:** {r['prompt']}")
        lines.append("")
        if r["reasoning"]:
            lines += ["<details><summary>Reasoning</summary>", "", r["reasoning"], "", "</details>", ""]
        lines.append("**Answer:**")
        lines.append("")
        lines.append(r["answer"] if r["answer"].strip() else "_(empty)_")
        lines.append("")
        cut = " (cut at max_tokens)" if r["finish_reason"] == "length" else ""
        lines.append(f"_{r['completion_tokens']} tokens in {r['elapsed_s']} s, "
                     f"{(r['tg_tps'] or 0):.1f} t/s generation, {(r['pp_tps'] or 0):.1f} t/s prompt{cut}_")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = os.environ
    port = int(env.get("PORT") or 8080)
    max_tokens = int(env.get("MAX_TOKENS") or 512)
    temperature = float(env.get("TEMPERATURE") or 0.7)
    thinking = env.get("THINKING", "false").lower() == "true"
    label = f'{env.get("MODEL_NAME", "")} {env.get("QUANT", "")}'.strip()

    items = [i for i in load_prompts() if i["prompt"].strip()]
    if not items:
        sys.exit("ask.py: no prompt to send")

    os.makedirs(a.out, exist_ok=True)
    results = []
    with open(os.path.join(a.out, "answers.jsonl"), "w", encoding="utf-8") as jf:
        for n, item in enumerate(items, 1):
            print(f"[{n}/{len(items)}] {item['prompt'][:80]!r}")
            r = ask(port, item, max_tokens, temperature, thinking)
            r["n"] = n
            results.append(r)
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"    {r['completion_tokens']} tokens in {r['elapsed_s']}s, {(r['tg_tps'] or 0):.1f} t/s, {r['finish_reason']}")

    md = render(label, results, max_tokens, temperature, thinking)
    with open(os.path.join(a.out, "answer.md"), "w", encoding="utf-8") as f:
        f.write(md)
    summary = env.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
