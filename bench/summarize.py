#!/usr/bin/env python3
"""Collect the per-job result JSONs into one markdown table (+ one combined JSON).

Usage: summarize.py --results DIR [--matrix-json JSON] --out summary.md --json combined.json

Rows follow the matrix order; a matrix entry without a result file becomes a
"no result" row (the job died before run_bench.sh could write anything).
"""
import argparse
import glob
import json
import os


def fmt(v, nd=1):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def key(entry):
    return (entry.get("model", ""), entry.get("quant", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--matrix-json", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    results = {}
    for path in sorted(glob.glob(os.path.join(a.results, "**", "*.json"), recursive=True)):
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and "model" in r and "quant" in r:
            results[key(r)] = r

    matrix = []
    if a.matrix_json.strip():
        try:
            matrix = json.loads(a.matrix_json)
        except ValueError:
            matrix = []
    order = [key(m) for m in matrix] + [k for k in results if k not in {key(m) for m in matrix}]

    rows = []
    for k in order:
        r = results.get(k)
        m = next((m for m in matrix if key(m) == k), {})
        if r is None:
            r = {"model": k[0], "quant": k[1], "note": m.get("note", ""),
                 "status": "no result", "error": "job ended before writing a result (see its log)"}
        rows.append(r)

    lines = ["## Qwen 3.5 GGUF bench", ""]
    tags = sorted({r.get("llama_cpp_tag") for r in rows if r.get("llama_cpp_tag")})
    threads = sorted({r.get("threads") for r in rows if r.get("threads")})
    cpus = sorted({r.get("cpu_info") for r in rows if r.get("cpu_info")})
    ctx = sorted({(r.get("ppl_ctx"), r.get("ppl_chunks")) for r in rows if r.get("ppl_ctx")})
    meta = []
    if tags:
        meta.append("llama.cpp " + ", ".join(tags))
    if threads:
        meta.append("threads " + ", ".join(str(t) for t in threads))
    if ctx:
        meta.append("PPL on bench/ppl.txt, " + ", ".join(f"ctx {c} x {n} chunks" for c, n in ctx))
    if cpus:
        meta.append("CPU: " + "; ".join(cpus))
    if meta:
        lines += ["_" + " · ".join(meta) + "_", ""]

    lines += [
        "| Model | Quant | File GB | pp512 t/s | tg128 t/s | PPL | Peak RAM GB | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        status = r.get("status", "?")
        if r.get("error"):
            status += f" — {r['error']}"
        if r.get("note"):
            status += f" ({r['note']})"
        ppl = fmt(r.get("ppl"), 3)
        if r.get("ppl_err") is not None and r.get("ppl") is not None:
            ppl += f" ± {fmt(r.get('ppl_err'), 3)}"
        lines.append(
            f"| {r.get('model','')} | {r.get('quant','')} | {fmt(r.get('file_gb'), 2)} | "
            f"{fmt(r.get('pp512_tps'))} | {fmt(r.get('tg128_tps'))} | {ppl} | "
            f"{fmt(r.get('peak_rss_gb'), 2)} | {status} |"
        )

    ok = sum(1 for r in rows if r.get("status") == "ok")
    lines += ["", f"{ok}/{len(rows)} models benchmarked. pp512 = prompt processing speed, "
              "tg128 = generation speed, both tokens/s on the runner's CPU. PPL is a relative "
              "measure on a fixed text (lower is better); compare rows against each other, "
              "not against published numbers.", ""]

    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    with open(a.json, "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
