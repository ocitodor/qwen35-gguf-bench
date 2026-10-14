#!/usr/bin/env python3
"""Collect the per-job result JSONs into one markdown table (+ one combined JSON).

Usage: summarize.py --results DIR [--matrix-json JSON] --out summary.md --json combined.json

Rows follow the matrix order; a matrix entry without a result file becomes a
"no result" row (the job died before run_bench.sh could write anything).

Rows are grouped by the host CPU family (one sub-table per family): runner
hosts differ between jobs, so pp/tg speeds are comparable only within a group.
PPL, HellaSwag and peak RAM do not depend on the host and compare across groups.

The family is a normalized key (noise words stripped, whitespace collapsed,
lower-cased: "INTEL(R) XEON(R) PLATINUM 8573C" and "Intel(R) Xeon(R) Platinum
8573C" are one family, "intel xeon platinum 8573c"); the raw CPU string stays
in its own column (cpu_model, from /proc/cpuinfo; cpu_info is llama-bench's).
"""
import argparse
import glob
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the tables carry non-ASCII; never crash on a narrow console

UNKNOWN_CPU = "unknown cpu"

# Stripped from the CPU name to get a family: "AMD EPYC 7763 64-Core Processor" -> "amd epyc 7763",
# "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz" -> "intel xeon platinum 8370c".
_CPU_NOISE = [
    r"\(R\)", r"\(TM\)", r"\d+-Core Processor", r"\bProcessor\b", r"\bCPU\b", r"@\s*[\d.]+\s*GHz",
]


def cpu_family(row):
    """Normalized grouping key for a result row (strip noise, collapse whitespace, lower-case)."""
    name = row.get("cpu_model") or row.get("cpu_info") or ""
    for pat in _CPU_NOISE:
        name = re.sub(pat, " ", name, flags=re.IGNORECASE)
    name = " ".join(name.strip().split()).lower()
    return name or UNKNOWN_CPU


def cpu_raw(row):
    """The CPU string as the host reported it (own column, never used as a key)."""
    return " ".join((row.get("cpu_model") or row.get("cpu_info") or "").split())


def fmt(v, nd=1):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def key(entry):
    return (entry.get("model", ""), entry.get("quant", ""))


TABLE_HEAD = [
    "| Model | Quant | File GB | pp512 t/s | tg128 t/s | PPL | HellaSwag % | Peak RAM GB | Status | CPU (raw) |",
    "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
]


def hellaswag_cell(r):
    acc = r.get("hellaswag_acc")
    if acc is None:
        return "-"
    cell = fmt(acc, 1)
    if r.get("hellaswag_tasks"):
        cell += f" (n={int(r['hellaswag_tasks'])})"
    return cell


def table_row(r):
    status = r.get("status", "?")
    if r.get("error"):
        status += f" — {r['error']}"
    if r.get("note"):
        status += f" ({r['note']})"
    if r.get("hellaswag_note"):
        status += f" [{r['hellaswag_note']}]"
    ppl = fmt(r.get("ppl"), 3)
    if r.get("ppl_err") is not None and r.get("ppl") is not None:
        ppl += f" ± {fmt(r.get('ppl_err'), 3)}"
    return (
        f"| {r.get('model','')} | {r.get('quant','')} | {fmt(r.get('file_gb'), 2)} | "
        f"{fmt(r.get('pp512_tps'))} | {fmt(r.get('tg128_tps'))} | {ppl} | {hellaswag_cell(r)} | "
        f"{fmt(r.get('peak_rss_gb'), 2)} | {status} | {r.get('cpu_raw') or '-'} |"
    )


def group_rows(rows):
    """[(family, [rows in matrix order])], largest group first, ties alphabetical, unknown CPU last."""
    groups = {}
    for r in rows:
        groups.setdefault(r["cpu_family"], []).append(r)
    return sorted(groups.items(), key=lambda kv: (kv[0] == UNKNOWN_CPU, -len(kv[1]), kv[0]))


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
        r["cpu_family"] = cpu_family(r)
        r["cpu_raw"] = cpu_raw(r)
        rows.append(r)

    lines = ["## Qwen 3.5 GGUF bench", ""]
    tags = sorted({r.get("llama_cpp_tag") for r in rows if r.get("llama_cpp_tag")})
    threads = sorted({r.get("threads") for r in rows if r.get("threads")})
    ctx = sorted({(r.get("ppl_ctx"), r.get("ppl_chunks")) for r in rows if r.get("ppl_ctx")})
    hs = sorted({int(r["hellaswag_tasks"]) for r in rows if r.get("hellaswag_tasks")})
    meta = []
    if tags:
        meta.append("llama.cpp " + ", ".join(tags))
    if threads:
        meta.append("threads " + ", ".join(str(t) for t in threads))
    if ctx:
        meta.append("PPL on bench/ppl.txt, " + ", ".join(f"ctx {c} x {n} chunks" for c, n in ctx))
    if hs:
        meta.append("HellaSwag acc_norm over " + ", ".join(str(n) for n in hs) + " validation tasks")
    if meta:
        lines += ["_" + " · ".join(meta) + "_", ""]

    groups = group_rows(rows)
    lines += ["Rows are grouped by the runner's CPU family (normalized key; the raw string is in the last "
              "column): speed (pp/tg) is comparable only within a group; PPL, HellaSwag and RAM are "
              "comparable across groups.", ""]
    for family, members in groups:
        lines += [f"### {family} ({len(members)} row{'s' if len(members) != 1 else ''})", ""]
        lines += TABLE_HEAD
        lines += [table_row(r) for r in members]
        lines.append("")

    ok = sum(1 for r in rows if r.get("status") == "ok")
    lines += [f"{ok}/{len(rows)} models benchmarked on {len(groups)} CPU "
              f"{'family' if len(groups) == 1 else 'families'}. pp512 = prompt processing speed, "
              "tg128 = generation speed, both tokens/s on the runner's CPU. PPL is a relative "
              "measure on a fixed text (lower is better); compare rows against each other, "
              "not against published numbers. HellaSwag is acc_norm on a fixed random subset of the "
              "validation set (higher is better; a few hundred tasks carry a ±4-5 point interval).", ""]

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
