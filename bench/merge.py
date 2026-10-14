#!/usr/bin/env python3
"""Merge the result rows of several bench runs into one table of medians.

Usage: merge.py [--out merged.md] [--json merged.json] PATH...

Each PATH is a combined.json (a list of rows, as summarize.py writes it), a
single result JSON (one row, as run_bench.sh writes it) or a directory that is
searched for *.json recursively. Rows are de-duplicated by (run_id, model, quant).

Groups: (model, quant, cpu_family, llama_cpp_tag). Within a group:
- pp512 / tg128 / PPL / HellaSwag: the median over the *successful* rows only
  (status ok), with n and the spread (IQR for n >= 4, else min-max);
- n < 2 is flagged: a single run is not a median;
- failed / skipped / OOM rows are counted separately per group and never
  enter the medians;
- peak RAM is the max over all rows that reported it (RAM is a ceiling, not a
  typical value).
The raw rows are kept in the JSON output ("rows") and in a collapsed section
of the markdown, so nothing is medianed away.
"""
import argparse
import glob
import json
import os
import statistics
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the tables carry non-ASCII; never crash on a narrow console

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import cpu_family, cpu_raw, fmt  # noqa: E402

OOM_WORDS = ("oom", "out of memory", "killed", "cannot allocate", "bad_alloc", "exit code 137")


def load_rows(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        else:
            files.append(p)
    rows, seen = [], set()
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        items = data if isinstance(data, list) else [data]
        for r in items:
            if not (isinstance(r, dict) and "model" in r and "quant" in r):
                continue
            k = (str(r.get("run_id", "")), r["model"], r["quant"], r.get("cpu_model") or r.get("cpu_info") or "")
            if k in seen:
                continue
            seen.add(k)
            r = dict(r)
            r["cpu_family"] = cpu_family(r)
            r["cpu_raw"] = cpu_raw(r)
            r["source_file"] = path
            rows.append(r)
    return rows


def is_oom(r):
    text = " ".join(str(r.get(k, "")) for k in ("error", "status", "note")).lower()
    return any(w in text for w in OOM_WORDS)


def nums(rows, field):
    out = []
    for r in rows:
        v = r.get(field)
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def stat(values, nd=1):
    """(median, spread label, spread text) over values; None if empty."""
    if not values:
        return None
    med = statistics.median(values)
    if len(values) >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
        return med, "IQR", f"{fmt(q1, nd)}–{fmt(q3, nd)}"
    if len(values) >= 2:
        return med, "min–max", f"{fmt(min(values), nd)}–{fmt(max(values), nd)}"
    return med, "n=1", ""


def cell(s, nd=1):
    if s is None:
        return "-"
    med, label, spread = s
    return fmt(med, nd) if not spread else f"{fmt(med, nd)} ({label} {spread})"


def group_key(r):
    return (r.get("model", ""), r.get("quant", ""), r.get("cpu_family", ""), r.get("llama_cpp_tag", ""))


def merge(rows):
    order, groups = [], {}
    for r in rows:
        k = group_key(r)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)

    out = []
    for k in order:
        members = groups[k]
        ok = [r for r in members if r.get("status") == "ok"]
        bad = [r for r in members if r.get("status") != "ok"]
        oom = [r for r in bad if is_oom(r)]
        pp, tg = stat(nums(ok, "pp512_tps")), stat(nums(ok, "tg128_tps"))
        ppl, hs = stat(nums(ok, "ppl"), 3), stat(nums(ok, "hellaswag_acc"), 1)
        rss = nums(members, "peak_rss_gb")
        g = {
            "model": k[0], "quant": k[1], "cpu_family": k[2], "llama_cpp_tag": k[3],
            "n_runs": len(members), "n_ok": len(ok), "n_failed": len(bad) - len(oom), "n_oom": len(oom),
            "low_n": len(ok) < 2,
            "pp512_tps_median": pp[0] if pp else None, "pp512_tps_spread": pp[2] if pp else "",
            "tg128_tps_median": tg[0] if tg else None, "tg128_tps_spread": tg[2] if tg else "",
            "ppl_median": ppl[0] if ppl else None, "ppl_spread": ppl[2] if ppl else "",
            "hellaswag_median": hs[0] if hs else None, "hellaswag_spread": hs[2] if hs else "",
            "peak_rss_gb_max": max(rss) if rss else None,
            "run_ids": sorted({str(r.get("run_id", "")) for r in members if r.get("run_id")}),
            "cpu_raw": sorted({r.get("cpu_raw", "") for r in members if r.get("cpu_raw")}),
            "_pp": pp, "_tg": tg, "_ppl": ppl, "_hs": hs,
        }
        out.append(g)
    return out


HEAD = [
    "| Model | Quant | CPU family | llama.cpp | n ok | failed | OOM | pp512 t/s median | tg128 t/s median | PPL median | HellaSwag % median | Peak RAM GB max | Flag |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
RAW_HEAD = [
    "| Run | Model | Quant | CPU (raw) | Status | pp512 | tg128 | PPL | HellaSwag % | Peak RAM GB |",
    "|---|---|---|---|---|---:|---:|---:|---:|---:|",
]


def render(groups, rows):
    lines = ["## Across runs (medians over successful runs)", "",
             "One row per (model, quant, CPU family, llama.cpp build). Medians use only status ok rows; "
             "failed / OOM runs are counted, never averaged in. Spread is the IQR for n ≥ 4, else min–max. "
             "Peak RAM is the max over all runs of the group.", ""]
    lines += HEAD
    for g in groups:
        flag = "⚠ n<2" if g["low_n"] else ""
        lines.append(
            f"| {g['model']} | {g['quant']} | {g['cpu_family']} | {g['llama_cpp_tag']} | {g['n_ok']} | "
            f"{g['n_failed']} | {g['n_oom']} | {cell(g['_pp'])} | {cell(g['_tg'])} | {cell(g['_ppl'], 3)} | "
            f"{cell(g['_hs'])} | {fmt(g['peak_rss_gb_max'], 2)} | {flag} |")
    lines.append("")
    runs = sorted({str(r.get("run_id", "")) for r in rows if r.get("run_id")})
    lines += [f"{len(rows)} rows from {len(runs)} run{'s' if len(runs) != 1 else ''}: {', '.join(runs)}", ""]
    lines += ["<details><summary>Raw rows</summary>", ""]
    lines += RAW_HEAD
    for r in rows:
        lines.append(
            f"| {r.get('run_id', '')} | {r.get('model', '')} | {r.get('quant', '')} | {r.get('cpu_raw') or '-'} | "
            f"{r.get('status', '?')} | {fmt(r.get('pp512_tps'))} | {fmt(r.get('tg128_tps'))} | "
            f"{fmt(r.get('ppl'), 3)} | {fmt(r.get('hellaswag_acc'))} | {fmt(r.get('peak_rss_gb'), 2)} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()

    rows = load_rows(a.paths)
    if not rows:
        sys.exit("merge.py: no result rows found")
    groups = merge(rows)
    md = render(groups, rows)
    public = [{k: v for k, v in g.items() if not k.startswith("_")} for g in groups]
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"groups": public, "rows": rows}, f, indent=2)
    print(md)


if __name__ == "__main__":
    main()
