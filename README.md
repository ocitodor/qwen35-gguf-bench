# Qwen 3.5 GGUF bench on GitHub Actions

Benchmarks a list of quantized Qwen 3.5 GGUF files on free GitHub-hosted runners —
one job per file, in parallel — and writes a single comparison table to the run
summary. Nothing runs anywhere else: no local machine, no GPU, no paid service.

Each job downloads a pinned prebuilt `llama.cpp` release (no compiling) and one GGUF,
then runs

- `llama-bench` — prompt processing at 512 tokens (**pp512**) and generation of 128
  tokens (**tg128**), tokens/s, threads = all runner cores;
- `llama-perplexity` — on the fixed text `bench/ppl.txt`, 8 chunks × 512 tokens, so it
  ends in minutes on a CPU. **PPL is relative**: compare rows against each other, not
  against published wikitext numbers;
- peak RAM of the two runs (resident set size, GB).

Runner (public repo, `ubuntu-latest`): 4 vCPU, 16 GB RAM, 14 GB SSD, 6 h max per job,
free and unlimited minutes. Up to 20 jobs run at once.

## Default model list

Edit **`bench/models.json`** — the only place the list lives.

| Model | Quant | File | GB | Cached |
|---|---|---|---:|---|
| Qwen3.5-0.8B | Q4_K_M | `Qwen3.5-0.8B-Q4_K_M.gguf` | 0.53 | yes |
| Qwen3.5-0.8B | Q8_0 | `Qwen3.5-0.8B-Q8_0.gguf` | 0.81 | yes |
| Qwen3.5-2B | Q4_K_M | `Qwen3.5-2B-Q4_K_M.gguf` | 1.28 | yes |
| Qwen3.5-2B | Q8_0 | `Qwen3.5-2B-Q8_0.gguf` | 2.01 | yes |
| Qwen3.5-4B | Q4_K_M | `Qwen3.5-4B-Q4_K_M.gguf` | 2.74 | yes |
| Qwen3.5-4B | Q8_0 | `Qwen3.5-4B-Q8_0.gguf` | 4.48 | no |
| Qwen3.5-9B | Q4_K_M | `Qwen3.5-9B-Q4_K_M.gguf` | 5.68 | no |
| Qwen3.5-9B | Q8_0 | `Qwen3.5-9B-Q8_0.gguf` | 9.53 | no |
| Qwen3.5-35B-A3B | UD-Q2_K_XL | `Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf` | 12.2 | no — speed-check only |

All from the `unsloth/<model>-GGUF` repos on Hugging Face. The 35B-A3B row is there to
measure speed of the MoE on a CPU; Q2 quality is degraded, and its Q4 (22 GB) does not
fit a 16 GB runner — run that one on Kaggle (2×T4 or the 30 GB CPU session) instead.

Each entry:

```json
{"model": "Qwen3.5-4B", "quant": "Q4_K_M",
 "repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-Q4_K_M.gguf",
 "size_gb": 2.74, "cache": true, "timeout_min": 45, "note": ""}
```

- `model`, `quant`, `repo`, `file` — required. `model` + `quant` must be unique; they name
  the job, the artifact and the table row.
- `size_gb` — expected size, used only to plan disk space (defaults to 0 = no planning).
- `cache` — keep the file in the Actions cache between runs. The whole repository shares
  a 10 GB cache, so only mark small files; the defaults total ~7.4 GB.
- `timeout_min` — per-job timeout (default 60).
- `note` — free text shown in the Status column.

What fits a 16 GB / 14 GB-disk runner: anything up to ~13.5 GB on disk (the job frees
~25 GB by deleting preinstalled SDKs when needed) and up to ~14 GB in RAM. A file that
does not fit is reported as `skipped` with the numbers, not as a crash.

## Set up (once)

1. Create a **public** repository on your free GitHub account (public = free, unlimited
   minutes on the 4-core runner; a private repo gets 2,000 min/month on a 2-core/8 GB one).
2. Copy the contents of this folder to the repository root, including the hidden
   `.github/` directory:

   ```bash
   cp -r qwen-bench/. /path/to/new-repo/
   cd /path/to/new-repo && git add -A && git commit -m "Qwen 3.5 GGUF bench" && git push
   ```

3. Optional but recommended — add a Hugging Face token so downloads are not
   rate-limited:
   - get a token with **Read** scope at <https://huggingface.co/settings/tokens>;
   - in the repository: **Settings → Secrets and variables → Actions → New repository
     secret**, name `HF_TOKEN`, paste the token.

   Without the secret the workflow still works; downloads are anonymous.

## Run

**Actions → Qwen 3.5 GGUF bench → Run workflow.** Leave the input empty for the default
list, or paste a JSON array in the same format to bench something else just this once:

```json
[{"model":"Qwen3.5-4B","quant":"UD-Q4_K_XL","repo":"unsloth/Qwen3.5-4B-GGUF","file":"Qwen3.5-4B-UD-Q4_K_XL.gguf","size_gb":2.8}]
```

The workflow never runs on push or on a schedule.

Where the results are:

- the run's **Summary** page — the planned list at the top, the comparison tables at the
  bottom (`Comparison table` job), with model, quant, file GB, pp512 t/s, tg128 t/s, PPL,
  peak RAM GB and status. Rows are grouped into one table per runner **CPU family**
  (`AMD EPYC 7763`, `AMD EPYC 9V45`, ...): jobs land on different hosts, so speed is
  comparable only within a group, while PPL and RAM compare across groups;
- artifacts: `qwen-bench-summary` (the table as `summary.md` + `combined.json`), one
  `result-<model>-<quant>` JSON per job (kept 30 days) and one `log-<model>-<quant>` with
  the raw llama.cpp output (kept 7 days) for anything that failed.

A typical default run finishes in 10–20 minutes wall clock; the 9B Q8_0 and the
35B-A3B jobs are the slow ones.

## Tuning

Top of `.github/workflows/qwen-bench.yml`:

- `LLAMA_CPP_TAG` — the pinned `ggml-org/llama.cpp` **build tag** (`bNNNNN`); the job
  downloads `llama-<tag>-bin-ubuntu-x64.tar.gz` from that release. The binaries live only
  on the build tags — a versioned release such as `v0.6.0` carries no binaries, its page
  just says which build it corresponds to ("Nightly build: bNNNNN"). Bump the tag to
  pick up a newer build; if the asset is ever renamed the job falls back to asking the
  release API for the ubuntu-x64 asset.
- `PPL_CTX` / `PPL_CHUNKS` — perplexity context and chunk count (8 × 512 = 4096 tokens
  of `bench/ppl.txt`, which holds ~4,700 tokens).
- `BENCH_REPS` — llama-bench repetitions per test.

## Reading the numbers

- Runner CPUs vary between jobs, even inside one run (different Azure hosts), so compare
  speeds only **within one CPU group**; each result JSON carries `cpu_model` (from
  `/proc/cpuinfo`, recorded before anything can fail) and `cpu_family` (the grouping key).
- pp512 is what matters for long prompts, tg128 for chat-style generation.
- PPL on a 4k-token text has a visible error bar (shown as ±); differences smaller than
  the error bar are noise.
- Peak RAM is the resident set of the llama.cpp process; with memory-mapped GGUFs it is
  roughly the file size plus the KV cache.

## Files

```
.github/workflows/qwen-bench.yml   the workflow: plan -> bench (matrix) -> summary
bench/models.json                  the default model list (edit this)
bench/run_bench.sh                 one job: disk, llama.cpp, download, bench, perplexity, result JSON
bench/measure.py                   runs a command, records wall time and peak RSS
bench/summarize.py                 result JSONs -> markdown table + combined.json
bench/ppl.txt                      fixed text for perplexity (~3,800 words)
```
