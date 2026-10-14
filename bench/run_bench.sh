#!/usr/bin/env bash
# One matrix job: fetch a pinned llama.cpp release and one GGUF, run llama-bench
# and llama-perplexity, write results/<model>-<quant>.json.
#
# Everything comes in through the environment (set by the workflow):
#   MODEL_REPO   HF repo, e.g. unsloth/Qwen3.5-4B-GGUF
#   MODEL_FILE   file inside that repo, e.g. Qwen3.5-4B-Q4_K_M.gguf
#   MODEL_NAME   label for the table, e.g. Qwen3.5-4B
#   QUANT        label for the table, e.g. Q4_K_M
#   SIZE_GB      expected file size, used only to plan disk space
#   NOTE         free text carried into the table (e.g. "speed-check only")
#   LLAMA_CPP_TAG  release tag of ggml-org/llama.cpp to download (prebuilt, no compiling)
#   HF_TOKEN     optional; anonymous downloads work for public repos
#   THREADS, PPL_CTX, PPL_CHUNKS, BENCH_REPS, PPL_FILE, RESULT_DIR  optional tuning
#
# A result JSON is written on every exit path (ok / failed / skipped) so the
# summary job can always show a row.
set -euo pipefail

: "${MODEL_REPO:?}" "${MODEL_FILE:?}" "${MODEL_NAME:?}" "${QUANT:?}" "${LLAMA_CPP_TAG:?}"
SIZE_GB="${SIZE_GB:-0}"
NOTE="${NOTE:-}"
THREADS="${THREADS:-$(nproc)}"
PPL_CTX="${PPL_CTX:-512}"
PPL_CHUNKS="${PPL_CHUNKS:-8}"
BENCH_REPS="${BENCH_REPS:-3}"
PPL_FILE="${PPL_FILE:-bench/ppl.txt}"
RESULT_DIR="${RESULT_DIR:-results}"
export THREADS PPL_CTX PPL_CHUNKS NOTE

WORK="$PWD"
MODEL_DIR="$WORK/models"
TOOL_DIR="$WORK/tools"
OUT="$WORK/out"
mkdir -p "$MODEL_DIR" "$TOOL_DIR" "$OUT" "$RESULT_DIR"
RESULT="$RESULT_DIR/${MODEL_NAME}-${QUANT}.json"

# Host CPU, recorded before anything can fail so a failed/skipped row still groups by host
# (cpu_info below stays as llama-bench reports it).
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || true)
[ -n "$CPU_MODEL" ] || CPU_MODEL=$(lscpu 2>/dev/null | grep -m1 'Model name' | cut -d: -f2- | xargs || true)
echo "host CPU: ${CPU_MODEL:-unknown}"

# --- result bookkeeping ------------------------------------------------------
STATUS="failed"; ERROR=""
PP_TPS=""; TG_TPS=""; PPL=""; PPL_ERR=""; PEAK_RSS_GB=""; FILE_GB=""
DOWNLOAD_S=""; BENCH_S=""; PPL_S=""; CPU_INFO=""

write_result() {
  export R_STATUS="$STATUS" R_ERROR="$ERROR" R_PP="$PP_TPS" R_TG="$TG_TPS" \
         R_PPL="$PPL" R_PPL_ERR="$PPL_ERR" R_RSS="$PEAK_RSS_GB" R_FILE_GB="$FILE_GB" \
         R_DL_S="$DOWNLOAD_S" R_BENCH_S="$BENCH_S" R_PPL_S="$PPL_S" R_CPU="$CPU_INFO" \n         R_CPU_MODEL="$CPU_MODEL"
  python3 - "$RESULT" <<'PY'
import datetime, json, os, sys
e = os.environ
def num(k):
    v = e.get(k, "")
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None
out = {
    "model": e["MODEL_NAME"], "quant": e["QUANT"],
    "repo": e["MODEL_REPO"], "file": e["MODEL_FILE"], "note": e.get("NOTE", ""),
    "status": e["R_STATUS"], "error": e["R_ERROR"],
    "file_gb": num("R_FILE_GB"),
    "pp512_tps": num("R_PP"), "tg128_tps": num("R_TG"),
    "ppl": num("R_PPL"), "ppl_err": num("R_PPL_ERR"),
    "peak_rss_gb": num("R_RSS"),
    "download_s": num("R_DL_S"), "bench_s": num("R_BENCH_S"), "ppl_s": num("R_PPL_S"),
    "threads": int(e.get("THREADS") or 0), "cpu_info": e.get("R_CPU", ""),
    "cpu_model": e.get("R_CPU_MODEL", ""),
    "llama_cpp_tag": e["LLAMA_CPP_TAG"],
    "ppl_ctx": int(e.get("PPL_CTX") or 0), "ppl_chunks": int(e.get("PPL_CHUNKS") or 0),
    "runner_image": (e.get("ImageOS", "") + " " + e.get("ImageVersion", "")).strip(),
    "run_id": e.get("GITHUB_RUN_ID", ""),
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY
}

# fail "<message>" [status]  -> writes the result and exits 1
fail() {
  ERROR="$1"; STATUS="${2:-failed}"
  echo "::error::${MODEL_NAME} ${QUANT}: ${ERROR}"
  write_result
  exit 1
}
# Any other failing command: still leave a result row behind.
trap 'if [ -z "$ERROR" ]; then ERROR="command failed (see job log)"; write_result; fi' ERR

avail_gb() { df --output=avail -B1 "$WORK" | tail -1 | awk '{printf "%.1f", $1/1e9}'; }
# lt A B -> true if A < B (floats)
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < b)}'; }

echo "== ${MODEL_NAME} ${QUANT}  (${MODEL_REPO}/${MODEL_FILE}, ~${SIZE_GB} GB, llama.cpp ${LLAMA_CPP_TAG}, ${THREADS} threads)"

# --- disk ---------------------------------------------------------------------
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
if [ -s "$MODEL_PATH" ]; then
  NEED_GB=1.5                                     # cache hit: only llama.cpp + logs
else
  NEED_GB=$(awk -v s="$SIZE_GB" 'BEGIN{printf "%.1f", s*1.05 + 1.5}')
fi
echo "disk: need ~${NEED_GB} GB, available $(avail_gb) GB"
if lt "$(avail_gb)" "$(awk -v n="$NEED_GB" 'BEGIN{print n+2}')"; then
  echo "freeing disk: removing preinstalled SDKs this job never uses"
  sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup \
              /opt/hostedtoolcache/CodeQL /usr/local/share/powershell /usr/share/swift 2>/dev/null || true
  sudo docker image prune -af >/dev/null 2>&1 || true
  sudo apt-get clean >/dev/null 2>&1 || true
  echo "disk after cleanup: $(avail_gb) GB available"
fi
if lt "$(avail_gb)" "$NEED_GB"; then
  fail "needs ~${NEED_GB} GB of disk, only $(avail_gb) GB free after cleanup" skipped
fi

# --- llama.cpp prebuilt -------------------------------------------------------
# Binaries live on the bNNNNN build tags; the CPU asset is
# llama-<tag>-bin-ubuntu-x64.tar.gz (verified on b11429). If that name 404s
# (asset renamed), ask the release API for whatever ubuntu-x64 asset the tag carries.
ASSET="llama-${LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_TAG}/${ASSET}"
echo "llama.cpp: $URL"
if ! curl -fsSL --retry 3 -o "$TOOL_DIR/$ASSET" "$URL"; then
  echo "direct download failed, looking the asset up through the release API"
  gh_auth=()
  [ -n "${GH_TOKEN:-}" ] && gh_auth=(-H "Authorization: Bearer $GH_TOKEN")
  URL=$(curl -fsSL --retry 3 "${gh_auth[@]}" -H "Accept: application/vnd.github+json" \
          "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${LLAMA_CPP_TAG}" \
        | python3 -c '
import json, sys
assets = json.load(sys.stdin).get("assets", [])
hits = [a["browser_download_url"] for a in assets
        if "ubuntu-x64" in a["name"] and "vulkan" not in a["name"]
        and (a["name"].endswith(".zip") or a["name"].endswith(".tar.gz"))]
print(hits[0] if hits else "")') || URL=""
  [ -n "$URL" ] || fail "no ubuntu-x64 asset found on llama.cpp release ${LLAMA_CPP_TAG}"
  ASSET="${URL##*/}"
  echo "llama.cpp: $URL"
  curl -fsSL --retry 3 -o "$TOOL_DIR/$ASSET" "$URL" || fail "could not download $URL"
fi
case "$ASSET" in
  *.zip)    unzip -q -o "$TOOL_DIR/$ASSET" -d "$TOOL_DIR" ;;
  *.tar.gz) tar -xzf "$TOOL_DIR/$ASSET" -C "$TOOL_DIR" ;;
  *)        fail "unknown archive type: $ASSET" ;;
esac
BENCH_BIN=$(find "$TOOL_DIR" -type f -name llama-bench | head -1 || true)
PPL_BIN=$(find "$TOOL_DIR" -type f -name llama-perplexity | head -1 || true)
[ -n "$BENCH_BIN" ] && [ -n "$PPL_BIN" ] || fail "llama-bench / llama-perplexity not found inside $ASSET"
chmod +x "$BENCH_BIN" "$PPL_BIN"

# --- model ----------------------------------------------------------------------
if [ -s "$MODEL_PATH" ]; then
  echo "model: present from cache"
  DOWNLOAD_S=0
else
  if [ -z "${HF_TOKEN:-}" ]; then
    unset HF_TOKEN
    echo "HF_TOKEN not set: anonymous download (works for public repos, may be rate-limited)"
  fi
  t0=$(date +%s)
  if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    python3 -m pip install -q -U "huggingface_hub[cli]" \
      || python3 -m pip install -q -U --user --break-system-packages "huggingface_hub[cli]" || true
    export PATH="$HOME/.local/bin:$PATH"
  fi
  dl_ok=0
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR" && dl_ok=1
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR" && dl_ok=1
  fi
  if [ "$dl_ok" != 1 ] || [ ! -s "$MODEL_PATH" ]; then
    echo "hf CLI download did not produce the file, falling back to curl"
    auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    curl -fL --retry 3 -C - "${auth[@]}" -o "$MODEL_PATH" \
      "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
      || fail "download failed: ${MODEL_REPO}/${MODEL_FILE}"
  fi
  DOWNLOAD_S=$(( $(date +%s) - t0 ))
fi
if [ ! -s "$MODEL_PATH" ]; then
  # a repo path with a folder lands at models/<folder>/<file>: point at it
  found=$(find "$MODEL_DIR" -type f -name "$(basename "$MODEL_FILE")" | head -1 || true)
  [ -n "$found" ] && MODEL_PATH="$found"
fi
[ -s "$MODEL_PATH" ] || fail "model file missing after download: $MODEL_PATH"
FILE_GB=$(stat -c %s "$MODEL_PATH" | awk '{printf "%.2f", $1/1e9}')
echo "model: ${FILE_GB} GB on disk, downloaded in ${DOWNLOAD_S}s"

# --- llama-bench: pp512 + tg128 ---------------------------------------------------
echo "llama-bench: -p 512 -n 128 -t ${THREADS} -r ${BENCH_REPS}"
python3 bench/measure.py --stdout "$OUT/bench.json" --stderr "$OUT/bench.log" --meta "$OUT/bench.meta.json" -- \
  "$BENCH_BIN" -m "$MODEL_PATH" -p 512 -n 128 -t "$THREADS" -r "$BENCH_REPS" -o json \
  || fail "llama-bench failed (see artifact out/bench.log)"
python3 - "$OUT/bench.json" > "$OUT/bench.parsed" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
pp = tg = cpu = ""
for r in rows:
    if r.get("n_prompt") and not r.get("n_gen"):
        pp = f'{r["avg_ts"]:.2f}'
    if r.get("n_gen") and not r.get("n_prompt"):
        tg = f'{r["avg_ts"]:.2f}'
    cpu = cpu or " ".join((r.get("cpu_info") or "").split())
print(pp); print(tg); print(cpu)   # one value per line: an empty one cannot shift the others
PY
{ read -r PP_TPS; read -r TG_TPS; read -r CPU_INFO; } < "$OUT/bench.parsed"
BENCH_S=$(python3 -c "import json;print(json.load(open('$OUT/bench.meta.json'))['elapsed_s'])")
[ -n "$PP_TPS" ] && [ -n "$TG_TPS" ] || fail "could not parse llama-bench JSON output"
echo "pp512 ${PP_TPS} t/s, tg128 ${TG_TPS} t/s, ${BENCH_S}s"

# --- llama-perplexity on the fixed text ------------------------------------------
echo "llama-perplexity: -c ${PPL_CTX} --chunks ${PPL_CHUNKS} on ${PPL_FILE}"
python3 bench/measure.py --stdout "$OUT/ppl.out" --stderr "$OUT/ppl.log" --meta "$OUT/ppl.meta.json" -- \
  "$PPL_BIN" -m "$MODEL_PATH" -f "$PPL_FILE" -c "$PPL_CTX" -b "$PPL_CTX" --chunks "$PPL_CHUNKS" -t "$THREADS" \
  || fail "llama-perplexity failed (see artifact out/ppl.log)"
PPL_LINE=$(grep -h -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$OUT/ppl.out" "$OUT/ppl.log" | tail -1 || true)
[ -n "$PPL_LINE" ] || fail "no 'Final estimate' line in llama-perplexity output"
PPL=$(echo "$PPL_LINE" | awk '{print $5}')
PPL_ERR=$(echo "$PPL_LINE" | awk '{print $7}')
PPL_S=$(python3 -c "import json;print(json.load(open('$OUT/ppl.meta.json'))['elapsed_s'])")
echo "PPL ${PPL} +/- ${PPL_ERR}, ${PPL_S}s"

PEAK_RSS_GB=$(python3 - "$OUT/bench.meta.json" "$OUT/ppl.meta.json" <<'PY'
import json, sys
print(max(json.load(open(p))["max_rss_gb"] for p in sys.argv[1:]))
PY
)

STATUS="ok"
write_result
echo "::notice::${MODEL_NAME} ${QUANT}: pp512 ${PP_TPS} t/s, tg128 ${TG_TPS} t/s, PPL ${PPL}, peak RSS ${PEAK_RSS_GB} GB"
