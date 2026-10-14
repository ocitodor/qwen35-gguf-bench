#!/usr/bin/env bash
# One job: fetch a pinned llama.cpp release and one GGUF, start llama-server,
# answer the prompt(s) through /v1/chat/completions, write out/answer.md and
# out/answers.jsonl, stop the server.
#
# Environment (set by the workflow):
#   MODEL_REPO MODEL_FILE MODEL_NAME QUANT SIZE_GB LLAMA_CPP_TAG   as in bench/run_bench.sh
#   PROMPT        the question (single mode)
#   PROMPTS_FILE  a repo path: .txt one prompt per line, or .jsonl {"prompt": ..., "system": ...} (batch mode)
#   SYSTEM MAX_TOKENS TEMPERATURE THINKING(true/false)
#   HF_TOKEN      optional
#   THREADS CTX PORT  optional tuning
# The disk / llama.cpp / download parts mirror bench/run_bench.sh on purpose: the
# bench script stays untouched and verified.
set -euo pipefail

: "${MODEL_REPO:?}" "${MODEL_FILE:?}" "${MODEL_NAME:?}" "${QUANT:?}" "${LLAMA_CPP_TAG:?}"
SIZE_GB="${SIZE_GB:-0}"
THREADS="${THREADS:-$(nproc)}"
CTX="${CTX:-8192}"
PORT="${PORT:-8080}"
PROMPT="${PROMPT:-}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
THINKING="${THINKING:-false}"
export MODEL_NAME QUANT PROMPT PROMPTS_FILE SYSTEM MAX_TOKENS TEMPERATURE THINKING PORT

if [ -z "$PROMPT" ] && [ -z "$PROMPTS_FILE" ]; then
  echo "::error::give a prompt or a prompts_file"; exit 1
fi
if [ -n "$PROMPTS_FILE" ] && [ ! -s "$PROMPTS_FILE" ]; then
  echo "::error::prompts_file '$PROMPTS_FILE' is missing or empty in the repo"; exit 1
fi

WORK="$PWD"
MODEL_DIR="$WORK/models"
TOOL_DIR="$WORK/tools"
OUT="$WORK/out"
mkdir -p "$MODEL_DIR" "$TOOL_DIR" "$OUT"

avail_gb() { df --output=avail -B1 "$WORK" | tail -1 | awk '{printf "%.1f", $1/1e9}'; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < b)}'; }

echo "== ask ${MODEL_NAME} ${QUANT}  (${MODEL_REPO}/${MODEL_FILE}, ~${SIZE_GB} GB, llama.cpp ${LLAMA_CPP_TAG}, ${THREADS} threads, ctx ${CTX}, thinking ${THINKING})"

# --- disk ---------------------------------------------------------------------
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
if [ -s "$MODEL_PATH" ]; then
  NEED_GB=1.5
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
  echo "::error::needs ~${NEED_GB} GB of disk, only $(avail_gb) GB free after cleanup"; exit 1
fi

# --- llama.cpp prebuilt -------------------------------------------------------
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
  [ -n "$URL" ] || { echo "::error::no ubuntu-x64 asset on llama.cpp release ${LLAMA_CPP_TAG}"; exit 1; }
  ASSET="${URL##*/}"
  curl -fsSL --retry 3 -o "$TOOL_DIR/$ASSET" "$URL"
fi
case "$ASSET" in
  *.zip)    unzip -q -o "$TOOL_DIR/$ASSET" -d "$TOOL_DIR" ;;
  *.tar.gz) tar -xzf "$TOOL_DIR/$ASSET" -C "$TOOL_DIR" ;;
esac
SERVER_BIN=$(find "$TOOL_DIR" -type f -name llama-server | head -1 || true)
[ -n "$SERVER_BIN" ] || { echo "::error::llama-server not found inside $ASSET"; exit 1; }
chmod +x "$SERVER_BIN"

# --- model ----------------------------------------------------------------------
if [ -s "$MODEL_PATH" ]; then
  echo "model: present from cache"
else
  if [ -z "${HF_TOKEN:-}" ]; then
    unset HF_TOKEN
    echo "HF_TOKEN not set: anonymous download"
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
      "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
  fi
  echo "model: downloaded in $(( $(date +%s) - t0 ))s"
fi
if [ ! -s "$MODEL_PATH" ]; then
  found=$(find "$MODEL_DIR" -type f -name "$(basename "$MODEL_FILE")" | head -1 || true)
  [ -n "$found" ] && MODEL_PATH="$found"
fi
[ -s "$MODEL_PATH" ] || { echo "::error::model file missing after download: $MODEL_PATH"; exit 1; }
echo "model: $(stat -c %s "$MODEL_PATH" | awk '{printf "%.2f", $1/1e9}') GB on disk"

# --- server -----------------------------------------------------------------------
REASONING=off
[ "$THINKING" = "true" ] && REASONING=on
echo "llama-server: -c ${CTX} -t ${THREADS} --reasoning ${REASONING}, port ${PORT}"
"$SERVER_BIN" -m "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" -c "$CTX" -t "$THREADS" \
  --no-webui --reasoning "$REASONING" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 600); do
  if curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "server ready after ${i}s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "::error::llama-server exited while loading (see artifact out/server.log)"; tail -20 "$OUT/server.log"; exit 1
  fi
  sleep 1
done
curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "::error::llama-server not ready after 600 s"; exit 1; }

# --- ask ---------------------------------------------------------------------------
python3 ask/ask.py --out "$OUT"
