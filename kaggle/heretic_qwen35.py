#!/usr/bin/env python3
"""Kaggle GPU kernel: fresh Heretic abliteration of a Qwen3.5 model -> Q4_K_M GGUF -> (optional) HF upload.

Runs as a Kaggle *script* kernel (pushed by kaggle/push.py, which fills the CONFIG block below).
Needs: GPU (T4 x2 or P100), Internet on. Optional Kaggle secret HF_TOKEN (attached to this
notebook in the Kaggle UI: Add-ons -> Secrets) for the upload; without it the GGUF stays in the
kernel output only.

Phases
  0. environment: GPU info, install Heretic from a pinned master commit (PyPI 1.4.0 has no
     headless mode; PR #392 is on master since 27 Jun 2026), llama.cpp b11429 (same build as the bench).
  1. optimize: `heretic --model M --n-trials N ... --trial-index 0 --model-action ""`
     runs the Optuna study (checkpointed to /kaggle/working/checkpoints) and exits without exporting.
  2. pick: read the checkpoint, sort the Pareto front the way Heretic does (refusals, then KL,
     ascending), and choose the lowest-KL trial whose refusals are within REFUSAL_SLACK of the
     best one. Heretic's own index 0 is simply "fewest refusals", whatever its KL.
  3. export: `heretic --checkpoint-action continue --trial-index K --model-action save
     --export-strategy merge --save-directory /kaggle/tmp/model` (merged safetensors).
  4. GGUF: convert_hf_to_gguf.py (text model only, bf16) -> llama-quantize Q4_K_M -> a quick
     llama-perplexity on the bench's ppl.txt as a sanity check against the plain model's PPL.
  5. upload (only if HF_TOKEN is attached and UPLOAD_REPO is set): GGUF + a model card with the
     chosen trial's parameters and scores, private repo by default.

Everything lands in /kaggle/working (kernel output, 20 GB cap): the Q4_K_M GGUF, logs, the
checkpoint and pick.json. Big intermediates (merged safetensors, bf16 GGUF) live in /kaggle/tmp.
"""
import json
import os
import shlex
import subprocess
import sys
import time

# ---- CONFIG (kaggle/push.py rewrites these lines) -------------------------------------------
MODEL = "Qwen/Qwen3.5-4B"                 # HF base model
OUT_NAME = "Qwen3.5-4B-heretic-ara"       # GGUF base name
N_TRIALS = 100                            # Heretic master default
N_STARTUP_TRIALS = 30                     # Heretic master default
SEED = 42
REFUSAL_SLACK = 3                         # accept up to best_refusals + 3 (of the refusal prompts) for lower KL
HERETIC_REF = "66a952b"                   # p-e-w/heretic master commit (headless mode)
LLAMA_CPP_TAG = "b11429"                  # same build as the bench
FLA_VERSION = "0.5.2"                     # flash-linear-attention: fast Gated DeltaNet kernels
UPLOAD_REPO = ""                          # e.g. "<hf user>/Qwen3.5-4B-heretic-ara-GGUF"; empty = no upload
UPLOAD_PRIVATE = True
# ----------------------------------------------------------------------------------------------

WORK = "/kaggle/working"
TMP = "/kaggle/tmp"
CKPT = f"{WORK}/checkpoints"
MODEL_DIR = f"{TMP}/model"
LOG = f"{WORK}/run.log"
T0 = time.time()


def log(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, logfile=None, check=True, env=None):
    """Run a command, stream it to stdout (and a log file), never attached to a TTY."""
    if isinstance(cmd, str):
        shown, shell = cmd, True
    else:
        shown, shell = " ".join(shlex.quote(c) for c in cmd), False
    log(f"$ {shown}")
    out = open(logfile, "a", encoding="utf-8") if logfile else None
    p = subprocess.Popen(cmd, shell=shell, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in p.stdout:
        sys.stdout.write(line)
        if out:
            out.write(line)
    rc = p.wait()
    if out:
        out.close()
    if check and rc != 0:
        raise SystemExit(f"command failed ({rc}): {shown}")
    return rc


def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN") or ""
    except Exception as e:  # not attached, or not on Kaggle
        log(f"HF_TOKEN secret not available ({type(e).__name__}); no upload")
        return ""


def heretic(args, logfile):
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["TERM"] = "dumb"
    return run(["heretic", "--model", MODEL, "--study-checkpoint-dir", CKPT, *args], logfile=logfile, env=env)


PREFLIGHT = r'''
import json, sys, time, torch
from transformers import AutoTokenizer
from heretic.model import get_model_class
model_id, dtype = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_id)
model = get_model_class(model_id).from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="auto")
msgs = [[{"role": "user", "content": q}] for q in
        ["What is the capital of France? One word.", "Name the largest planet in the solar system. One word.",
         "What is 12 times 12? Just the number.", "Which ocean lies between Europe and America? One word."] * 4]
text = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
tok.padding_side = "left"
enc = tok(text, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    logits = model(**enc).logits[:, -1, :]
    finite = bool(torch.isfinite(logits).all())
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=32, do_sample=False)
    dt = time.time() - t0
answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
expect = ["paris", "jupiter", "144", "atlantic"]
correct = sum(e in a.lower() for e, a in zip(expect * 4, answers))
new = (out[:, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum().item()
print(json.dumps({"dtype": dtype, "finite_logits": finite, "correct": correct, "of": len(answers),
                  "tokens_per_s_batch16": round(new / dt, 1), "sample": answers[:4]}))
'''


def preflight():
    """float16 is fast on a T4 (no native bf16), but Qwen can overflow in fp16: check before 100 trials."""
    path = f"{TMP}/preflight.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREFLIGHT)
    for dtype in ("float16", "bfloat16"):
        r = subprocess.run([sys.executable, path, MODEL, dtype], capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        log(f"preflight {dtype}: rc {r.returncode} {tail[:600]}")
        if r.returncode != 0:
            with open(f"{WORK}/preflight_{dtype}.log", "w", encoding="utf-8") as f:
                f.write(r.stdout + r.stderr)
            continue
        res = json.loads(tail)
        if res["finite_logits"] and res["correct"] >= res["of"] - 2:
            log(f"using dtype {dtype} ({res['tokens_per_s_batch16']} tokens/s at batch 16)")
            return [dtype]
    raise SystemExit("preflight failed in float16 and bfloat16: see preflight_*.log")


def pick_trial():
    """Replicate Heretic's Pareto sort and pick lowest KL within REFUSAL_SLACK of the fewest refusals."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    files = [f for f in os.listdir(CKPT) if f.endswith(".jsonl")]
    assert len(files) == 1, f"expected one checkpoint in {CKPT}, found {files}"
    path = os.path.join(CKPT, files[0])
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    study = optuna.load_study(study_name=storage.get_all_studies()[0].study_name, storage=storage)

    def scores(t):
        return {s["name"]: s["score"]["value"] for s in t.user_attrs["scores"]}

    names = list(scores(study.best_trials[0]).keys())
    refusal_name = next((n for n in names if "keyword" in n.lower() or "refus" in n.lower()), names[0])
    kl_name = next((n for n in names if "kl" in n.lower()), names[-1])
    # Heretic: sorted(study.best_trials, key=tuple of score values in objective order) - objective
    # order is the scorer order (KeywordRate, KLDivergence) = the order of user_attrs["scores"].
    front = sorted(study.best_trials, key=lambda t: tuple(scores(t)[n] for n in names))
    best_ref = min(scores(t)[refusal_name] for t in front)
    ok = [i for i, t in enumerate(front) if scores(t)[refusal_name] <= best_ref + REFUSAL_SLACK]
    k = min(ok, key=lambda i: scores(front[i])[kl_name])

    table = [{"pareto_index": i, "trial": t.user_attrs.get("index"), **scores(t),
              "parameters": t.user_attrs.get("parameters")} for i, t in enumerate(front)]
    pick = {"chosen_pareto_index": k, "rule": f"lowest {kl_name} with {refusal_name} <= {best_ref} + {REFUSAL_SLACK}",
            "objectives": names, "n_trials_completed": len(study.trials),
            "finished": study.user_attrs.get("finished"), "front": table}
    with open(f"{WORK}/pick.json", "w", encoding="utf-8") as f:
        json.dump(pick, f, indent=2, default=str)
    log(f"Pareto front ({len(front)} trials), objectives {names}:")
    for row in table:
        mark = "  <== chosen" if row["pareto_index"] == k else ""
        log(f"  [{row['pareto_index']:>2}] trial {row['trial']:>3}  " +
            "  ".join(f"{n}={row[n]:.4g}" for n in names) + mark)
    return k, pick


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    log(f"config: model {MODEL}, {N_TRIALS} trials ({N_STARTUP_TRIALS} startup), seed {SEED}, "
        f"heretic {HERETIC_REF}, llama.cpp {LLAMA_CPP_TAG}, upload '{UPLOAD_REPO or '-'}'")
    run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", check=False)
    run("df -h /kaggle/working /kaggle/tmp /tmp | cat; free -g", check=False)

    # ---- 0. environment ----------------------------------------------------------------------
    run([sys.executable, "-m", "pip", "install", "-q", f"git+https://github.com/p-e-w/heretic@{HERETIC_REF}"])
    # Qwen3.5's Gated DeltaNet layers use flash-linear-attention's Triton kernels when `fla` is
    # importable; without it transformers falls back to plain torch: 9-15 tokens/s on a T4 in kernel v1.
    # The base package needs only einops, so Kaggle's CUDA torch stays.
    run([sys.executable, "-m", "pip", "install", "-q", f"flash-linear-attention=={FLA_VERSION}"])
    run([sys.executable, "-c", "import fla, torch; print('fla', fla.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"])
    run(["heretic", "--help"], logfile=f"{WORK}/heretic_help.txt")
    dtypes = preflight()
    run(f"git clone -q --depth 1 --branch {LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp {TMP}/llama.cpp")
    # converter deps without its pinned CPU torch (Heretic's CUDA torch is already installed)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{TMP}/llama.cpp/gguf-py", "sentencepiece", "protobuf"])
    asset = f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz"
    run(f"curl -fsSL --retry 3 -o {TMP}/{asset} https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}/{asset}"
        f" && mkdir -p {TMP}/bin && tar -xzf {TMP}/{asset} -C {TMP}/bin")
    qbin = subprocess.check_output(f"find {TMP}/bin -type f -name llama-quantize | head -1", shell=True, text=True).strip()
    pbin = subprocess.check_output(f"find {TMP}/bin -type f -name llama-perplexity | head -1", shell=True, text=True).strip()
    assert qbin and pbin, "llama-quantize / llama-perplexity not in the release archive"
    run(f"chmod +x {qbin} {pbin}")

    # ---- 1. optimize (no export) ---------------------------------------------------------------
    heretic(["--dtypes", json.dumps(dtypes),
             "--n-trials", str(N_TRIALS), "--n-startup-trials", str(N_STARTUP_TRIALS), "--seed", str(SEED),
             "--checkpoint-action", "restart", "--trial-index", "0", "--model-action", ""],
            logfile=f"{WORK}/heretic_optimize.log")

    # ---- 2. pick -----------------------------------------------------------------------------------
    k, pick = pick_trial()

    # ---- 3. export the chosen trial (merged) ------------------------------------------------------
    heretic(["--checkpoint-action", "continue", "--trial-index", str(k), "--model-action", "save",
             "--export-strategy", "merge", "--save-directory", MODEL_DIR],
            logfile=f"{WORK}/heretic_export.log")
    run(f"ls -la {MODEL_DIR}; du -sh {MODEL_DIR}", check=False)

    # ---- 4. GGUF ------------------------------------------------------------------------------------
    bf16 = f"{TMP}/{OUT_NAME}.bf16.gguf"
    q4 = f"{WORK}/{OUT_NAME}.Q4_K_M.gguf"
    run([sys.executable, f"{TMP}/llama.cpp/convert_hf_to_gguf.py", MODEL_DIR, "--outtype", "bf16", "--outfile", bf16],
        logfile=f"{WORK}/convert.log")
    run([qbin, bf16, q4, "Q4_K_M"], logfile=f"{WORK}/quantize.log")
    os.remove(bf16)
    size_gb = os.path.getsize(q4) / 1e9
    log(f"GGUF: {q4} ({size_gb:.2f} GB)")
    run(f"curl -fsSL -o {TMP}/ppl.txt https://raw.githubusercontent.com/ocitodor/qwen35-gguf-bench/main/bench/ppl.txt")
    run([pbin, "-m", q4, "-f", f"{TMP}/ppl.txt", "-c", "512", "-b", "512", "--chunks", "8", "-ngl", "0"],
        logfile=f"{WORK}/ppl.log", check=False)
    ppl = subprocess.run(f"grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' {WORK}/ppl.log | tail -1",
                         shell=True, capture_output=True, text=True).stdout.strip()
    log(f"sanity PPL (bench text, ctx 512 x 8, CPU): {ppl or 'n/a'}  (plain Qwen3.5-4B Q4_K_M = 7.119)")

    summary = {"model": MODEL, "gguf": os.path.basename(q4), "size_gb": round(size_gb, 2), "ppl": ppl,
               "heretic_ref": HERETIC_REF, "llama_cpp_tag": LLAMA_CPP_TAG, "seed": SEED,
               "chosen": pick["front"][k], "rule": pick["rule"], "minutes": round((time.time() - T0) / 60, 1)}
    with open(f"{WORK}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- 5. upload ------------------------------------------------------------------------------------
    token = hf_token() if UPLOAD_REPO else ""
    if token:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(UPLOAD_REPO, repo_type="model", private=UPLOAD_PRIVATE, exist_ok=True)
        card = (f"---\nbase_model: {MODEL}\ntags: [heretic, abliterated, gguf]\n---\n\n"
                f"# {OUT_NAME} (GGUF Q4_K_M)\n\nFresh Heretic run (p-e-w/heretic@{HERETIC_REF}, default ARA modifier, "
                f"{N_TRIALS} trials, seed {SEED}) on {MODEL}; trial chosen by: {pick['rule']}. Converted and quantized with "
                f"llama.cpp {LLAMA_CPP_TAG}.\n\nChosen trial:\n\n```json\n{json.dumps(pick['front'][k], indent=2, default=str)}\n```\n\n"
                f"Sanity PPL on a fixed 4k-token text: {ppl}.\n")
        with open(f"{TMP}/README.md", "w", encoding="utf-8") as f:
            f.write(card)
        api.upload_file(path_or_fileobj=f"{TMP}/README.md", path_in_repo="README.md", repo_id=UPLOAD_REPO)
        api.upload_file(path_or_fileobj=q4, path_in_repo=os.path.basename(q4), repo_id=UPLOAD_REPO)
        api.upload_file(path_or_fileobj=f"{WORK}/pick.json", path_in_repo="heretic_pick.json", repo_id=UPLOAD_REPO)
        log(f"uploaded to https://huggingface.co/{UPLOAD_REPO} ({'private' if UPLOAD_PRIVATE else 'public'})")
    else:
        log("no upload: the GGUF is in this kernel's output (/kaggle/working)")
    log("done")


if __name__ == "__main__":
    main()
