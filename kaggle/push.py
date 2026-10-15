#!/usr/bin/env python3
"""Push kaggle/heretic_qwen35.py as a private Kaggle GPU script kernel, or check on it.

Usage (on the PC, needs the Kaggle API token and `pip install kaggle`):
  python kaggle/push.py push --size 4B [--upload-repo USER/Qwen3.5-4B-heretic-ara-GGUF] [--public-upload]
  python kaggle/push.py status --size 4B
  python kaggle/push.py output --size 4B --dest DIR      # logs + pick.json + summary.json (+ the GGUF)

Credentials, in C:\\Scripts\\secrets\\kaggle (outside every repo), never printed:
- kaggle.txt: a new-style API token (KGAT_...), passed to the kaggle process only as
  KAGGLE_API_TOKEN; the username is USER below (ocitodor);
- or kaggle.json: a legacy token {"username", "key"} (KAGGLE_CONFIG_DIR).
The kaggle CLI comes from the repo's .venv when there is one (pip install kaggle).
The kernel is private, GPU T4 x2, Internet on. Attach the HF_TOKEN secret to it once in the
Kaggle UI (Add-ons -> Secrets) if it should upload.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SECRETS = r"C:\Scripts\secrets\kaggle"


USER = "ocitodor"
ENV = dict(os.environ)


def kaggle_user():
    """Set up the kaggle child's credentials (never printed) and return the username."""
    txt, js = os.path.join(SECRETS, "kaggle.txt"), os.path.join(SECRETS, "kaggle.json")
    if os.path.exists(txt):
        with open(txt, encoding="utf-8-sig") as f:
            ENV["KAGGLE_API_TOKEN"] = f.read().strip()
        return USER
    if os.path.exists(js):
        ENV["KAGGLE_CONFIG_DIR"] = SECRETS
        with open(js, encoding="utf-8") as f:
            return json.load(f)["username"]
    sys.exit(f"no Kaggle API token in {SECRETS} (Kaggle -> Settings -> API Tokens)")


def python():
    venv = os.path.join(os.path.dirname(HERE), ".venv", "Scripts", "python.exe")
    return venv if os.path.exists(venv) else sys.executable


def slug(size):
    return f"qwen35-{size.lower()}-heretic-ara"


def kaggle(*args):
    cmd = [python(), "-m", "kaggle.cli", *args]
    print("$ kaggle " + " ".join(args), flush=True)
    return subprocess.call(cmd, env=ENV)


def push(a):
    user = kaggle_user()
    src = open(os.path.join(HERE, "heretic_qwen35.py"), encoding="utf-8").read()
    subs = {
        "MODEL": f'"Qwen/Qwen3.5-{a.size}"',
        "OUT_NAME": f'"Qwen3.5-{a.size}-heretic-ara"',
        "UPLOAD_REPO": f'"{a.upload_repo or ""}"',
        "UPLOAD_PRIVATE": "False" if a.public_upload else "True",
        "RESUME": "True" if a.resume_from else "False",
    }
    sources = []
    if a.resume_from:
        sources = [make_checkpoint_dataset(user, a.size, a.resume_from)]
    for name, value in subs.items():
        src, n = re.subn(rf"^{name} = .*?(\s+#.*)?$", lambda m: f"{name} = {value}{m.group(1) or ''}", src, count=1, flags=re.M)
        assert n == 1, name
    tmp = tempfile.mkdtemp(prefix="kaggle-push-")
    with open(os.path.join(tmp, "heretic_qwen35.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    meta = {
        "id": f"{user}/{slug(a.size)}",
        "title": slug(a.size),
        "code_file": "heretic_qwen35.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sources, "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(tmp, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"kernel {meta['id']}: Qwen/Qwen3.5-{a.size}, upload {a.upload_repo or '-'} "
          f"({'public' if a.public_upload else 'private'})")
    rc = kaggle("kernels", "push", "-p", tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return rc


def make_checkpoint_dataset(user, size, jsonl):
    """Upload a finished Heretic study (.jsonl) as a private Kaggle dataset; return its id."""
    ds = f"{user}/{slug(size)}-checkpoint"
    tmp = tempfile.mkdtemp(prefix="kaggle-ds-")
    shutil.copy(jsonl, tmp)
    with open(os.path.join(tmp, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": f"{slug(size)}-checkpoint", "id": ds, "licenses": [{"name": "other"}]}, f)
    exists = kaggle("datasets", "status", ds) == 0
    rc = (kaggle("datasets", "version", "-p", tmp, "-m", "checkpoint") if exists
          else kaggle("datasets", "create", "-p", tmp))
    shutil.rmtree(tmp, ignore_errors=True)
    if rc:
        sys.exit(f"could not upload the checkpoint dataset {ds}")
    return ds


def status(a):
    return kaggle("kernels", "status", f"{kaggle_user()}/{slug(a.size)}")


def output(a):
    os.makedirs(a.dest, exist_ok=True)
    return kaggle("kernels", "output", f"{kaggle_user()}/{slug(a.size)}", "-p", a.dest)


def upload(a):
    """Download the kernel output and upload the GGUF + pick.json + a card to a private HF repo.

    The HF token is read from HF_TOKEN_FILE and handed to huggingface_hub only; never printed.
    """
    if not a.gguf:
        rc = output(a)
        if rc:
            return rc
    name = f"Qwen3.5-{a.size}-heretic-ara"
    gguf = a.gguf or os.path.join(a.dest, f"{name}.Q4_K_M.gguf")
    if not os.path.exists(gguf):
        sys.exit(f"no {os.path.basename(gguf)} in {a.dest}: see run.log there")
    with open(gguf, "rb") as f:
        if f.read(4) != b"GGUF":  # Kaggle sometimes serves a 502 HTML page in place of the file
            sys.exit(f"{gguf} is not a GGUF ({os.path.getsize(gguf)} bytes): download again")
    summary = json.load(open(os.path.join(a.dest, "summary.json"), encoding="utf-8"))
    with open(HF_TOKEN_FILE, encoding="utf-8-sig") as f:
        token = f.read().strip()
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo = a.repo or f"{USER}/{name}-GGUF"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    card = (f"---\nbase_model: Qwen/Qwen3.5-{a.size}\ntags: [heretic, abliterated, gguf]\n---\n\n"
            f"# {name} (GGUF Q4_K_M)\n\nFresh Heretic run on Kaggle (p-e-w/heretic@{summary.get('heretic_ref')}, "
            f"default ARA modifier, seed {summary.get('seed')}), trial chosen by: {summary.get('rule')}. "
            f"Converted and quantized with llama.cpp {summary.get('llama_cpp_tag')}. Sanity PPL on a fixed "
            f"4k-token text: {summary.get('ppl')}.\n\nChosen trial:\n\n```json\n"
            f"{json.dumps(summary.get('chosen'), indent=2, default=str)}\n```\n")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo)
    pick = os.path.join(a.dest, "pick.json")
    if os.path.exists(pick) and open(pick, "rb").read(1) == b"{":
        api.upload_file(path_or_fileobj=pick, path_in_repo="heretic_pick.json", repo_id=repo)
    api.upload_file(path_or_fileobj=gguf, path_in_repo=f"{name}.Q4_K_M.gguf", repo_id=repo)
    print(f"uploaded {os.path.basename(gguf)} to https://huggingface.co/{repo} (private)")
    return 0


HF_TOKEN_FILE = os.path.join(SECRETS, "hf.txt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("push", "status", "output", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--size", default="4B", choices=["0.8B", "2B", "4B", "9B"])
        if name == "push":
            p.add_argument("--upload-repo", default="")
            p.add_argument("--public-upload", action="store_true")
            p.add_argument("--resume-from", default="",
                           help="a finished study .jsonl (from `output`): skip the trials, only pick + export + GGUF")
        if name in ("output", "upload"):
            p.add_argument("--dest", required=True)
        if name == "upload":
            p.add_argument("--repo", default="", help="HF repo id (default ocitodor/Qwen3.5-<size>-heretic-ara-GGUF)")
            p.add_argument("--gguf", default="", help="a local GGUF (e.g. downloaded in the browser when the API 502s); "
                                                      "--dest still holds summary.json")
    a = ap.parse_args()
    sys.exit({"push": push, "status": status, "output": output, "upload": upload}[a.cmd](a))


if __name__ == "__main__":
    main()
