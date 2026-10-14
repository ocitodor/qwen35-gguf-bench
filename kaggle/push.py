#!/usr/bin/env python3
"""Push kaggle/heretic_qwen35.py as a private Kaggle GPU script kernel, or check on it.

Usage (on the PC, needs the Kaggle API token and `pip install kaggle`):
  python kaggle/push.py push --size 4B [--upload-repo USER/Qwen3.5-4B-heretic-ara-GGUF] [--public-upload]
  python kaggle/push.py status --size 4B
  python kaggle/push.py output --size 4B --dest DIR      # logs + pick.json + summary.json (+ the GGUF)

Credentials: KAGGLE_CONFIG_DIR points at the folder holding kaggle.json. Default
C:\\Scripts\\secrets\\kaggle (outside every repo); the token is never printed.
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


def kaggle_user():
    cfg = os.environ.setdefault("KAGGLE_CONFIG_DIR", SECRETS)
    path = os.path.join(cfg, "kaggle.json")
    if not os.path.exists(path):
        sys.exit(f"no Kaggle API token at {path} (Kaggle -> Settings -> API -> Create New Token)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["username"]


def slug(size):
    return f"qwen35-{size.lower()}-heretic-ara"


def kaggle(*args):
    cmd = [sys.executable, "-m", "kaggle.cli", *args]
    print("$ kaggle " + " ".join(args), flush=True)
    return subprocess.call(cmd)


def push(a):
    user = kaggle_user()
    src = open(os.path.join(HERE, "heretic_qwen35.py"), encoding="utf-8").read()
    subs = {
        "MODEL": f'"Qwen/Qwen3.5-{a.size}"',
        "OUT_NAME": f'"Qwen3.5-{a.size}-heretic-ara"',
        "UPLOAD_REPO": f'"{a.upload_repo or ""}"',
        "UPLOAD_PRIVATE": "False" if a.public_upload else "True",
    }
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
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(tmp, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"kernel {meta['id']}: Qwen/Qwen3.5-{a.size}, upload {a.upload_repo or '-'} "
          f"({'public' if a.public_upload else 'private'})")
    rc = kaggle("kernels", "push", "-p", tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return rc


def status(a):
    return kaggle("kernels", "status", f"{kaggle_user()}/{slug(a.size)}")


def output(a):
    os.makedirs(a.dest, exist_ok=True)
    return kaggle("kernels", "output", f"{kaggle_user()}/{slug(a.size)}", "-p", a.dest)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("push", "status", "output"):
        p = sub.add_parser(name)
        p.add_argument("--size", default="4B", choices=["0.8B", "2B", "4B", "9B"])
        if name == "push":
            p.add_argument("--upload-repo", default="")
            p.add_argument("--public-upload", action="store_true")
        if name == "output":
            p.add_argument("--dest", required=True)
    a = ap.parse_args()
    sys.exit({"push": push, "status": status, "output": output}[a.cmd](a))


if __name__ == "__main__":
    main()
