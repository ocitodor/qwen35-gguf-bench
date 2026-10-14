#!/usr/bin/env python3
"""Run one command, capture its stdout/stderr to files, record wall time and peak RSS.

Usage: measure.py --stdout FILE --stderr FILE --meta FILE -- <command> [args...]

Peak RSS comes from getrusage(RUSAGE_CHILDREN) after the child exits, so no
extra tools (GNU time, /proc pollers) are needed on the runner. Exit code is
the child's.
"""
import json
import resource
import subprocess
import sys
import time


def main(argv):
    opts = {}
    if "--" not in argv:
        sys.exit("measure.py: expected '--' before the command")
    split = argv.index("--")
    flags, cmd = argv[:split], argv[split + 1:]
    it = iter(flags)
    for flag in it:
        if flag in ("--stdout", "--stderr", "--meta"):
            opts[flag[2:]] = next(it, None)
        else:
            sys.exit(f"measure.py: unknown option {flag}")
    missing = [k for k in ("stdout", "stderr", "meta") if not opts.get(k)]
    if missing or not cmd:
        sys.exit("measure.py: --stdout, --stderr, --meta and a command are required")

    t0 = time.monotonic()
    with open(opts["stdout"], "wb") as out, open(opts["stderr"], "wb") as err:
        rc = subprocess.call(cmd, stdout=out, stderr=err)
    elapsed = time.monotonic() - t0
    rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # KiB on Linux

    meta = {
        "cmd": cmd,
        "returncode": rc,
        "elapsed_s": round(elapsed, 1),
        "max_rss_gb": round(rss_kib * 1024 / 1e9, 2),
    }
    with open(opts["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
