#!/usr/bin/env python3
"""Download the mixbench source archives and verify their checksums.

Every file in sources.json is fetched into $REIM_BENCH_DATA/raw (default
~/datasets/reim-mixbench/raw), resumed when partial, and verified against the
published size and SHA-256 (PTDB-TUG mirror) or MD5 (DEMAND on Zenodo). A
file that fails verification is reported and left in place for inspection.

Run:
    python3 eval/mixbench/fetch.py [--only ptdb|demand] [--jobs N]
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REIM_BENCH_DATA", Path.home() / "datasets" / "reim-mixbench"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["ptdb", "demand"])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    files = json.loads((HERE / "sources.json").read_text())["files"]
    files = [f for f in files if not args.only or f["name"].startswith(args.only + "/")]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(fetch, files))
    return 0 if all(results) else 1


def fetch(f):
    path = DATA / "raw" / f["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.stat().st_size == f["size"]):
        subprocess.run(
            ["curl", "-sSL", "--retry", "5", "-C", "-", "-o", str(path), f["url"]],
            check=True,
        )
    ok = verify(path, f)
    print(f"{'ok  ' if ok else 'FAIL'} {f['name']}", flush=True)
    return ok


def verify(path, f):
    if path.stat().st_size != f["size"]:
        return False
    algo = "sha256" if "sha256" in f else "md5"
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == f[algo]


if __name__ == "__main__":
    sys.exit(main())
