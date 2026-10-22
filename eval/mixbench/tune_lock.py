#!/usr/bin/env python3
"""Grid-search the target-lock parameters on the dev split.

Applies lock.py to the saved dev predictions of each base method and prints
the mean target-pitch F1 over all dev clips for every parameter pair.
Test predictions are never read.

Run:
    uv run --no-project --with-requirements eval/mixbench/requirements.txt \
        python eval/mixbench/tune_lock.py reim-live swiftf0 ...
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import lock
import score

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REIM_BENCH_DATA", Path.home() / "datasets" / "reim-mixbench"))
LEVEL_DB = [3, 4.5, 6, 9, 12, 15, 1000]
RANGE_CENTS = [400, 600, 800, 1000, 100000]


def main():
    clips = load_dev()
    for base in sys.argv[1:]:
        tracks = [(refs, lock.frame_levels(x, fs, t), t, f) for refs, x, fs, t, f in with_preds(clips, base)]
        for mode in ["causal", "offline"]:
            results = []
            for level_db, range_cents in itertools.product(LEVEL_DB, RANGE_CENTS):
                f1 = []
                for refs, levels, t, f in tracks:
                    if mode == "causal":
                        g = lock.lock_causal(t, f, levels, level_db, range_cents)
                    else:
                        g = lock.lock_offline(f, levels, level_db, range_cents)
                    f1.append(score.clip_metrics(refs, t, g)["f1"])
                results.append((float(np.mean(f1)), level_db, range_cents))
            best = max(results)
            none = next(r for r in results if r[1] == 1000 and r[2] == 100000)
            print(f"{base:16s} {mode:8s} best F1 {best[0]:.4f} at level_db={best[1]} range_cents={best[2]} (no lock {none[0]:.4f})")
            for r in sorted(results, reverse=True)[:5]:
                print(f"    {r[0]:.4f} level_db={r[1]} range_cents={r[2]}")
    return 0


def load_dev():
    clips = []
    for line in (HERE / "manifest" / "dev.jsonl").read_text().splitlines():
        name = json.loads(line)["id"].split("/", 1)[1].replace("/", "__")
        refs = json.loads((DATA / "mix" / "dev" / f"{name}.refs.json").read_text())
        x, fs = sf.read(DATA / "mix" / "dev" / f"{name}.wav", dtype="float64")
        clips.append((name, refs, x, fs))
    return clips


def with_preds(clips, base):
    for name, refs, x, fs in clips:
        t, f = score.read_prediction(DATA / "preds" / "dev" / base / f"{name}.csv")
        yield refs, x, fs, t, f


if __name__ == "__main__":
    sys.exit(main())
