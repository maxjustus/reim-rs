#!/usr/bin/env python3
"""Grid-search the target-lock parameters on the dev split.

Applies lock.py to the saved dev predictions of each base method. For each
parameter pair it computes the mean target-pitch F1 over all dev clips; the
best pair is the one run.py uses. It also reports a leave-one-speaker-out
estimate: each dev speaker is scored with the pair chosen on the other seven,
which estimates the gain on unseen speakers. Test predictions are never read.

Run:
    uv run --no-project --with-requirements eval/mixbench/requirements.txt \
        python eval/mixbench/tune_lock.py reim-live-gate swiftf0 ...
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
NO_LOCK = (1000, 100000)


def main():
    clips = load_dev()
    speakers = np.array([speaker for speaker, *_ in clips])
    params = list(itertools.product(LEVEL_DB, RANGE_CENTS))
    for base in sys.argv[1:]:
        tracks = [(refs, lock.frame_levels(x, fs, t), t, f) for _, refs, x, fs, t, f in with_preds(clips, base)]
        for mode in ["causal", "offline"]:
            f1 = np.array([[clip_f1(mode, track, *pair) for track in tracks] for pair in params])
            best = int(np.argmax(f1.mean(axis=1)))
            held_out = np.zeros(len(tracks))
            for s in np.unique(speakers):
                pick = int(np.argmax(f1[:, speakers != s].mean(axis=1)))
                held_out[speakers == s] = f1[pick, speakers == s]
            none = f1[params.index(NO_LOCK)].mean()
            print(
                f"{base:20s} {mode:8s} chosen level_db={params[best][0]} range_cents={params[best][1]}: "
                f"F1 {f1[best].mean():.4f} on all dev, {held_out.mean():.4f} leave-one-speaker-out, "
                f"{none:.4f} without lock"
            )
    return 0


def clip_f1(mode, track, level_db, range_cents):
    refs, levels, t, f = track
    if mode == "causal":
        g = lock.lock_causal(t, f, levels, level_db, range_cents)
    else:
        g = lock.lock_offline(f, levels, level_db, range_cents)
    return score.clip_metrics(refs, t, g)["f1"]


def load_dev():
    clips = []
    for line in (HERE / "manifest" / "dev.jsonl").read_text().splitlines():
        name = json.loads(line)["id"].split("/", 1)[1].replace("/", "__")
        refs = json.loads((DATA / "mix" / "dev" / f"{name}.refs.json").read_text())
        x, fs = sf.read(DATA / "mix" / "dev" / f"{name}.wav", dtype="float64")
        clips.append((name.split("_")[0], name, refs, x, fs))
    return clips


def with_preds(clips, base):
    for speaker, name, refs, x, fs in clips:
        t, f = score.read_prediction(DATA / "preds" / "dev" / base / f"{name}.csv")
        yield speaker, refs, x, fs, t, f


if __name__ == "__main__":
    sys.exit(main())
