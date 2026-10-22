#!/usr/bin/env python3
"""Measure the time of PTDB-TUG reference frames against the laryngograph.

PTDB-TUG documents RAPT with a 32 ms window and a 10 ms hop, but not the
time stamp of frame 0. This script tracks pitch on the raw laryngograph
signal of every 6th dev utterance with Praat (0.5 ms step), and finds, per
utterance, the extra shift of the reference grid (frame i at 0.016 + 0.010 i
s) that minimizes the median cents error over frames within 200 cents. The
median of those shifts plus 16 ms is build.py REF_T0.

Run:
    uv run --no-project --with-requirements eval/mixbench/requirements-methods.txt \
        python eval/mixbench/ref_offset.py
"""

import io
import os
from pathlib import Path

import numpy as np
import parselmouth
import pyarrow.parquet as pq
import soundfile as sf

DATA = Path(os.environ.get("REIM_BENCH_DATA", Path.home() / "datasets" / "reim-mixbench"))
DEV_SPEAKERS = {"F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"}
SHIFTS = np.arange(-10, 25.5, 0.5) / 1000
STEP = 0.0005


def main():
    best = []
    for shard in sorted((DATA / "raw" / "ptdb").glob("*.parquet")):
        rows = pq.read_table(shard, columns=["speaker_id", "lar_audio", "f0_data"]).to_pylist()
        for row in [r for r in rows if r["speaker_id"] in DEV_SPEAKERS][::6]:
            shift = best_shift(row)
            if shift is not None:
                best.append(shift)
    best = np.array(best) * 1000
    print(
        f"{len(best)} dev utterances; best extra shift median {np.median(best):.1f} ms, "
        f"IQR [{np.percentile(best, 25):.1f}, {np.percentile(best, 75):.1f}] ms"
    )


def best_shift(row):
    x, fs = sf.read(io.BytesIO(row["lar_audio"]["bytes"]))
    pitch = parselmouth.Sound(x, fs).to_pitch_ac(time_step=STEP, pitch_floor=60, pitch_ceiling=600)
    t, f = pitch.xs(), pitch.selected_array["frequency"]
    ref = np.array([frame[0] for frame in row["f0_data"]])
    errors = []
    for shift in SHIFTS:
        i = np.clip(np.round((0.016 + 0.010 * np.arange(len(ref)) + shift - t[0]) / STEP).astype(int), 0, len(t) - 1)
        both = (ref > 0) & (f[i] > 0)
        cents = np.abs(1200 * np.log2(f[i][both] / ref[both]))
        cents = cents[cents < 200]
        errors.append(np.median(cents) if len(cents) > 50 else np.nan)
    errors = np.array(errors)
    return SHIFTS[np.nanargmin(errors)] if np.isfinite(errors).sum() > 10 else None


if __name__ == "__main__":
    main()
