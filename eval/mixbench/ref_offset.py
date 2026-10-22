#!/usr/bin/env python3
"""Measure the time of PTDB-TUG reference frames as heard at the microphone.

PTDB-TUG documents RAPT with a 32 ms window and a 10 ms hop, but not the
time stamp of frame 0. For every 6th dev utterance this script tracks pitch
with Praat autocorrelation (0.5 ms step) on the raw laryngograph and on the
microphone signal, and finds per utterance:

  lar  the shift of the reference grid (frame i at 0.016 + 0.010 i s) that
       best matches the laryngograph track
  mic  the delay of the microphone track behind the laryngograph track
       (sound travel from the larynx to the microphone)

"Best" minimizes the median cents error over frames within 200 cents. build.py
REF_T0 is 0.016 s plus the median of each.

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
STEP = 0.0005


def main():
    lar, mic = [], []
    for shard in sorted((DATA / "raw" / "ptdb").glob("*.parquet")):
        rows = pq.read_table(shard, columns=["speaker_id", "lar_audio", "mic_audio", "f0_data"]).to_pylist()
        for row in [r for r in rows if r["speaker_id"] in DEV_SPEAKERS][::6]:
            lt, lf = track(row["lar_audio"])
            mt, mf = track(row["mic_audio"])
            ref = np.array([frame[0] for frame in row["f0_data"]])
            grid = 0.016 + 0.010 * np.arange(len(ref))
            lar.append(best_shift(grid, ref, lt, lf, np.arange(-10, 25.5, 0.5) / 1000))
            mic.append(best_shift(lt[::20], lf[::20], mt, mf, np.arange(-5, 5.25, 0.25) / 1000))
    for name, shifts in [("reference vs laryngograph", lar), ("microphone after laryngograph", mic)]:
        ms = np.array([s for s in shifts if s is not None]) * 1000
        print(f"{name}: {len(ms)} utterances, median {np.median(ms):+.2f} ms, IQR [{np.percentile(ms, 25):+.2f}, {np.percentile(ms, 75):+.2f}] ms")


def track(audio):
    x, fs = sf.read(io.BytesIO(audio["bytes"]))
    pitch = parselmouth.Sound(x, fs).to_pitch_ac(time_step=STEP, pitch_floor=60, pitch_ceiling=600)
    return pitch.xs(), pitch.selected_array["frequency"]


def best_shift(grid, ref, t, f, shifts):
    """Shift of `grid` that best matches the track (t, f); None when too few frames overlap."""
    errors = []
    for shift in shifts:
        i = np.clip(np.round((grid + shift - t[0]) / STEP).astype(int), 0, len(t) - 1)
        both = (ref > 0) & (f[i] > 0)
        cents = np.abs(1200 * np.log2(f[i][both] / ref[both]))
        cents = cents[cents < 200]
        errors.append(np.median(cents) if len(cents) > 50 else np.nan)
    errors = np.array(errors)
    return shifts[np.nanargmin(errors)] if np.isfinite(errors).sum() > 10 else None


if __name__ == "__main__":
    main()
