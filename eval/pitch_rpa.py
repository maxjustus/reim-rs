#!/usr/bin/env python3
"""Sung-pitch accuracy of reim's F0 tracker on the Vocadito dataset.

Runs `reim f0` on each clip, aligns its 5 ms grid against Vocadito's ~5.8 ms
ground-truth grid via mir_eval.melody (which does the cents conversion, time
resampling, and voicing handling for us), and reports the standard melody
metrics plus a voicing precision/recall/F1 comparable to the Vocadito paper's
F1 ~= 0.86.

The RPA-vs-RCA gap is the octave-error signature: RCA is octave-invariant, so
RPA << RCA means the tracker is landing on the right pitch class but the wrong
octave.

Run:
    uv run --no-project --with numpy --with soundfile --with mir_eval \
        python eval/pitch_rpa.py

    uv run --no-project --with numpy --with soundfile --with mir_eval \
        python eval/pitch_rpa.py --self-test

The self-test runs automatically (and is the only thing that runs) when the
dataset is not present, so this stays useful in an ephemeral /tmp environment.
"""

import argparse
import csv
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

import mir_eval.melody as melody

# reim's hop is 5 ms but alternates 220/221 samples at 44.1 kHz (4.989 vs
# 5.011 ms), which trips mir_eval's strict uniform-timescale check. Its warning
# only matters when silences are encoded as MISSING samples; reim encodes them
# as 0 Hz (which mir_eval handles correctly), so the warning is a false alarm.
warnings.filterwarnings("ignore", message="Non-uniform timescale", category=UserWarning)

# Vocadito is 44.1 kHz solo singing; these match reim's own fmin/fmax defaults
# so the tracker is measured under the same band it ships with.
FMIN_HZ = 71.0
FMAX_HZ = 800.0

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REIM = REPO_ROOT / "target" / "release" / "reim"
DEFAULT_DATA_DIR = Path("/private/tmp/datasets/vocadito_raw")


def run_reim_f0(reim_bin, wav_path):
    """Invoke `reim f0` and parse its headerless "time,fo" CSV into two arrays."""
    cmd = [str(reim_bin), "f0", str(wav_path), str(FMIN_HZ), str(FMAX_HZ)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    times, freqs = parse_two_col_csv_text(proc.stdout)
    return cmd, dedup_times(times, freqs)


def dedup_times(times, freqs):
    """reim stamps its first few analysis frames at t=0.0 (window center clamped
    to the signal start), producing duplicate timestamps. mir_eval's resampler
    (scipy interp1d) rejects non-strictly-increasing time bases, so collapse any
    run of equal timestamps to its last sample. Those leading frames are
    unvoiced (0 Hz), so this drops no real pitch data."""
    if len(times) == 0:
        return times, freqs
    # keep an index only if the NEXT timestamp differs (i.e. the last of a run)
    keep = np.ones(len(times), dtype=bool)
    keep[:-1] = times[1:] != times[:-1]
    return times[keep], freqs[keep]


def parse_two_col_csv_text(text):
    """Parse "time,value" rows into (time[], value[]). Skips a header row and
    any blank/garbage lines so we don't choke on stray output."""
    times, vals = [], []
    for row in csv.reader(text.splitlines()):
        if len(row) < 2:
            continue
        try:
            t = float(row[0])
            v = float(row[1])
        except ValueError:
            # header line like "time_seconds,fo_hz"
            continue
        times.append(t)
        vals.append(v)
    return np.asarray(times, dtype=float), np.asarray(vals, dtype=float)


def load_ref_csv(path):
    """Load a Vocadito F0 annotation: headerless "time_seconds,f0_hz", f0=0 == unvoiced."""
    return parse_two_col_csv_text(Path(path).read_text())


def voicing_prf(ref_time, ref_freq, est_time, est_freq):
    """Voicing precision/recall/F1 on the common (reference) time grid.

    mir_eval.melody.to_cent_voicing resamples the estimate onto the reference
    grid and returns aligned voicing arrays. We binarize the estimate's voicing
    at >0 and compute a standard P/R/F1 so it lines up with the binary voiced/
    unvoiced framing used by the Vocadito F1 number.
    """
    _, ref_voicing, _, est_voicing = melody.to_cent_voicing(
        ref_time, ref_freq, est_time, est_freq
    )
    ref_v = ref_voicing > 0
    est_v = est_voicing > 0
    tp = int(np.sum(ref_v & est_v))
    fp = int(np.sum(~ref_v & est_v))
    fn = int(np.sum(ref_v & ~est_v))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"voicing_precision": precision, "voicing_recall": recall, "voicing_f1": f1}


def eval_clip(reim_bin, wav_path, ref_csv_path):
    """Full metric bundle for one clip."""
    cmd, (est_time, est_freq) = run_reim_f0(reim_bin, wav_path)
    ref_time, ref_freq = load_ref_csv(ref_csv_path)
    scores = melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    metrics = {
        "rpa": scores["Raw Pitch Accuracy"],
        "rca": scores["Raw Chroma Accuracy"],
        "overall": scores["Overall Accuracy"],
        "voicing_recall_me": scores["Voicing Recall"],
        "voicing_fa": scores["Voicing False Alarm"],
    }
    metrics.update(voicing_prf(ref_time, ref_freq, est_time, est_freq))
    return cmd, metrics


def discover_clips(data_dir):
    """Find (clip_id, wav, ref_csv) triples for every clip with both files present."""
    audio_dir = data_dir / "Audio"
    f0_dir = data_dir / "Annotations" / "F0"
    clips = []
    for wav in sorted(audio_dir.glob("vocadito_*.wav"), key=_clip_sort_key):
        clip_id = wav.stem  # "vocadito_N"
        ref = f0_dir / f"{clip_id}_f0.csv"
        if ref.exists():
            clips.append((clip_id, wav, ref))
    return clips


def _clip_sort_key(p):
    """Sort vocadito_2 before vocadito_10 by numeric suffix."""
    try:
        return int(p.stem.split("_")[1])
    except (IndexError, ValueError):
        return 1 << 30


# ---- metric column layout for the summary table ----
COLUMNS = [
    ("rpa", "RPA"),
    ("rca", "RCA"),
    ("overall", "OvAcc"),
    ("voicing_recall_me", "V.Rec"),
    ("voicing_fa", "V.FA"),
    ("voicing_precision", "vP"),
    ("voicing_recall", "vR"),
    ("voicing_f1", "vF1"),
]


def print_table(rows, label_width=16):
    """rows: list of (label, metrics_dict). Prints a fixed-width metric table."""
    header = "clip".ljust(label_width) + "".join(name.rjust(8) for _, name in COLUMNS)
    print(header)
    print("-" * len(header))
    for label, m in rows:
        line = label.ljust(label_width) + "".join(
            f"{m[key]:8.3f}" for key, _ in COLUMNS
        )
        print(line)


def aggregate(per_clip):
    """Mean of each metric across clips."""
    return {
        key: float(np.mean([m[key] for _, m in per_clip])) for key, _ in COLUMNS
    }


def run_dataset(reim_bin, data_dir, limit=None):
    clips = discover_clips(data_dir)
    if limit is not None:
        clips = clips[:limit]
    if not clips:
        print(f"No Vocadito clips found under {data_dir}", file=sys.stderr)
        return 1

    per_clip = []
    invocation_printed = False
    for clip_id, wav, ref in clips:
        cmd, metrics = eval_clip(reim_bin, wav, ref)
        if not invocation_printed:
            print("reim invocation: " + " ".join(cmd))
            print()
            invocation_printed = True
        per_clip.append((clip_id, metrics))

    print_table(per_clip)
    print("-" * (16 + 8 * len(COLUMNS)))
    print_table([("MEAN", aggregate(per_clip))])
    print()
    print(
        "RPA-RCA gap (octave-error signature): "
        f"{aggregate(per_clip)['rca'] - aggregate(per_clip)['rpa']:.3f}"
    )
    return 0


# --------------------------------------------------------------------------
# Self-test: synthetic signals with known ground truth, asserting the metric
# behaves as expected. This is the correctness proof, independent of reim.
# --------------------------------------------------------------------------

def _cents_to_ratio(cents):
    return 2.0 ** (cents / 1200.0)


def self_test():
    print("=== self-test (synthetic, known ground truth) ===")

    # Reference track: a slow voiced sweep with an unvoiced gap. The cents/
    # octave assertions below put the estimate on the SAME time grid as the
    # reference, so mir_eval short-circuits resampling (see resample_melody_
    # series: "If times and times_new are equivalent, no resampling will be
    # performed") and the metric reflects the frequency shift alone, with no
    # interpolation-boundary noise. Grid resampling (reim's 5 ms vs Vocadito's
    # ~5.8 ms) is exercised separately in test (5).
    ref_time = np.arange(0, 4.0, 0.005805)
    ref_freq = 200.0 + 80.0 * np.sin(2 * np.pi * 0.25 * ref_time)
    voiced = (ref_time <= 1.0) | (ref_time >= 1.5)
    ref_freq = np.where(voiced, ref_freq, 0.0)  # unvoiced gap from 1.0-1.5 s

    def shifted(ratio):
        """Multiply voiced frames by a frequency ratio, keep 0-Hz gaps at 0."""
        return np.where(voiced, ref_freq * ratio, 0.0)

    # (1) est == ref  -> RPA == 1.0, RCA == 1.0
    s = melody.evaluate(ref_time, ref_freq, ref_time.copy(), ref_freq.copy())
    print(f"(1) identical:        RPA={s['Raw Pitch Accuracy']:.4f} RCA={s['Raw Chroma Accuracy']:.4f}")
    assert s["Raw Pitch Accuracy"] == 1.0, s["Raw Pitch Accuracy"]
    assert s["Raw Chroma Accuracy"] == 1.0, s["Raw Chroma Accuracy"]

    # (2) est shifted by +50.1 cents -> every voiced frame is just past the
    # 50-cent tolerance, so RPA collapses to 0 (all frames flip out of bounds).
    s2 = melody.evaluate(ref_time, ref_freq, ref_time.copy(), shifted(_cents_to_ratio(50.1)))
    print(f"(2) +50.1 cents:      RPA={s2['Raw Pitch Accuracy']:.4f} RCA={s2['Raw Chroma Accuracy']:.4f}")
    assert s2["Raw Pitch Accuracy"] == 0.0, s2["Raw Pitch Accuracy"]
    # Contrast: +49.9 cents stays inside tolerance -> RPA flips back to 1.0,
    # demonstrating the frames cross the 50-cent boundary, not a global decay.
    s2b = melody.evaluate(ref_time, ref_freq, ref_time.copy(), shifted(_cents_to_ratio(49.9)))
    print(f"(2b) +49.9 cents:     RPA={s2b['Raw Pitch Accuracy']:.4f} (tolerance edge)")
    assert s2b["Raw Pitch Accuracy"] == 1.0, s2b["Raw Pitch Accuracy"]

    # (3) est shifted up exactly one octave (2x) -> RPA collapses to 0 but RCA
    # stays 1.0, because chroma accuracy is octave-invariant. This is the
    # property the RPA-vs-RCA gap relies on to surface octave errors.
    s3 = melody.evaluate(ref_time, ref_freq, ref_time.copy(), shifted(2.0))
    print(f"(3) +1 octave (2x):   RPA={s3['Raw Pitch Accuracy']:.4f} RCA={s3['Raw Chroma Accuracy']:.4f}")
    assert s3["Raw Pitch Accuracy"] == 0.0, s3["Raw Pitch Accuracy"]
    assert s3["Raw Chroma Accuracy"] == 1.0, s3["Raw Chroma Accuracy"]

    # (4) voicing P/R/F1 derivation: perfect agreement on identical tracks.
    prf = voicing_prf(ref_time, ref_freq, ref_time.copy(), ref_freq.copy())
    print(
        f"(4) voicing P/R/F1 identical: "
        f"P={prf['voicing_precision']:.4f} R={prf['voicing_recall']:.4f} F1={prf['voicing_f1']:.4f}"
    )
    assert prf["voicing_f1"] == 1.0, prf

    # (5) cross-grid resampling: estimate on reim's 5 ms grid interpolated from
    # the same underlying contour. mir_eval resamples it back onto the ref grid;
    # we expect near-perfect RPA, with at most a frame or two of slack at the
    # voiced/unvoiced gap boundaries (a documented resampling edge effect).
    est_time = np.arange(0, 4.02, 0.005)
    est_freq = np.interp(est_time, ref_time, ref_freq)
    nearest = np.searchsorted(ref_time, est_time).clip(0, len(ref_time) - 1)
    est_freq = np.where(ref_freq[nearest] == 0.0, 0.0, est_freq)
    s5 = melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    print(f"(5) 5ms vs 5.8ms grid: RPA={s5['Raw Pitch Accuracy']:.4f} (resampling sanity)")
    assert s5["Raw Pitch Accuracy"] >= 0.99, s5["Raw Pitch Accuracy"]

    print("self-test PASSED")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sung-pitch accuracy (RPA/RCA/voicing) of reim's F0 tracker on Vocadito.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Vocadito root containing Audio/ and Annotations/F0/.",
    )
    parser.add_argument(
        "--reim",
        type=Path,
        default=DEFAULT_REIM,
        help="Path to the prebuilt reim release binary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N clips (smoke runs).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the synthetic correctness self-test and exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    # If the dataset is absent (e.g. ephemeral /tmp), still prove correctness
    # via the self-test and exit 0 rather than failing.
    audio_dir = args.data_dir / "Audio"
    if not audio_dir.is_dir():
        print(f"Dataset not found at {args.data_dir}; running self-test only.\n")
        return self_test()

    if not args.reim.exists():
        print(f"reim binary not found at {args.reim}", file=sys.stderr)
        return 2

    return run_dataset(args.reim, args.data_dir, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
