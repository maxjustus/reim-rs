#!/usr/bin/env python3
"""Pitch/voicing accuracy of reim's F0 tracker on Vocadito and PTDB-TUG.

Runs `reim f0` on each clip, aligns its 5 ms grid against the dataset's
ground-truth grid via mir_eval.melody (which does the cents conversion, time
resampling, and voicing handling for us), and reports the standard melody
metrics plus a voicing precision/recall/F1 comparable to the Vocadito paper's
F1 ~= 0.86.

Datasets (--dataset vocadito|ptdb|all):
  vocadito  /private/tmp/datasets/vocadito_raw    44.1 kHz solo singing
  ptdb      /private/tmp/datasets/ptdb_tug        48 kHz English speech,
            laryngograph-derived RAPT reference (10 ms hop, .f0 4-column)

Noise conditions (--noise none|white20|white10|hum50|all) are mixed on the fly
into temp wavs (seeded per clip, deterministic): white noise at 20/10 dB SNR
and 50 Hz mains hum + harmonics at -20 dB relative to signal RMS.

--sweep v1,v2,... re-runs the pooled eval per REIM_VOICING_SCORE_MIN value.

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
import zlib
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
PTDB_DATA_DIR = Path("/private/tmp/datasets/ptdb_tug")

NOISE_CONDITIONS = ("none", "white20", "white10", "hum50")


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


def eval_clip(reim_bin, wav_path, ref_time, ref_freq):
    """Full metric bundle for one clip."""
    cmd, (est_time, est_freq) = run_reim_f0(reim_bin, wav_path)
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


def load_vocadito(data_dir):
    """Yield (clip_id, wav_path, ref_time, ref_freq) for every Vocadito clip."""
    for clip_id, wav, ref in discover_clips(data_dir):
        ref_time, ref_freq = load_ref_csv(ref)
        yield clip_id, wav, ref_time, ref_freq


def load_ptdb(data_dir):
    """Yield (clip_id, wav_path, ref_time, ref_freq) for every PTDB-TUG clip.

    Layout: SPEECH DATA/{FEMALE,MALE}/MIC/<spk>/mic_<spk>_<utt>.wav with the
    RAPT reference at .../REF/<spk>/ref_<spk>_<utt>.f0. The .f0 file is ASCII,
    4 whitespace columns [f0_hz, voicing_flag, rms, ac_peak], no time column,
    10 ms hop; f0=0 marks unvoiced frames. Frame i is stamped at the center of
    its 32 ms RAPT window (i*10ms + 16ms); without that offset RPA drops ~40
    points from pure misalignment.
    """
    root = data_dir / "SPEECH DATA"
    for wav in sorted(root.glob("*/MIC/*/mic_*.wav")):
        ref = Path(str(wav.parent).replace("/MIC/", "/REF/")) / wav.name.replace(
            "mic_", "ref_"
        ).replace(".wav", ".f0")
        if not ref.exists():
            continue
        f0 = np.array([float(line.split()[0]) for line in ref.read_text().splitlines() if line.split()])
        yield wav.stem, wav, np.arange(len(f0)) * 0.010 + 0.016, f0


DATASETS = {
    "vocadito": (load_vocadito, DEFAULT_DATA_DIR),
    "ptdb": (load_ptdb, PTDB_DATA_DIR),
}


def add_noise(wav_path, condition, seed):
    """Return a path to a noisy copy of wav_path (a temp file the caller must
    unlink), or wav_path itself for condition "none"."""
    if condition == "none":
        return Path(wav_path)
    import os
    import tempfile

    import soundfile as sf

    x, fs = sf.read(wav_path, dtype="float64")
    rms = float(np.sqrt(np.mean(x**2))) or 1.0
    rng = np.random.default_rng(seed)
    if condition in ("white20", "white10"):
        snr_db = 20.0 if condition == "white20" else 10.0
        noise = rng.standard_normal(len(x)) * rms / (10 ** (snr_db / 20.0))
    elif condition == "hum50":
        t = np.arange(len(x)) / fs
        hum = sum(
            np.sin(2 * np.pi * 50.0 * h * t + rng.uniform(0, 2 * np.pi)) / h
            for h in (1, 2, 3)
        )
        noise = hum * rms * 10 ** (-20.0 / 20.0)
    else:
        raise ValueError(condition)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, x + noise, fs, subtype="FLOAT")
    return Path(tmp)


def _clip_sort_key(p):
    """Sort vocadito_2 before vocadito_10 by numeric suffix."""
    try:
        return int(p.stem.split("_")[1])
    except (IndexError, ValueError):
        return 1 << 30


# ---- metric column layout for the summary table ----
LABEL_WIDTH = 30
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


def run_datasets(reim_bin, datasets, conditions, limit=None, quiet=False):
    """Evaluate every (dataset, noise condition) group; returns the pooled
    per-clip rows, or None when no clips were found. Prints per-group tables
    and a pooled MEAN unless quiet."""
    pooled = []
    invocation_printed = False
    for ds_name in datasets:
        loader, data_dir = DATASETS[ds_name]
        clips = list(loader(data_dir))
        if limit is not None:
            clips = clips[:limit]
        if not clips:
            print(f"No {ds_name} clips found under {data_dir}", file=sys.stderr)
            continue
        for cond in conditions:
            group = []
            for clip_id, wav, ref_time, ref_freq in clips:
                noisy = add_noise(wav, cond, seed=zlib.crc32(clip_id.encode()))
                try:
                    cmd, metrics = eval_clip(reim_bin, noisy, ref_time, ref_freq)
                finally:
                    if noisy != Path(wav):
                        noisy.unlink(missing_ok=True)
                if not invocation_printed and not quiet:
                    print("reim invocation: " + " ".join(cmd))
                    print()
                    invocation_printed = True
                label = clip_id if cond == "none" and len(datasets) == 1 else f"{ds_name}/{cond}:{clip_id}"
                group.append((label, metrics))
            pooled.extend(group)
            if not quiet:
                print_table(group, label_width=LABEL_WIDTH)
                print("-" * (LABEL_WIDTH + 8 * len(COLUMNS)))
                print_table(
                    [(f"MEAN {ds_name}/{cond}", aggregate(group))],
                    label_width=LABEL_WIDTH,
                )
                print()
    if not pooled:
        return None
    if not quiet:
        if len(datasets) * len(conditions) > 1:
            print_table([("MEAN pooled", aggregate(pooled))], label_width=LABEL_WIDTH)
            print()
        print(
            "RPA-RCA gap (octave-error signature): "
            f"{aggregate(pooled)['rca'] - aggregate(pooled)['rpa']:.3f}"
        )
    return pooled


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
        "--dataset",
        choices=[*DATASETS, "all"],
        default="vocadito",
        help="Which dataset(s) to evaluate.",
    )
    parser.add_argument(
        "--noise",
        choices=[*NOISE_CONDITIONS, "all"],
        default="none",
        help="Noise condition mixed into each clip on the fly.",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Comma-separated REIM_VOICING_SCORE_MIN values; prints a pooled "
        "metric row per value instead of the full tables.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the synthetic correctness self-test and exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    conditions = list(NOISE_CONDITIONS) if args.noise == "all" else [args.noise]
    if args.data_dir != DEFAULT_DATA_DIR and len(datasets) == 1:
        DATASETS[datasets[0]] = (DATASETS[datasets[0]][0], args.data_dir)

    # If every requested dataset is absent (e.g. ephemeral /tmp), still prove
    # correctness via the self-test and exit 0 rather than failing.
    if not any(DATASETS[d][1].is_dir() for d in datasets):
        print("Dataset(s) not found; running self-test only.\n")
        return self_test()

    if not args.reim.exists():
        print(f"reim binary not found at {args.reim}", file=sys.stderr)
        return 2

    if args.sweep is not None:
        import os

        print("score_min".ljust(LABEL_WIDTH) + "".join(n.rjust(8) for _, n in COLUMNS))
        for v in args.sweep.split(","):
            os.environ["REIM_VOICING_SCORE_MIN"] = v
            pooled = run_datasets(
                args.reim, datasets, conditions, limit=args.limit, quiet=True
            )
            if pooled is None:
                return 1
            m = aggregate(pooled)
            print(v.ljust(LABEL_WIDTH) + "".join(f"{m[k]:8.3f}" for k, _ in COLUMNS))
        return 0

    pooled = run_datasets(args.reim, datasets, conditions, limit=args.limit)
    return 0 if pooled is not None else 1


if __name__ == "__main__":
    sys.exit(main())
