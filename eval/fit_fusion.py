#!/usr/bin/env python3
"""Fit logistic-regression fusion weights for reim's voicing gate.

Collects per-frame features from `reim features` (SRH score, NCCF, CPP)
across the datasets/noise conditions from pitch_rpa.py, labels each frame
with the reference voicing (nearest reference frame, ref_freq > 0), fits a
plain-numpy logistic regression, and prints a ready-to-paste Rust const
block with the standardization folded into the weights.

Run:
    uv run --no-project --with numpy --with soundfile \
        python eval/fit_fusion.py --dataset all --noise all
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from pitch_rpa import DATASETS, DEFAULT_REIM, FMAX_HZ, FMIN_HZ, NOISE_CONDITIONS, add_noise
import zlib


def run_reim_features(reim_bin, wav_path):
    cmd = [str(reim_bin), "features", str(wav_path), str(FMIN_HZ), str(FMAX_HZ)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = proc.stdout.splitlines()
    names = lines[0].split(",")
    rows = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    return {n: rows[:, i] for i, n in enumerate(names)}


def collect(reim_bin, datasets, conditions, limit):
    xs, ys, groups = [], [], []
    for ds_name in datasets:
        loader, data_dir = DATASETS[ds_name]
        clips = list(loader(data_dir))
        if limit is not None:
            clips = clips[:limit]
        for clip_id, wav, ref_time, ref_freq in clips:
            for cond in conditions:
                noisy = add_noise(wav, cond, seed=zlib.crc32(clip_id.encode()))
                try:
                    feat = run_reim_features(reim_bin, noisy)
                finally:
                    if noisy != Path(wav):
                        noisy.unlink(missing_ok=True)
                # Nearest reference frame per estimate frame; drop estimate
                # frames past the annotated range (half a ref hop of slack).
                hop = np.median(np.diff(ref_time)) if len(ref_time) > 1 else 0.01
                idx = np.searchsorted(ref_time, feat["time"]).clip(0, len(ref_time) - 1)
                left = (idx - 1).clip(0)
                use_left = np.abs(ref_time[left] - feat["time"]) < np.abs(
                    ref_time[idx] - feat["time"]
                )
                idx = np.where(use_left, left, idx)
                ok = np.abs(ref_time[idx] - feat["time"]) <= hop / 2
                label = (ref_freq[idx] > 0)[ok]
                x = np.column_stack(
                    [
                        np.log(np.maximum(feat["score"][ok], 1e-12)),
                        feat["nccf"][ok],
                        feat["cpp"][ok],
                    ]
                )
                xs.append(x)
                ys.append(label)
                groups.extend([f"{ds_name}/{cond}"] * int(label.sum() + (~label).sum()))
    return np.vstack(xs), np.concatenate(ys), np.array(groups)


def fit_logistic(x, y, l2=1e-3, iters=500, lr=0.5):
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-12
    xs = (x - mean) / std
    w = np.zeros(x.shape[1])
    b = 0.0
    n = len(y)
    yf = y.astype(float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xs @ w + b)))
        g = p - yf
        w -= lr * (xs.T @ g / n + l2 * w)
        b -= lr * g.mean()
    # fold standardization into raw-feature weights
    w_raw = w / std
    b_raw = b - float(w @ (mean / std))
    return w_raw, b_raw


def metrics(x, y, w, b):
    z = x @ w + b
    p = 1.0 / (1.0 + np.exp(-z))
    pred = p >= 0.5
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(len(p))
    pos, neg = ranks[y], ranks[~y]
    auc = (pos.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return prec, rec, f1, auc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reim", type=Path, default=DEFAULT_REIM)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--noise", choices=[*NOISE_CONDITIONS, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args(argv)

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    conditions = list(NOISE_CONDITIONS) if args.noise == "all" else [args.noise]
    datasets = [d for d in datasets if DATASETS[d][1].is_dir()]
    if not datasets:
        print("no datasets present", file=sys.stderr)
        return 1

    x, y, groups = collect(args.reim, datasets, conditions, args.limit)
    print(f"{len(y)} frames, {y.mean():.1%} voiced, from {sorted(set(groups))}")

    # deterministic train/val split by frame index stride
    val = np.zeros(len(y), dtype=bool)
    val[:: max(2, int(1 / args.val_frac))] = True
    w, b = fit_logistic(x[~val], y[~val])
    for name, xv, yv in [("train", x[~val], y[~val]), ("val", x[val], y[val])]:
        prec, rec, f1, auc = metrics(xv, yv, w, b)
        print(f"{name}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")
    for g in sorted(set(groups)):
        m = groups == g
        prec, rec, f1, auc = metrics(x[m], y[m], w, b)
        print(f"  {g}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")

    print("\n// paste into src/lib.rs (features: [ln(score), nccf, cpp], raw units)")
    print(f"const FUSION_BIAS: f64 = {float(b)!r};")
    print(
        "const FUSION_WEIGHTS: [f64; 3] = "
        f"[{float(w[0])!r}, {float(w[1])!r}, {float(w[2])!r}];"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
