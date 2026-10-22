#!/usr/bin/env python3
"""Score target-pitch predictions against the mixbench references.

Input: one "time,f0" CSV per mixture (f0 <= 0 = unvoiced, header optional)
under $REIM_BENCH_DATA/preds/<split>/<method>/, and the reference sidecars
that `build.py render` writes. Output: per-clip metrics in
results/<split>/<method>.scores.csv, and with --compare a summary table.

Every metric is evaluated on the target's 10 ms reference grid; README.md
defines each one.

Run:
    uv run --no-project --with-requirements eval/mixbench/requirements.txt \
        python eval/mixbench/score.py --split dev reim-live [method ...]
    ... score.py --split dev --compare reim-live praat dfn3+reim-live
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REIM_BENCH_DATA", Path.home() / "datasets" / "reim-mixbench"))

TOLERANCE_CENTS = 50.0
MAX_EST_DT = 0.010  # an estimate frame further than this from a reference frame is unvoiced
MAX_REF_DT = 0.005  # interferer reference frame lookup (10 ms hop)
METRICS = ["f1", "precision", "oa", "rpa", "rpa_overlap", "vr", "vfa", "gpe", "capture", "fpe"]
BOOTSTRAP = 2000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("methods", nargs="*")
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--compare", action="store_true", help="print summary + paired deltas vs the first method")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.compare:
        return compare(args.split, args.methods)
    for method in args.methods:
        rows = score_method(args.split, method)
        out = HERE / "results" / args.split / f"{method}.scores.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{method}: {len(rows)} clips -> {out.relative_to(HERE)}")
    return 0


def score_method(split, method):
    pred_dir = DATA / "preds" / split / method
    rows = []
    for line in (HERE / "manifest" / f"{split}.jsonl").read_text().splitlines():
        m = json.loads(line)
        name = m["id"].split("/", 1)[1].replace("/", "__")
        refs = json.loads((DATA / "mix" / split / f"{name}.refs.json").read_text())
        est_t, est_f = read_prediction(pred_dir / f"{name}.csv")
        speaker = m["target"]["source"].split(":")[1].split("/")[0]
        rows.append({"id": m["id"], "condition": m["condition"], "speaker": speaker, **clip_metrics(refs, est_t, est_f)})
    return rows


def read_prediction(path):
    t, f = [], []
    with open(path) as fh:
        for row in csv.reader(fh):
            try:
                t.append(float(row[0]))
                f.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return np.array(t), np.array(f)


def clip_metrics(refs, est_t, est_f):
    ref_t = np.array(refs["target"]["time"])
    ref_f = np.array(refs["target"]["f0"])
    est = sample_nearest(est_t, est_f, ref_t, MAX_EST_DT)
    interferers = [
        sample_nearest(np.array(r["time"]), np.array(r["f0"]), ref_t, MAX_REF_DT)
        for r in refs["interferers"]
    ]

    t_voiced = ref_f > 0
    e_voiced = est > 0
    cents = np.full(len(ref_t), np.inf)
    both = t_voiced & e_voiced
    cents[both] = np.abs(1200 * np.log2(est[both] / ref_f[both]))
    hit = cents <= TOLERANCE_CENTS
    i_voiced = np.zeros(len(ref_t), dtype=bool)
    i_match = np.zeros(len(ref_t), dtype=bool)
    for f in interferers:
        v = f > 0
        i_voiced |= v
        ok = v & e_voiced
        i_match[ok] |= np.abs(1200 * np.log2(est[ok] / f[ok])) <= TOLERANCE_CENTS

    precision = ratio(hit.sum(), e_voiced.sum())
    recall = ratio(hit.sum(), t_voiced.sum())
    return {
        "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        "precision": precision,
        "oa": ratio(hit.sum() + (~t_voiced & ~e_voiced).sum(), len(ref_t)),
        "rpa": recall,
        "rpa_overlap": ratio((hit & i_voiced).sum(), (t_voiced & i_voiced).sum()),
        "vr": ratio(both.sum(), t_voiced.sum()),
        "vfa": ratio((~t_voiced & e_voiced).sum(), (~t_voiced).sum()),
        "gpe": ratio((both & ~hit).sum(), both.sum()),
        "capture": ratio((e_voiced & ~hit & i_match).sum(), i_voiced.sum()),
        "fpe": float(np.median(cents[hit])) if hit.any() else math.nan,
    }


def sample_nearest(t, f, grid, max_dt):
    """Value of the nearest frame in (t, f) at each grid time; 0 when none is within max_dt.

    Times are compared in whole microseconds, so a tie is exact and goes to the
    earlier frame. Of several rows with one time stamp, the last row counts."""
    if len(t) == 0:
        return np.zeros(len(grid))
    t, grid, max_dt = microseconds(t), microseconds(grid), microseconds(max_dt)
    order = np.argsort(t, kind="stable")
    t, f = t[order], np.asarray(f)[order]
    last = np.append(t[1:] != t[:-1], True)
    t, f = t[last], f[last]
    right = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    left = np.clip(right - 1, 0, len(t) - 1)
    i = np.where(np.abs(grid - t[left]) <= np.abs(t[right] - grid), left, right)
    out = np.where(np.abs(t[i] - grid) <= max_dt, f[i], 0.0)
    return np.maximum(out, 0.0)


def microseconds(seconds):
    return np.round(np.asarray(seconds, dtype=np.float64) * 1e6).astype(np.int64)


def ratio(num, den):
    return float(num) / float(den) if den else math.nan


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def compare(split, methods):
    tables = {m: load_scores(split, m) for m in methods}
    base = methods[0]
    conditions = list(dict.fromkeys(r["condition"] for r in tables[base]))
    rng = np.random.default_rng(0)
    for metric in ["f1", "rpa", "precision", "capture", "vfa", "oa"]:
        print(f"\n## {metric} (mean over clips, 95% CI by speaker-cluster bootstrap; delta vs {base})\n")
        print("| condition | " + " | ".join(methods) + " |")
        print("|---" * (len(methods) + 1) + "|")
        for cond in conditions:
            cells = []
            for m in methods:
                rows = [r for r in tables[m] if r["condition"] == cond]
                vals = np.array([r[metric] for r in rows])
                if np.all(np.isnan(vals)):
                    cells.append("–")
                    continue
                spk = np.array([r["speaker"] for r in rows])
                mean, lo, hi = cluster_bootstrap(vals, spk, rng)
                cell = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
                if m != base:
                    b = np.array([r[metric] for r in tables[base] if r["condition"] == cond])
                    d, dlo, dhi = cluster_bootstrap(vals - b, spk, rng)
                    mark = "*" if dlo > 0 or dhi < 0 else ""
                    cell += f" Δ{d:+.3f}{mark}"
                cells.append(cell)
            print(f"| {cond} | " + " | ".join(cells) + " |")
    print("\n* = 95% CI of the paired delta excludes 0.")
    return 0


def load_scores(split, method):
    with open(HERE / "results" / split / f"{method}.scores.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in METRICS:
            r[k] = float(r[k])
    return rows


def cluster_bootstrap(vals, clusters, rng):
    """Mean and 95% CI; resamples whole speakers, then averages their clips (NaN-aware)."""
    ok = ~np.isnan(vals)
    vals, clusters = vals[ok], clusters[ok]
    names = np.unique(clusters)
    groups = [vals[clusters == c] for c in names]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    picks = rng.integers(0, len(names), size=(BOOTSTRAP, len(names)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(vals.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def self_test():
    """Synthetic tracks with known answers for every metric."""
    t = np.round(0.016 + 0.01 * np.arange(100), 6)
    target = np.where((t > 0.2) & (t < 0.8), 200.0, 0.0)
    other = np.where(t > 0.5, 300.0, 0.0)
    refs = {
        "target": {"time": t.tolist(), "f0": target.tolist()},
        "interferers": [{"time": (t + 0.003).tolist(), "f0": other.tolist()}],
    }
    t_voiced = target > 0
    est_t = np.arange(0, 1.1, 0.005)

    def run(f):
        return clip_metrics(refs, est_t, resample_track(est_t, t, f))

    perfect = run(target)
    assert perfect["oa"] == 1.0 and perfect["rpa"] == 1.0 and perfect["vfa"] == 0.0, perfect
    assert perfect["capture"] == 0.0 and perfect["fpe"] == 0.0 and perfect["f1"] == 1.0, perfect

    sharp49 = run(target * 2 ** (49 / 1200))
    sharp51 = run(target * 2 ** (51 / 1200))
    assert sharp49["rpa"] == 1.0 and sharp51["rpa"] == 0.0 and sharp51["gpe"] == 1.0, (sharp49, sharp51)

    # Follows the interferer wherever it speaks, the target elsewhere.
    captured = run(np.where(other > 0, other, target))
    overlap = t_voiced & (other > 0)
    assert captured["capture"] == 1.0, captured
    assert np.isclose(captured["rpa"], (t_voiced & ~overlap).sum() / t_voiced.sum()), captured
    assert captured["rpa_overlap"] == 0.0, captured
    precision = (t_voiced & ~overlap).sum() / ((other > 0) | t_voiced).sum()
    assert np.isclose(captured["precision"], precision), captured
    assert np.isclose(captured["f1"], 2 * precision * captured["rpa"] / (precision + captured["rpa"])), captured
    assert np.isclose(captured["vfa"], ((other > 0) & ~t_voiced).sum() / (~t_voiced).sum()), captured

    silent = clip_metrics(refs, np.array([]), np.array([]))
    assert silent["rpa"] == 0.0 and silent["vr"] == 0.0 and np.isclose(silent["oa"], 1 - t_voiced.mean()), silent
    assert silent["capture"] == 0.0 and math.isnan(silent["fpe"]) and math.isnan(silent["precision"]), silent

    # Estimate frames every 40 ms, 2 ms late: grid frames 18 and 12 ms away read as unvoiced.
    sparse = clip_metrics(refs, t[::4] + 0.002, target[::4])
    near = np.arange(len(t)) % 4 < 2
    assert np.isclose(sparse["vr"], (t_voiced & near).sum() / t_voiced.sum()), sparse
    # A grid time exactly between two frames takes the earlier one; a repeated
    # time stamp takes the last row.
    grid = np.array([0.010])
    assert sample_nearest(np.array([0.005, 0.015]), np.array([100.0, 200.0]), grid, MAX_EST_DT)[0] == 100.0
    assert sample_nearest(np.array([0.010, 0.010]), np.array([100.0, 200.0]), grid, MAX_EST_DT)[0] == 200.0
    print("score self-test PASSED")
    return 0


def resample_track(est_t, t, f):
    """Value of the nearest frame of a uniform 10 ms track starting at t[0]."""
    i = np.round((est_t - t[0]) / 0.01).astype(int)
    inside = (i >= 0) & (i < len(t))
    return np.where(inside, f[i.clip(0, len(t) - 1)], 0.0)


if __name__ == "__main__":
    sys.exit(main())
