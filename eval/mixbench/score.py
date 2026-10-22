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
    ... score.py --split test --primary
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REIM_BENCH_DATA", Path.home() / "datasets" / "reim-mixbench"))

TOLERANCE_CENTS = 50.0
MAX_EST_DT = 0.010  # an estimate frame further than this from a reference frame is unvoiced
MAX_REF_DT = 0.005  # interferer reference frame lookup (10 ms hop)
METRICS = ["f1", "precision", "oa", "rpa", "rpa_overlap", "vr", "vfa", "ifa", "gpe", "capture", "fpe"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("methods", nargs="*")
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--compare", action="store_true", help="print summary + paired deltas vs the first method")
    parser.add_argument("--primary", action="store_true", help="print the fixed primary comparisons")
    parser.add_argument("--ref-shift-ms", type=float, default=0.0, help="move every reference frame by this much")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    shift_dir = f"shift{args.ref_shift_ms:+g}ms" if args.ref_shift_ms else ""
    if args.self_test:
        return self_test()
    if args.compare:
        return compare(args.split, args.methods, shift_dir)
    if args.primary:
        return primary(args.split, shift_dir)
    for method in args.methods:
        rows = score_method(args.split, method, args.ref_shift_ms / 1000)
        out = HERE / "results" / args.split / shift_dir / f"{method}.scores.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{method}: {len(rows)} clips -> {out.relative_to(HERE)}")
    return 0


def score_method(split, method, ref_shift):
    pred_dir = DATA / "preds" / split / method
    rows = []
    for line in (HERE / "manifest" / f"{split}.jsonl").read_text().splitlines():
        m = json.loads(line)
        name = m["id"].split("/", 1)[1].replace("/", "__")
        refs = json.loads((DATA / "mix" / split / f"{name}.refs.json").read_text())
        for track in [refs["target"], *refs["interferers"]]:
            track["time"] = [t + ref_shift for t in track["time"]]
        est_t, est_f = read_prediction(pred_dir / f"{name}.csv")
        speaker = m["target"]["source"].split(":")[1].split("/")[0]
        rows.append({"id": m["id"], "condition": m["condition"], "speaker": speaker, **clip_metrics(refs, est_t, est_f)})
    return rows


def read_prediction(path):
    t, f = [], []
    with open(path) as fh:
        for row in csv.reader(fh):
            try:
                time, f0 = float(row[0]), float(row[1])
            except (ValueError, IndexError):
                continue
            t.append(time)
            f.append(f0)
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
        "ifa": ratio((~t_voiced & e_voiced & i_voiced).sum(), (~t_voiced & i_voiced).sum()),
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

# Fixed before the test split was scored: (method, baseline, question).
PRIMARY = [
    ("dfn3+reim-live-gate", "reim-live-gate", "DeepFilterNet3 front end for live reim"),
    ("dfn3+reim-live-gate+lock", "dfn3+reim-live-gate", "causal clearest-voice lock"),
    ("dfn3+reim-live-gate+lock-offline", "dfn3+reim-live-gate+lock", "offline lock over causal lock"),
    ("swiftf0", "dfn3+reim-live-gate", "SwiftF0 over the best live reim chain"),
]
GROUPS = ["clean", "noise", "talker", "ahead", "babble"]


def compare(split, methods, shift_dir):
    tables = {m: load_scores(split, m, shift_dir) for m in methods}
    base = methods[0]
    conditions = list(dict.fromkeys(r["condition"] for r in tables[base].values()))
    for metric in ["f1", "rpa", "precision", "capture", "ifa", "vfa"]:
        print(f"\n## {metric}: mean, 95% t interval over speakers; paired delta vs {base}\n")
        print("| condition | " + " | ".join(methods) + " |")
        print("|---" * (len(methods) + 1) + "|")
        for cond in conditions:
            cells = []
            for m in methods:
                ids = [i for i, r in tables[m].items() if r["condition"] == cond]
                est = speaker_t(tables[m], ids, metric)
                if est is None:
                    cells.append("-")
                    continue
                cell = f"{est[0]:.3f} [{est[1]:.3f}, {est[2]:.3f}]"
                if m != base:
                    d = speaker_t(paired(tables[m], tables[base]), ids, metric)
                    cell += f" Δ{d[0]:+.3f}{'*' if d[3] < 0.05 else ''}"
                cells.append(cell)
            print(f"| {cond} | " + " | ".join(cells) + " |")
    print("\n* = paired two-sided t test over speakers, p < 0.05, not corrected for multiple comparisons.")
    return 0


def primary(split, shift_dir):
    """The PRIMARY comparisons on F1 per condition group, with Holm-adjusted p values."""
    rows = []
    for method, baseline, question in PRIMARY:
        diff = paired(load_scores(split, method, shift_dir), load_scores(split, baseline, shift_dir))
        for group in GROUPS:
            ids = [i for i, r in diff.items() if r["condition"].split("_")[0] == group]
            mean, lo, hi, p = speaker_t(diff, ids, "f1")
            rows.append([f"{method} vs {baseline}", question, group, mean, lo, hi, p])
    order = np.argsort([r[-1] for r in rows])
    adjusted, running = [0.0] * len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[i][-1]))
        adjusted[i] = running
    print("| comparison | group | ΔF1 | 95% CI | p | Holm p |")
    print("|---|---|---|---|---|---|")
    for r, holm in zip(rows, adjusted, strict=True):
        print(f"| {r[0]} | {r[2]} | {r[3]:+.3f} | [{r[4]:+.3f}, {r[5]:+.3f}] | {r[6]:.4f} | {holm:.4f} |")
    return 0


def load_scores(split, method, shift_dir=""):
    with open(HERE / "results" / split / shift_dir / f"{method}.scores.csv") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    for r in rows.values():
        for k in METRICS:
            r[k] = float(r[k])
    return rows


def paired(a, b):
    """Per-clip differences a - b, joined on mixture id."""
    if a.keys() != b.keys():
        raise ValueError("score files cover different mixtures")
    return {i: {**a[i], **{k: a[i][k] - b[i][k] for k in METRICS}} for i in a}


def speaker_t(rows, ids, metric):
    """Mean of speaker means, its 95% t interval, and the p value against 0.

    Clips of one speaker are averaged first, so the speakers are the
    independent units (df = speakers - 1). None when no clip has a value."""
    by_speaker = {}
    for i in ids:
        v = rows[i][metric]
        if not math.isnan(v):
            by_speaker.setdefault(rows[i]["speaker"], []).append(v)
    if not by_speaker:
        return None
    means = np.array([np.mean(v) for v in by_speaker.values()])
    mean, k = float(means.mean()), len(means)
    if k < 2 or means.std() == 0:
        return mean, mean, mean, 1.0 if mean == 0 else 0.0
    se = means.std(ddof=1) / math.sqrt(k)
    half = stats.t.ppf(0.975, k - 1) * se
    return mean, mean - half, mean + half, float(2 * stats.t.sf(abs(mean) / se, k - 1))


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
    assert captured["rpa_overlap"] == 0.0 and captured["ifa"] == 1.0, captured
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
    # A row that does not parse as two numbers is skipped whole.
    path = Path(tempfile.mkstemp(suffix=".csv")[1])
    path.write_text("time,f0\n0.11\n0.12,abc\n0.13,200,extra\n")
    assert [list(a) for a in read_prediction(path)] == [[0.13], [200.0]]
    path.unlink()

    # Speaker means are the units: two speakers with means 0.2 and 0.4.
    rows = {
        "a": {"speaker": "S1", "f1": 0.1},
        "b": {"speaker": "S1", "f1": 0.3},
        "c": {"speaker": "S2", "f1": 0.4},
    }
    mean, lo, hi, p = speaker_t(rows, ["a", "b", "c"], "f1")
    half = stats.t.ppf(0.975, 1) * 0.1
    assert np.isclose(mean, 0.3) and np.isclose(lo, 0.3 - half) and np.isclose(hi, 0.3 + half), (mean, lo, hi)
    assert np.isclose(p, 2 * stats.t.sf(3.0, 1)), p
    print("score self-test PASSED")
    return 0


def resample_track(est_t, t, f):
    """Value of the nearest frame of a uniform 10 ms track starting at t[0]."""
    i = np.round((est_t - t[0]) / 0.01).astype(int)
    inside = (i >= 0) & (i < len(t))
    return np.where(inside, f[i.clip(0, len(t) - 1)], 0.0)


if __name__ == "__main__":
    sys.exit(main())
