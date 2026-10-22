#!/usr/bin/env python3
"""Run one target-pitch method over every mixture of a split.

Writes one "time,f0" CSV per mixture to $REIM_BENCH_DATA/preds/<split>/<method>/
and a run record (tool hashes, versions, timing, prediction digest) to
results/<split>/<method>.run.json. Tools live in $REIM_BENCH_DATA/tools; see
README.md for how to build them.

Run:
    uv run --no-project --with-requirements eval/mixbench/requirements-methods.txt \
        python eval/mixbench/run.py --split dev reim-live praat ...
"""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import lock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = Path(os.environ.get("REIM_BENCH_DATA", Path.home() / "datasets" / "reim-mixbench"))
TOOLS = DATA / "tools"
REIM = REPO / "target" / "release" / "reim"
FMIN, FMAX = 71.0, 800.0

LIVE_REIM = {"REIM_VOICING_REFINE": "0", "REIM_PITCH_REFINE": "0"}


def reim_f0(wav, env=None):
    out = subprocess.run(
        [str(REIM), "f0", str(wav), str(FMIN), str(FMAX)],
        capture_output=True, text=True, check=True, env={**os.environ, **(env or {})},
    ).stdout
    rows = np.array([[float(v) for v in line.split(",")] for line in out.splitlines()])
    return rows[:, 0], rows[:, 1]


def praat(wav):
    import parselmouth

    pitch = parselmouth.Sound(str(wav)).to_pitch_ac(time_step=0.01, pitch_floor=FMIN, pitch_ceiling=FMAX)
    return pitch.xs(), pitch.selected_array["frequency"]


def pyin(wav):
    import librosa

    y, sr = librosa.load(str(wav), sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=1024, hop_length=160)
    return librosa.times_like(f0, sr=sr, hop_length=160), np.where(voiced, f0, 0.0)


def swiftf0(wav):
    from swift_f0 import SwiftF0

    r = SwiftF0(threads=1).detect_file(str(wav), fmin=FMIN, fmax=FMAX)
    return r.timestamps, np.where(r.confidence >= 0.5, r.pitch_hz, 0.0)


def deep_filter(wav, tmp):
    subprocess.run(
        [str(TOOLS / "deep-filter-0.5.6"), "-D", "-o", str(tmp), str(wav)],
        capture_output=True, check=True,
    )
    return Path(tmp) / Path(wav).name


def rnnoise(wav, tmp):
    x, fs = sf.read(wav, dtype="float64")
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    raw_in, raw_out = Path(tmp) / "in.raw", Path(tmp) / "out.raw"
    raw_in.write_bytes(pcm.tobytes())
    subprocess.run([str(TOOLS / "rnnoise_demo-70f1d25"), str(raw_in), str(raw_out)], capture_output=True, check=True)
    y = np.frombuffer(raw_out.read_bytes(), dtype="<i2").astype(np.float64) / 32768
    y = y[RNNOISE_DELAY:]  # rnnoise_demo output lags its input by one 10 ms frame
    out = Path(tmp) / "rnnoise.wav"
    sf.write(out, y, fs, subtype="FLOAT")
    return out


RNNOISE_DELAY = 480


def wiener(wav, tmp):
    out = Path(tmp) / "wiener.wav"
    subprocess.run([str(TOOLS / "reim-enhance-19ace40"), "enhance", str(wav), str(out)], capture_output=True, check=True)
    return out


def front_end(enhance, env=LIVE_REIM):
    def run(wav):
        with tempfile.TemporaryDirectory() as tmp:
            return reim_f0(enhance(wav, tmp), env)

    return run


# name: (function, mode, description, tool files, python packages)
METHODS = {
    "reim-live": (lambda w: reim_f0(w, LIVE_REIM), "live", "reim f0, causal voicing and pitch (no offline refinement)", [], []),
    "reim-offline": (reim_f0, "offline", "reim f0 defaults: offline Viterbi voicing + pitch refinement", [], []),
    "reim-live-gate": (
        lambda w: reim_f0(w, {**LIVE_REIM, "REIM_VOICING_SCORE_MIN": "0.6"}),
        "live", "reim f0 causal, fused voicing gate at 0.6", [], [],
    ),
    "praat": (praat, "offline", "Praat autocorrelation (parselmouth to_pitch_ac, Viterbi path), 10 ms", [], ["praat-parselmouth"]),
    "pyin": (pyin, "offline", "librosa pYIN at 16 kHz, 64 ms frame, 10 ms hop (HMM decoding)", [], ["librosa", "soxr"]),
    "swiftf0": (swiftf0, "live", "SwiftF0 0.3.0, confidence >= 0.5 (176 ms lookahead when streamed)", [], ["swift-f0", "onnxruntime"]),
    "dfn3+reim-live": (front_end(deep_filter), "live", "DeepFilterNet3 (deep-filter 0.5.6, -D) then reim-live", ["deep-filter-0.5.6"], []),
    "dfn3+reim-live-gate": (
        front_end(deep_filter, {**LIVE_REIM, "REIM_VOICING_SCORE_MIN": "0.6"}),
        "live", "DeepFilterNet3 (deep-filter 0.5.6, -D) then reim-live-gate", ["deep-filter-0.5.6"], [],
    ),
    "rnnoise+reim-live": (front_end(rnnoise), "live", "RNNoise (xiph main 70f1d25, 16-bit, 10 ms delay removed) then reim-live", ["rnnoise_demo-70f1d25"], []),
    "wiener+reim-live": (front_end(wiener), "offline", "reim enhance (voice-enhancement 19ace40, two-pass Wiener) then reim-live", ["reim-enhance-19ace40"], []),
}

# base: ((level_db, range_cents) causal, (level_db, range_cents) offline), chosen by tune_lock.py on dev
LOCKS = {
    "reim-live-gate": ((12, 600), (9, 800)),
    "dfn3+reim-live-gate": ((12, 600), (9, 800)),
    "swiftf0": ((9, math.inf), (6, 800)),
}


def locked(base, offline, level_db, range_cents):
    def run(wav):
        t, f0 = METHODS[base][0](wav)
        x, fs = sf.read(wav, dtype="float64")
        levels = lock.frame_levels(x, fs, t)
        if offline:
            return t, lock.lock_offline(f0, levels, level_db, range_cents)
        return t, lock.lock_causal(t, f0, levels, level_db, range_cents)

    return run


for _base, (_live, _offline) in LOCKS.items():
    METHODS[f"{_base}+lock"] = (
        locked(_base, False, *_live), METHODS[_base][1],
        f"{_base}, then causal target lock (level_db={_live[0]}, range_cents={_live[1]})",
        METHODS[_base][3], METHODS[_base][4],
    )
    METHODS[f"{_base}+lock-offline"] = (
        locked(_base, True, *_offline), "offline",
        f"{_base}, then offline target lock (level_db={_offline[0]}, range_cents={_offline[1]})",
        METHODS[_base][3], METHODS[_base][4],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("methods", nargs="+", choices=list(METHODS))
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    for method in args.methods:
        run_method(args.split, method, args.jobs)
    return 0


def run_method(split, method, jobs):
    fn, mode, description, tools, packages = METHODS[method]
    names = [
        json.loads(line)["id"].split("/", 1)[1].replace("/", "__")
        for line in (HERE / "manifest" / f"{split}.jsonl").read_text().splitlines()
    ]
    out_dir = DATA / "preds" / split / method
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = [DATA / "mix" / split / f"{n}.wav" for n in names]
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        elapsed = list(pool.map(predict, [method] * len(wavs), wavs, [out_dir / f"{n}.csv" for n in names]))
    digest = hashlib.sha256()
    for n in names:
        digest.update(f"{n} {hashlib.sha256((out_dir / f'{n}.csv').read_bytes()).hexdigest()}\n".encode())
    audio_s = sum(sf.info(w).duration for w in wavs)
    record = {
        "method": method,
        "mode": mode,
        "description": description,
        "split": split,
        "clips": len(names),
        "predictions_sha256": digest.hexdigest(),
        "code": code_identity(split),
        "tools": {t: file_sha256(TOOLS / t) for t in tools},
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}, Python {platform.python_version()}",
        "audio_seconds": round(audio_s, 3),
        "process_seconds": round(sum(elapsed), 3),
        "real_time_factor": round(sum(elapsed) / audio_s, 5),
        "wall_seconds": round(time.time() - started, 1),
    }
    out = HERE / "results" / split / f"{method}.run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{method}: {len(names)} clips, RTF {record['real_time_factor']}, digest {record['predictions_sha256'][:12]}")


def predict(method, wav, out):
    started = time.perf_counter()
    t, f0 = METHODS[method][0](wav)
    f0 = np.nan_to_num(np.asarray(f0, dtype=np.float64), nan=0.0)
    out.write_text("".join(f"{a:.6f},{b:.4f}\n" for a, b in zip(t, f0, strict=True)))
    return time.perf_counter() - started


def code_identity(split):
    """Commit, whether the code paths differ from it (untracked files included), and file hashes."""
    git = ["git", "-C", str(REPO)]
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    changes = subprocess.run(
        [*git, "status", "--porcelain", "--", "src", "Cargo.toml", "Cargo.lock", "eval/mixbench", ":!eval/mixbench/results"],
        capture_output=True, text=True, check=True,
    ).stdout
    files = [HERE / "run.py", HERE / "lock.py", HERE / "manifest" / f"{split}.jsonl"]
    return {
        "commit": commit,
        "clean": not changes,
        "reim_sha256": file_sha256(REIM),
        "files": {str(f.relative_to(REPO)): file_sha256(f) for f in files},
    }


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
