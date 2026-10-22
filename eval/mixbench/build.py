#!/usr/bin/env python3
"""Build the mixbench target-pitch benchmark: extract, manifest, render.

  extract   unpack the verified archives from fetch.py into WAV + reference
            files under $REIM_BENCH_DATA (default ~/datasets/reim-mixbench)
  manifest  regenerate manifest/{dev,test}.jsonl from the seed (maintainers
            only; the committed manifests are the benchmark definition)
  render    render every mixture in the committed manifests and check each
            source and mixture SHA-256 against the manifest

See README.md for the mixing rules and the verification protocol.

Run:
    uv run --no-project --with-requirements eval/mixbench/requirements.txt \
        python eval/mixbench/build.py extract|manifest|render
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REIM_BENCH_DATA", Path.home() / "datasets" / "reim-mixbench"))
FS = 48000
SEED = 20251021

# PTDB-TUG reference: RAPT, 32 ms window, 10 ms hop. Frame i sits at
# REF_T0 + i * REF_HOP; ref_offset.py measured REF_T0 against the laryngograph.
REF_T0 = 0.0195
REF_HOP = 0.010
REF_HALF_WINDOW = 0.016

SPLITS = {
    "dev": {
        "speakers": ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"],
        "envs": ["DKITCHEN", "NPARK", "OOFFICE", "PSTATION", "STRAFFIC", "TBUS"],
        "targets_per_speaker": 5,
    },
    "test": {
        "speakers": [f"F{i:02d}" for i in range(5, 11)] + [f"M{i:02d}" for i in range(5, 11)],
        "envs": [
            "DLIVING", "DWASHING", "NFIELD", "NRIVER", "OHALLWAY", "OMEETING",
            "PCAFETER", "PRESTO", "SCAFE", "SPSQUARE", "TCAR", "TMETRO",
        ],
        "targets_per_speaker": 7,
    },
}

# (condition, kind, level_db): level is SNR for noise, SIR otherwise.
CONDITIONS = [
    ("clean", "clean", None),
    ("noise_snr10", "noise", 10),
    ("noise_snr5", "noise", 5),
    ("noise_snr0", "noise", 0),
    ("talker_sir10", "talker", 10),
    ("talker_sir5", "talker", 5),
    ("talker_sir0", "talker", 0),
    ("ahead_sir10", "ahead", 10),
    ("ahead_sir5", "ahead", 5),
    ("babble_sir5", "babble", 5),
    ("babble_sir0", "babble", 0),
]
BABBLE_TALKERS = 3
PAD_S = 0.25  # target file kept around its active span
LEAD_RANGE_S = (0.5, 1.0)  # talker: after the target's active span starts; ahead/babble: target window start
GAP_RANGE_S = (0.1, 0.4)  # pause between an interferer's consecutive utterances
MIN_ACTIVE_S, MAX_DURATION_S = 2.5, 8.0
MIN_VOICED_FRAMES = 100
PEAK_LIMIT = 0.99


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["extract", "manifest", "render"])
    args = parser.parse_args()
    return {"extract": extract, "manifest": manifest, "render": render}[args.command]()


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def extract():
    extract_ptdb()
    extract_demand()
    return 0


def extract_ptdb():
    import pyarrow.parquet as pq

    cols = ["speaker_id", "utterance_id", "mic_audio", "f0_data"]
    for shard in sorted((DATA / "raw" / "ptdb").glob("*.parquet")):
        for row in pq.read_table(shard, columns=cols).to_pylist():
            key = f"{row['speaker_id']}/{row['utterance_id']}"
            wav = DATA / "ptdb" / "mic" / f"{key}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(row["mic_audio"]["bytes"])
            ref = DATA / "ptdb" / "ref" / f"{key}.f0.npy"
            ref.parent.mkdir(parents=True, exist_ok=True)
            np.save(ref, np.array([frame[0] for frame in row["f0_data"]], dtype=np.float64))
        print(f"extracted {shard.name}", flush=True)


def extract_demand():
    for archive in sorted((DATA / "raw" / "demand").glob("*_48k.zip")):
        env = archive.name.removesuffix("_48k.zip")
        with zipfile.ZipFile(archive) as z:
            data = z.read(f"{env}/ch01.wav")
        out = DATA / "demand" / f"{env}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"extracted {env}/ch01.wav", flush=True)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def source_path(source):
    corpus, key = source.split(":", 1)
    if corpus == "ptdb":
        return DATA / "ptdb" / "mic" / f"{key}.wav"
    return DATA / "demand" / f"{key}.wav"


def load_audio(source, start=0, frames=-1):
    x, fs = sf.read(source_path(source), start=start, frames=frames, dtype="float64", always_2d=True)
    if fs != FS:
        raise ValueError(f"{source}: {fs} Hz, expected {FS}")
    return x[:, 0]


def load_ref(source):
    key = source.split(":", 1)[1]
    return np.load(DATA / "ptdb" / "ref" / f"{key}.f0.npy")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref_sha256(source):
    """SHA-256 of the reference F0 column as float64 little-endian."""
    return hashlib.sha256(load_ref(source).astype("<f8").tobytes()).hexdigest()


def active_span(f0, n):
    """Sample range from the first to the last voiced reference frame."""
    voiced = np.flatnonzero(f0 > 0)
    lo = int(round((REF_T0 + REF_HOP * voiced[0] - REF_HALF_WINDOW) * FS))
    hi = int(round((REF_T0 + REF_HOP * voiced[-1] + REF_HALF_WINDOW) * FS))
    return max(lo, 0), min(hi, n)


def active_power(x, f0):
    lo, hi = active_span(f0, len(x))
    return float(np.mean(x[lo:hi] ** 2))


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def manifest():
    rng = np.random.default_rng(SEED)
    out_dir = HERE / "manifest"
    out_dir.mkdir(exist_ok=True)
    for split, spec in SPLITS.items():
        pool = eligible_utterances(spec["speakers"])
        envs = {e: sf.info(source_path(f"demand:{e}")).frames for e in spec["envs"]}
        lines = []
        for target_index, target in enumerate(pick_targets(rng, pool, spec)):
            lines.extend(mixtures_for_target(rng, split, target, target_index, pool, envs))
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in lines))
        print(f"{split}: {len(lines)} mixtures", flush=True)
    return 0


def eligible_utterances(speakers):
    pool = {}
    for spk in speakers:
        keys = []
        for wav in sorted((DATA / "ptdb" / "mic" / spk).glob("*.wav")):
            source = f"ptdb:{spk}/{wav.stem}"
            info = sf.info(wav)
            f0 = load_ref(source)
            lo, hi = active_span(f0, info.frames)
            if (
                (hi - lo) / FS >= MIN_ACTIVE_S
                and info.duration <= MAX_DURATION_S
                and np.count_nonzero(f0 > 0) >= MIN_VOICED_FRAMES
            ):
                keys.append(source)
        pool[spk] = keys
    return pool


def pick_targets(rng, pool, spec):
    targets = []
    for spk in spec["speakers"]:
        chosen = rng.choice(len(pool[spk]), spec["targets_per_speaker"], replace=False)
        targets.extend(pool[spk][i] for i in sorted(chosen))
    return targets


def mixtures_for_target(rng, split, target, target_index, pool, envs):
    """All conditions of one target, in five layouts. The target window is
    its active span plus PAD_S each side. In clean/noise/talker mixtures the
    target starts at sample 0 and the talker joins after a lead. In
    ahead/babble mixtures the interferers start at sample 0 and the target
    window starts after a lead."""
    spk = speaker_of(target)
    f0 = load_ref(target)
    x = load_audio(target)
    lo, hi = active_span(f0, len(x))
    w0, w1 = max(0, lo - int(PAD_S * FS)), min(len(x), hi + int(PAD_S * FS))
    window = w1 - w0
    p_target = active_power(x, f0)

    env = str(rng.choice(sorted(envs)))
    noise_start = int(rng.integers(0, envs[env] - window))
    noise_power = float(np.mean(load_audio(f"demand:{env}", noise_start, window) ** 2))

    others = [s for s in pool if s != spk]
    want_same = target_index % 2 == 0
    talker = str(rng.choice([s for s in others if (s[0] == spk[0]) == want_same]))
    talker_track = interferer_track(rng, talker, target, pool, lo - w0 + lead(rng), window)
    ahead_start = lead(rng)
    ahead_track = interferer_track(rng, talker, target, pool, 0, ahead_start + window)
    babble_speakers = [str(s) for s in rng.choice([s for s in others if s != talker], BABBLE_TALKERS, replace=False)]
    babble_start = lead(rng)
    babble_tracks = [interferer_track(rng, s, target, pool, 0, babble_start + window) for s in babble_speakers]
    layouts = {
        "clean": (0, []),
        "noise": (0, []),
        "talker": (0, [talker_track]),
        "ahead": (ahead_start, [ahead_track]),
        "babble": (babble_start, babble_tracks),
    }

    mixes = []
    for cond, kind, level in CONDITIONS:
        target_start, tracks = layouts[kind]
        m = {
            "id": f"{split}/{spk}_{target_key(target).split('/')[1]}/{cond}",
            "split": split,
            "condition": cond,
            "kind": kind,
            "level_db": level,
            "fs": FS,
            "length": target_start + window,
            "target": {
                "source": target,
                "sha256": file_sha256(source_path(target)),
                "ref_sha256": ref_sha256(target),
                "src_start": w0,
                "dst_start": target_start,
                "length": window,
                "gain": 1.0,
            },
            "interferers": [scaled(t, p_target, level, len(tracks)) for t in tracks],
            "noise": None,
        }
        if kind == "noise":
            m["noise"] = {
                "source": f"demand:{env}",
                "sha256": file_sha256(source_path(f"demand:{env}")),
                "src_start": noise_start,
                "gain": float(np.sqrt(p_target / (noise_power * 10 ** (level / 10)))),
            }
        y = mix(m)
        peak = float(np.max(np.abs(y)))
        m["out_gain"] = min(1.0, PEAK_LIMIT / peak) if peak > 0 else 1.0
        m["sha256"] = mixture_sha256(y * m["out_gain"])
        mixes.append(m)
    return mixes


def speaker_of(source):
    return source.split(":", 1)[1].split("/")[0]


def target_key(source):
    return source.split(":", 1)[1]


def lead(rng):
    return int(round(rng.uniform(*LEAD_RANGE_S) * FS))


def interferer_track(rng, spk, target, pool, start, n):
    """Concatenate the active spans of `spk`'s utterances (never the target's
    sentence), separated by short pauses, from sample `start` until the next
    utterance would start at or after sample `n`."""
    target_sentence = target.split("/")[1]
    candidates = [u for u in pool[spk] if u.split("/")[1] != target_sentence]
    pos = start
    segments = []
    for i in rng.permutation(len(candidates)):
        if pos >= n:
            break
        source = candidates[i]
        lo, hi = active_span(load_ref(source), sf.info(source_path(source)).frames)
        length = min(hi - lo, n - pos)
        segments.append({"source": source, "src_start": lo, "dst_start": pos, "length": length})
        pos += length + int(round(rng.uniform(*GAP_RANGE_S) * FS))
    return {"speaker": spk, "segments": segments}


def scaled(track, p_target, sir_db, talkers):
    """Gain so each talker's active power is p_target / (talkers * 10^(SIR/10))."""
    x = np.concatenate([load_audio(seg["source"], seg["src_start"], seg["length"]) for seg in track["segments"]])
    p = float(np.mean(x**2))
    g = float(np.sqrt(p_target / (talkers * 10 ** (sir_db / 10) * p)))
    return {
        "speaker": track["speaker"],
        "segments": [
            {**seg, "sha256": file_sha256(source_path(seg["source"])), "ref_sha256": ref_sha256(seg["source"]), "gain": g}
            for seg in track["segments"]
        ],
    }


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def mix(m):
    """Float64 sum in a fixed order: target, interferer segments, noise."""
    y = np.zeros(m["length"], dtype=np.float64)
    tgt = m["target"]
    y[tgt["dst_start"] : tgt["dst_start"] + tgt["length"]] += tgt["gain"] * load_audio(
        tgt["source"], tgt["src_start"], tgt["length"]
    )
    for interferer in m["interferers"]:
        for seg in interferer["segments"]:
            x = load_audio(seg["source"], seg["src_start"], seg["length"])
            y[seg["dst_start"] : seg["dst_start"] + seg["length"]] += seg["gain"] * x
    if m["noise"]:
        y += m["noise"]["gain"] * load_audio(m["noise"]["source"], m["noise"]["src_start"], m["length"])
    return y


def mixture_sha256(y):
    return hashlib.sha256(y.astype("<f4").tobytes()).hexdigest()


def render():
    failures = 0
    hashes = {}
    for split in SPLITS:
        out_dir = DATA / "mix" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for line in (HERE / "manifest" / f"{split}.jsonl").read_text().splitlines():
            m = json.loads(line)
            for source, digest, ref_digest in sources_of(m):
                if source not in hashes:
                    hashes[source] = (file_sha256(source_path(source)), ref_sha256(source) if ref_digest else None)
                if hashes[source] != (digest, ref_digest):
                    print(f"SOURCE MISMATCH {m['id']} {source}", flush=True)
                    failures += 1
            y = mix(m) * m["out_gain"]
            ok = mixture_sha256(y) == m["sha256"]
            failures += not ok
            name = m["id"].split("/", 1)[1].replace("/", "__")
            sf.write(out_dir / f"{name}.wav", y.astype(np.float32), FS, subtype="FLOAT")
            (out_dir / f"{name}.refs.json").write_text(json.dumps(references(m)))
            if not ok:
                print(f"MIXTURE MISMATCH {m['id']}", flush=True)
        print(f"rendered {split}", flush=True)
    print("render: all fingerprints match" if not failures else f"render: {failures} mismatches")
    return 1 if failures else 0


def sources_of(m):
    yield m["target"]["source"], m["target"]["sha256"], m["target"]["ref_sha256"]
    for interferer in m["interferers"]:
        for seg in interferer["segments"]:
            yield seg["source"], seg["sha256"], seg["ref_sha256"]
    if m["noise"]:
        yield m["noise"]["source"], m["noise"]["sha256"], None


def references(m):
    """Reference F0 tracks on the mixture timeline (times in seconds)."""
    def track(source, src_start, length, dst_start):
        f0 = load_ref(source)
        center = REF_T0 + REF_HOP * np.arange(len(f0))
        keep = (center * FS >= src_start) & (center * FS < src_start + length)
        t = center[keep] + (dst_start - src_start) / FS
        return {"time": t.round(6).tolist(), "f0": f0[keep].tolist()}

    return {
        "id": m["id"],
        "target": track(m["target"]["source"], m["target"]["src_start"], m["target"]["length"], m["target"]["dst_start"]),
        "interferers": [
            track(seg["source"], seg["src_start"], seg["length"], seg["dst_start"])
            for interferer in m["interferers"]
            for seg in interferer["segments"]
        ],
    }


if __name__ == "__main__":
    sys.exit(main())
