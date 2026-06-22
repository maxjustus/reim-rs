#!/usr/bin/env python3
"""Analysis->synthesis fidelity for the reim parametric vocoder.

reim regenerates excitation phase (velvet noise + min-phase pulses), so the
resynthesis is NOT sample-aligned to the input and raw waveform SNR / SI-SDR is
meaningless. This script uses two phase-invariant spectral metrics over the
analysis->synthesis round trip:

  MCD  Mel-Cepstral Distortion (dB) -- the standard vocoder fidelity metric.
       magnitude mel-spectrogram -> natural log -> DCT (MFCC) -> drop c0 ->
       Euclidean distance per frame, scaled by K = 10*sqrt(2)/ln(10).
  LSD  Log-Spectral Distance (dB) on the magnitude STFT -- RMS over frequency of
       the 20*log10 magnitude difference, averaged over frames. Sensitive to the
       harmonic-vs-noise balance that the upcoming aperiodicity work changes.

Both ignore phase (magnitude only), so they survive reim's phase regeneration.
The synthesis group delay (~fftsize/2 samples, larger in practice) is removed by
estimating the integer-sample lag from the energy envelope, then refining it on a
small grid that minimises LSD, before framing.

Run:
  uv run --no-project --python 3.11 --with numpy --with soundfile --with librosa \
    python eval/roundtrip_quality.py --reim ./target/release/reim
  uv run --no-project --python 3.11 --with numpy --with soundfile --with librosa \
    python eval/roundtrip_quality.py --self-test
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# --- metric parameters (fixed so numbers are comparable across runs) ---------
N_FFT = 1024
HOP = 256
N_MELS = 40
N_MFCC = 25          # incl. c0, which is dropped (energy term)
EPS = 1e-9
MCD_K = 10.0 * np.sqrt(2.0) / np.log(10.0)   # standard MCD constant ~= 6.142
ACTIVE_THRESH = 0.1  # frame active if input RMS > thresh * max input RMS


def mel_cepstrum(y, sr):
    """Magnitude mel-spectrogram -> natural-log -> DCT-II (MFCC), shape (N_MFCC, T).

    power=1.0 keeps the cepstrum in magnitude (not power) units so the standard
    MCD constant applies; using power=2.0 would double the scale (a real bug we
    hit while building this)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0
    )
    return librosa.feature.mfcc(S=np.log(mel + EPS), n_mfcc=N_MFCC)


def log_mag_stft(y):
    """20*log10 magnitude STFT, shape (1 + N_FFT/2, T)."""
    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    return 20.0 * np.log10(mag + 1e-8)


def active_mask(ref, n_frames):
    """Boolean per-frame mask: frames whose input RMS exceeds the threshold."""
    rms = librosa.feature.rms(y=ref, frame_length=N_FFT, hop_length=HOP)[0]
    rms = rms[:n_frames]
    return rms > ACTIVE_THRESH * rms.max()


def mcd_lsd(ref, syn, sr):
    """Frame-wise MCD and LSD over active frames. ref/syn already time-aligned.

    Returns (mcd_db, lsd_db, n_active, n_total)."""
    Cr = mel_cepstrum(ref, sr)[1:]   # drop c0
    Cs = mel_cepstrum(syn, sr)[1:]
    Lr = log_mag_stft(ref)
    Ls = log_mag_stft(syn)
    t = min(Cr.shape[1], Cs.shape[1], Lr.shape[1], Ls.shape[1])
    Cr, Cs, Lr, Ls = Cr[:, :t], Cs[:, :t], Lr[:, :t], Ls[:, :t]

    mask = active_mask(ref, t)
    if mask.sum() == 0:                      # silent clip -> use all frames
        mask = np.ones(t, dtype=bool)

    mcd_per = MCD_K * np.sqrt(np.sum((Cr - Cs) ** 2, axis=0))
    lsd_per = np.sqrt(np.mean((Lr - Ls) ** 2, axis=0))
    return (
        float(mcd_per[mask].mean()),
        float(lsd_per[mask].mean()),
        int(mask.sum()),
        int(t),
    )


def estimate_lag(ref, syn, sr):
    """Integer-sample delay of `syn` relative to `ref`.

    Coarse: cross-correlate the RMS energy envelopes (phase-insensitive, so it
    works on reim's phase-regenerated output). Fine: search a small grid around
    the coarse estimate for the lag that minimises LSD over active frames. Only
    non-negative lags are considered (synthesis is delayed, never advanced)."""
    env_hop = 64
    er = librosa.feature.rms(y=ref, frame_length=512, hop_length=env_hop)[0]
    es = librosa.feature.rms(y=syn, frame_length=512, hop_length=env_hop)[0]
    n = min(len(er), len(es))
    er = er[:n] - er[:n].mean()
    es = es[:n] - es[:n].mean()
    if er.std() < 1e-6 or es.std() < 1e-6:
        # flat envelope (e.g. steady tone) carries no lag information.
        return 0
    corr = np.correlate(es, er, mode="full")
    lags = np.arange(-n + 1, n)
    coarse = int(lags[np.argmax(corr)] * env_hop)
    coarse = max(coarse, 0)

    best_lag, best_lsd = coarse, np.inf
    for lag in range(max(coarse - env_hop, 0), coarse + env_hop + 1, HOP // 4 or 1):
        s = syn[lag:]
        m = min(len(ref), len(s))
        if m < N_FFT * 4:
            continue
        _, lsd, _, _ = mcd_lsd(ref[:m], s[:m], sr)
        if lsd < best_lsd:
            best_lsd, best_lag = lsd, lag
    return best_lag


def align(ref, syn, sr, lag=None):
    """Remove the synthesis group delay; returns (ref, syn) of equal length."""
    if lag is None:
        lag = estimate_lag(ref, syn, sr)
    if lag > 0:
        syn = syn[lag:]
    elif lag < 0:
        ref = ref[-lag:]
    m = min(len(ref), len(syn))
    return ref[:m], syn[:m], lag


# --- reim invocation ---------------------------------------------------------
def reim_process(reim_bin, in_wav):
    """Run `reim process` and return (audio float64 mono, sr, command string)."""
    out = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [str(reim_bin), "process", str(in_wav), str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    y, sr = sf.read(str(out))
    out.unlink(missing_ok=True)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), sr, " ".join(cmd)


def eval_clip(reim_bin, in_wav):
    ref, sr = sf.read(str(in_wav))
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    ref = ref.astype(np.float64)
    syn, sr_o, cmd = reim_process(reim_bin, in_wav)
    assert sr_o == sr, f"sr mismatch {sr_o} vs {sr}"
    ref_a, syn_a, lag = align(ref, syn, sr)
    mcd, lsd, n_act, n_tot = mcd_lsd(ref_a, syn_a, sr)
    return dict(name=Path(in_wav).name, mcd=mcd, lsd=lsd, lag=lag,
                active_frac=n_act / n_tot if n_tot else 0.0,
                n_active=n_act, n_total=n_tot, cmd=cmd)


# --- self-test ---------------------------------------------------------------
def make_tone(sr=16000, dur=2.0, seed=0):
    """Harmonic tone with a slow amplitude envelope (so lag estimation has
    something to lock onto)."""
    t = np.arange(int(sr * dur)) / sr
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)        # 1.5 Hz tremolo
    x = (0.5 * np.sin(2 * np.pi * 220 * t)
         + 0.25 * np.sin(2 * np.pi * 440 * t)
         + 0.12 * np.sin(2 * np.pi * 880 * t))
    return (env * x)


def lowpass_fft(x, sr, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X[f > fc] = 0.0
    return np.fft.irfft(X, len(x))


def self_test():
    sr = 16000
    x = make_tone(sr)

    # (1) identical signal -> MCD and LSD must be ~0.
    r, s, _ = align(x, x.copy(), sr, lag=0)
    mcd0, lsd0, _, _ = mcd_lsd(r, s, sr)
    print(f"[self-test 1] tone vs itself:           MCD={mcd0:.4f} dB  LSD={lsd0:.4f} dB")
    assert mcd0 < 1e-6 and lsd0 < 1e-6, "identical signal must score 0"

    # (2) spectrally distorted (low-pass) -> clearly > 0 and >> case 1.
    xl = lowpass_fft(x, sr, 300.0)   # removes the 440/880 Hz harmonics
    r, s, _ = align(x, xl, sr, lag=0)
    mcd1, lsd1, _, _ = mcd_lsd(r, s, sr)
    print(f"[self-test 2] tone vs low-pass(300 Hz):  MCD={mcd1:.4f} dB  LSD={lsd1:.4f} dB")
    assert mcd1 > 5.0 and mcd1 > mcd0 + 5.0, "distortion must raise MCD"
    assert lsd1 > 3.0 and lsd1 > lsd0 + 3.0, "distortion must raise LSD"

    # (3) phase/time invariance: an integer-hop delay of the SAME signal, once the
    #     delay is compensated, frames are bit-identical -> metrics return to ~0.
    #     This proves the metric depends only on magnitude, not on phase/time.
    delay = 4 * HOP
    xd = np.concatenate([np.zeros(delay), x])
    r, s, lag = align(x, xd, sr, lag=delay)
    mcd2, lsd2, _, _ = mcd_lsd(r, s, sr)
    print(f"[self-test 3] tone vs delayed({delay}):    MCD={mcd2:.4f} dB  LSD={lsd2:.4f} dB  (compensated lag={lag})")
    assert mcd2 < 1e-6 and lsd2 < 1e-6, "delay-compensated identical signal must score 0 (phase invariance)"

    # (3b) the lag estimator recovers an integer-hop delay from the envelope alone.
    est = estimate_lag(x, xd, sr)
    print(f"[self-test 3b] estimated lag={est} (true {delay})")
    assert abs(est - delay) <= HOP, "lag estimate must land within one hop of truth"

    print("SELF-TEST PASSED")
    return True


# --- CLI ---------------------------------------------------------------------
def find_clips(data_dir):
    audio = Path(data_dir) / "Audio"
    if not audio.is_dir():
        return []
    return sorted(audio.glob("vocadito_*.wav"),
                  key=lambda p: int(p.stem.split("_")[1]))


def print_table(rows):
    print(f"\n{'clip':<22}{'MCD(dB)':>9}{'LSD(dB)':>9}{'lag':>7}{'active':>9}{'frames':>9}")
    print("-" * 65)
    for r in rows:
        print(f"{r['name']:<22}{r['mcd']:>9.3f}{r['lsd']:>9.3f}{r['lag']:>7}"
              f"{r['active_frac']:>8.0%}{r['n_total']:>9}")
    print("-" * 65)
    mcd = np.mean([r["mcd"] for r in rows])
    lsd = np.mean([r["lsd"] for r in rows])
    af = np.mean([r["active_frac"] for r in rows])
    print(f"{'MEAN ('+str(len(rows))+' clips)':<22}{mcd:>9.3f}{lsd:>9.3f}{'':>7}{af:>8.0%}")


def main():
    ap = argparse.ArgumentParser(
        description="Analysis->synthesis fidelity (MCD/LSD) for the reim vocoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--reim", default="./target/release/reim",
                    help="path to the reim binary")
    ap.add_argument("--data-dir", default="/private/tmp/datasets/vocadito_raw",
                    help="Vocadito root (expects Audio/vocadito_N.wav)")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate at most N clips (0 = all)")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic correctness checks and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    clips = find_clips(args.data_dir)
    if not clips:
        print(f"no Vocadito clips under {args.data_dir!r}; running self-test instead.\n")
        self_test()
        return 0

    self_test()  # always verify the metric before reporting real numbers
    print()

    reim_bin = Path(args.reim)
    if not reim_bin.exists():
        print(f"reim binary not found at {reim_bin}", file=sys.stderr)
        return 2

    if args.limit:
        clips = clips[: args.limit]
    print(f"reim: {reim_bin}   clips: {len(clips)}")
    rows = []
    for c in clips:
        r = eval_clip(reim_bin, c)
        rows.append(r)
        print(f"  {r['name']:<22} MCD={r['mcd']:6.3f}  LSD={r['lsd']:6.3f}  "
              f"lag={r['lag']:5d}  active={r['active_frac']:.0%}")
    print(f"\ninvocation: {rows[0]['cmd']}")
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
