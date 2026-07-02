# reim-rs

A Rust port of [ReIm](https://github.com/nakakq/reim), a real-time
[WORLD](https://github.com/mmorise/World)-like speech vocoder by
[@nakakq](https://github.com/nakakq). The original C implementation is included
as a git submodule under `reim/` and serves as the correctness oracle for the
port.

ReIm analyzes a mono signal into three acoustic features compatible with
WORLD -- fundamental frequency (Fo), spectral envelope (Sp), and aperiodicity
(Ap) -- then resynthesizes it. The analysis order is Silence -> Fo -> Ap -> Sp.

The port is a single-file library (`src/lib.rs`) with a thin CLI (`src/main.rs`).
Add the crate as a dependency and `use reim::Reim`, or run the `reim` binary
directly.

## Design

- Allocation-free steady state: `Reim::process_sample` does no heap allocation;
  all working buffers are owned by the analyzer/synthesizer state and reused.
- Self-contained: hand-written radix-2 FFT, xorshift RNG, and WAV I/O.
- Faithful to the C reference for Fo/Sp/synthesis, matched to the f32 noise floor
  against a C oracle. Three small in-source deviations keep that output: the
  silence RMS skips a one-element out-of-bounds read present in the C; the Ap
  range guard mirrors the C's exact branch semantics (including NaN handling); and
  the Fo periodicity gate computes a real variance instead of the C's std-dev
  formula (sign error, same gate). All three are documented in the source.
- Aperiodicity is a deliberate departure: the C reference leaves the Ap analyzer
  unimplemented (a binary voiced/unvoiced flag), so ReIm instead computes WORLD's
  D4C band-aperiodicity. Voiced frames get a real per-band noise/periodic split
  (validated against WORLD's D4C to a per-frame aperiodicity MAE of ~0.004); the
  voiced/unvoiced decision itself is unchanged. This changes the synthesized
  waveform versus the C placeholder by design.

## Use as a library

The whole API is the `Reim` type: construct it once, then push samples through
`process_sample` (one in, one out) or `process_block`.

```rust
// Offline / block: analyze + resynthesize a buffer.
let mut reim = Reim::with_defaults(48_000.0); // fs; period 5 ms, fftsize 1024-2048 by fs, fo 71-800 Hz
let mut out = vec![0.0; input.len()];
reim.process_block(&input, &mut out);         // output slice is caller-owned; no allocation
```

Real-time: build `Reim` off the audio thread (construction allocates all working
buffers), then call `process_sample` in the callback — it allocates nothing,
takes no locks, and is deterministic.

```rust
// setup thread:
let mut reim = Reim::new(
    48_000.0, // fs
    5.0,      // frame period (ms)
    1024,     // fftsize (power of two): smaller = lower latency, coarser spectrum
    71.0,     // fo_floor (Hz)
    800.0,    // fo_ceil  (Hz)
);

// audio callback (hot path, allocation-free):
for (out_sample, &in_sample) in output.iter_mut().zip(input) {
    *out_sample = reim.process_sample(in_sample as f64) as f32;
}
```

### Reading the per-frame analysis parameters

For analysis without synthesis — real-time parameter extraction when you supply
your own synthesizer — use the standalone `Analyzer`. `push_sample` returns
`true` when a new analysis frame is ready; read the full WORLD parameter set
from the accessors. No synth runs.

```rust
use reim::Analyzer;
let mut a = Analyzer::with_defaults(48_000.0);
for &x in &input {
    if a.push_sample(x) {                 // true once per analysis frame
        let f0  = a.fo();                 // Hz, 0.0 = unvoiced
        let env = a.spectral_envelope();  // &[f64], len fftsize/2 + 1 (formants)
        let ap  = a.aperiodicity();       // &[f64], same length (D4C)
        let v   = a.voiced();
        // ... feed your own synthesis / feature pipeline
    }
}
```

(`Reim::analyze_sample` is the same analysis-only path on the fused type, kept
for callers already holding a `Reim`.)

### Manipulate then resynthesize

Pair `Analyzer` with the standalone `Synthesizer` to edit the parameters between
analysis and synthesis. A frequency-axis warp of the spectral envelope is a
formant shift; Fo and aperiodicity are untouched, so the pitch is preserved. See
`examples/formant_shift.rs` for the runnable version (`warp_formants` shown there).

```rust
use reim::{Analyzer, Synthesizer};
let mut a = Analyzer::with_defaults(fs);
let mut s = Synthesizer::with_defaults(fs);
let mut sp = vec![0.0; a.numbins()];
for &x in &input {
    if a.push_sample(x) {
        warp_formants(a.spectral_envelope(), 1.15, &mut sp); // +15% formant shift
        s.push_frame(a.fo(), a.voiced(), a.silence(), a.aperiodicity(), &sp);
    }
    out.push(s.next_sample());            // pitch preserved, formants shifted
}
```

`Reim::process_sample` remains the one-call analyze + resynthesize convenience;
it is now just `Analyzer` + `Synthesizer` composed.

Latency note: input->output latency is dominated by `fftsize`. Its floor is the
`fftsize/2`-sample synthesis group delay; measured end-to-end it is ≈49 ms at
48 kHz and ≈89 ms at 24 kHz with the default `fftsize=2048`. Because it is fixed
in samples, it falls as `fs` rises or `fftsize` shrinks — drop `fftsize` to cut
latency at the cost of frequency resolution. Throughput keeps up at ~14x real
time regardless (see Performance).

## Build and test

```
cargo build --release
cargo test --release
cargo clippy --all-targets --release   # clean (lints tests too)
```

`tests/snapshot.rs` freezes the analysis->synthesis waveform against a committed
golden (compared by SNR). The C oracle stays the reference for the analysis
decisions; this guards the synthesized output, which will diverge from the C once
aperiodicity becomes more than a placeholder. Regenerate after an intended change:
`REGEN_SNAPSHOT=1 cargo test --release --test snapshot`.

## Evaluation

Accuracy/fidelity scripts live in `eval/` (Python via `uv`, run from the repo root).
Each runs a synthetic self-test and falls back to it when the dataset is absent:

- `eval/pitch_rpa.py` -- sung-pitch accuracy (RPA/RCA, voicing P/R/F1) of `reim f0`
  against the [Vocadito](https://zenodo.org/records/5578807) ground truth, via `mir_eval`.
- `eval/roundtrip_quality.py` -- analysis->synthesis fidelity (mel-cepstral distortion
  and log-spectral distance, which are phase-invariant since the vocoder regenerates phase).

```
uv run --no-project --with numpy --with soundfile --with mir_eval python eval/pitch_rpa.py --self-test
```

## CLI

```
reim process <in.wav> <out.wav>           analyze + resynthesize a mono WAV
reim eval <ref.wav> <in.wav> [feat.csv]   compare output against a reference
reim bench [in.wav]                       throughput + per-stage latency
reim f0 <in.wav> [fmin] [fmax] [fftsize]  print per-frame "time,fo_hz" contour
reim ap <in.wav> <out.f64>                dump per-frame aperiodicity (raw f64)
```

Reads PCM8/PCM16/float32 mono WAV; writes float32 mono.

## Correctness

The reference C implementation is the oracle. `oracle.c` builds against the C
sources (it needs libsndfile) and emits a reference output WAV plus per-frame
features; `reim eval` then compares the Rust output sample-for-sample and checks
the silence/voiced/Fo decisions per frame.

```
cc -O2 -std=c99 -Ireim/include oracle.c reim/src/*.c \
   $(pkg-config --cflags --libs sndfile) -lm -o /tmp/oracle
/tmp/oracle in.wav /tmp/ref.wav /tmp/feat.csv /tmp/spap.csv
./target/release/reim eval /tmp/ref.wav in.wav /tmp/feat.csv
```

Measured against the C oracle, the Rust **analysis decisions** match it exactly:
100% per-frame agreement on silence, voiced/unvoiced, and Fo (0.0000% relative Fo
error) across the bundled 24 kHz voice file and synthetic 8/16/44.1 kHz signals
with vibrato, noise, and silence. The Fo and spectral-envelope paths reproduce the
C to the f32 storage noise floor (140-151 dB SNR).

The **synthesized waveform** no longer matches the C bit-for-bit: ReIm computes
real D4C band-aperiodicity where the C reference leaves aperiodicity a binary
placeholder, so the output diverges by design (the per-frame decisions above are
unaffected — voicing is decided before aperiodicity). `tests/snapshot.rs` guards
the synthesized output against a frozen reim golden instead of the C oracle.

## Pitch (Fo) accuracy

The Fo tracker is DIO zero-crossing candidates over a 2-channels/octave
filterbank, refined by instantaneous frequency (a StoneMask-equivalent step) and
selected by a Summation of Residual Harmonics (SRH) score, with the previous frame's Fo seeded for
continuity. Voicing is decided separately (in the Ap stage, via a D4C-style band
ratio), so this stage is a pitch tracker, not a pitch+VUV detector.

To check accuracy (not just faithfulness to the C), it was compared against two
classic dependency-free detectors -- YIN (de Cheveigne & Kawahara 2002) and
normalized autocorrelation -- on the same frame grid, using synthetic signals
with a known Fo. `gross` = error > 50 cents; `octave` = within 150 cents of an
octave; `median` = median absolute error.

| signal (known Fo)            | ReIm              | YIN               | autocorr        |
| ---------------------------- | ----------------- | ----------------- | --------------- |
| steady 200 Hz, ~3 dB SNR     | 0% gross / 0.8c   | 0% / 4.1c         | 0% / 3.7c       |
| steps 160/240 Hz + noise     | 17% gross, 0% oct | 36% gross, 20% oct| 37%, 21% oct    |
| weak fundamental 160 Hz      | 0% / 0.3c         | 0% / 0.3c         | 0% / 0.3c       |
| vibrato 190 Hz +-5% + noise  | 0% gross / 1.8c   | 0% / 13.3c        | 0% / 15.0c      |

ReIm matched or beat the YIN baseline on every case: 3-7x lower median error on
steady/vibrato tones (the instantaneous-frequency refinement gives finer
estimates than YIN's lag-quantized peak), and at noisy step transitions half the
gross errors and no octave errors where YIN/autocorr flip octaves ~20% of the
time. On the real voice file (no ground truth) ReIm and YIN agree to a median of
8.2 cents, 85% of frames within 50 cents.

Caveats: synthetic ground truth is controlled (laryngograph-referenced real
speech would be definitive); YIN/autocorr use standard untuned thresholds, so
this is "matches a standard baseline", not "beats a tuned YIN or a learned model
like CREPE". An offline/fixed-lag Viterbi pitch-smoothing pass was also tried and
measured no improvement over the greedy tracker on this material (the SRH score
plus the previous-Fo seed already handle octave errors and continuity), so it was
not added. Likewise, the instantaneous-frequency refinement is already
StoneMask-equivalent, and widening it (6/12 harmonics or a second pass) moved
median error by under 0.1 cent in mixed directions -- it is saturated at 3
harmonics, so it was left as is.

On real speech the picture is more sober. Run through the independent
[pitch-benchmark](https://github.com/lars76/pitch-benchmark) suite on its bundled
SpeechSynth set (219 TTS clips, 65-300 Hz, 16 kHz), using that project's own
algorithm wrappers and metrics, ReIm scores RPA 72% -- on par with WORLD's own
DIO/Harvest (72%/72%), behind pYIN (78%) and Praat (79%), and with a higher
gross/octave-error rate (6.0%/4.2% vs ~0.5-3%). About half that gap was the fixed
analysis window: 2048 points is 128 ms at 16 kHz, far too long for low-pitch
speech, and shrinking it to 1024 (64 ms) roughly halves gross/octave errors at no
RPA cost. `with_defaults` now picks the fftsize from the sample rate (1024 at
16 kHz, 2048 at 24-48 kHz), so this case is handled automatically; the synthetic
tones above (where it beat YIN) were too favorable. Learned models (CREPE,
SwiftF0, RMVPE) lead that benchmark and were not run here.

## Voicing

The voiced/unvoiced flag (`reim.last_voiced()`) is decided in the Ap stage,
independently of the pitch tracker, via a D4C-style high/low band-energy ratio.
On top of the faithful C decision, ReIm adds a sub-fundamental guard: a frame is
rejected as unvoiced when energy below ~0.8·Fo (rumble, mains hum, HVAC)
dominates the fundamental-plus-harmonic band. The C's "LoveTrain" only suppresses
*high*-band noise, so low-frequency rumble sitting inside `[fo_floor, fo_ceil]`
would otherwise read as voiced.

On synthetic probes the guard cuts broadband rumble from 99% to 15% voiced and
mains hum to 0%, at no cost to clean voice (vibrato/breathy stay 100%). On real
solo singing ([Vocadito](https://zenodo.org/records/5578807), 40 clips) voicing
F1 rises 83.5 -> 86.2 (precision up, recall flat) with RPA unchanged; on 16 kHz
TTS speech it is roughly neutral (RPA -1.3). The guard surfaces only through
`last_voiced()`; its threshold is an internal eval-chosen constant. It does not
fire on clean speech, so it leaves the oracle agreement above unchanged.

ReIm also ships an **optional, experimental periodicity gate, off by default**.
The Fo tracker returns a best candidate even on breath or noise, so a fo-in-range
frame is not necessarily voiced; true voiced frames carry an SRH harmonic score
orders of magnitude higher. Enabling the gate — `Reim::set_voicing_score_min(x)`,
or the `REIM_VOICING_SCORE_MIN` env var for the CLI (≈1e3 on Vocadito) — requires
that score to clear a threshold. On Vocadito (40 clips) it cuts the voicing
false-alarm rate from 0.67 to 0.19 and lifts voicing F1 from 0.85 to 0.93.
**Tradeoff:** it also rejects ~5% of true-voiced frames — mostly soft/decaying
ones and note edges — which then resynthesize as noise instead of pitched pulses
(audibly clipping soft note tails). The score is level-independent (a power
ratio), but the threshold is eval-chosen on a single dataset and **needs more
testing before it could become a default**, hence opt-in. With the default (0.0)
the gate is inert and voicing matches the faithful C decision.

The threshold is the recall/precision dial (Vocadito, 40 clips):

| `score_min` | voicing FA | recall | F1   |                                   |
| ----------- | ---------- | ------ | ---- | --------------------------------- |
| 0           | 0.67       | 0.997  | 0.85 | off (default; C-faithful voicing) |
| 1000        | 0.19       | 0.946  | 0.93 | balanced — F1 optimum / the knee  |
| 1e4         | 0.13       | 0.909  | 0.92 | precision-leaning                 |

1000 is the knee: it removes ~72% of the false alarms for ~5% recall. Lower it
(toward ~400-600) to lean back toward recall — fewer clipped soft tails, more
false alarms; raise it for fewer false alarms at more recall cost. Calibrated on
singing, so another corpus may shift the optimum. (Hysteresis on the gate was
tried and gave no better recall/precision tradeoff than this single threshold.)

`reim f0` reports pitch only on frames this decision marks voiced, so the printed
contour reflects the full voicing logic, not just the raw tracker.

## Performance

`reim bench` on the bundled voice file (24 kHz): ~14x real time, per-frame
analysis p99 ~375 us against a 5 ms frame budget (~13x headroom). Fo analysis
dominates (~52% of frame work). The per-sample synthesis path is the only work
done outside frame boundaries.

## Terminology

The three analysis parameters: **Fo** fundamental frequency (pitch), **Sp**
spectral envelope (formants), **Ap** aperiodicity (per-bin noise-to-total ratio).

Metrics (used in Pitch accuracy / Voicing / Evaluation):

- **cent** — 1/100 of a semitone (1200 cents per octave); the unit for pitch error.
- **precision / recall** (voicing) — of the frames *called* voiced, the fraction
  truly voiced (precision); of the *truly* voiced frames, the fraction called
  voiced (recall).
- **F1** — the harmonic mean of precision and recall, `2·P·R/(P+R)`: one 0-1 score
  that is high only when *both* are high. "Voicing F1" applies it to the per-frame
  voiced/unvoiced decision.
- **voicing false alarm (VFA)** — of the truly *unvoiced* frames, the fraction
  wrongly called voiced (over-voicing). Lower is better.
- **RPA** (raw pitch accuracy) — fraction of voiced frames whose estimated Fo is
  within 50 cents of ground truth. **RCA** (raw chroma accuracy) — the same but
  octave-invariant; RPA << RCA means the tracker has the right pitch class in the
  wrong octave (octave errors).
- **gross / octave error** — gross = Fo off by > 50 cents; octave = off by within
  50 cents of a 2x/0.5x of the true Fo.
- **MCD / LSD** — mel-cepstral distortion / log-spectral distance: phase-invariant
  spectral distances (dB) between input and resynthesis (lower = closer).
- **SNR** — signal-to-noise ratio (dB): reference energy over error energy.

Methods (the C reference is a WORLD-like vocoder; these are its stages):

- **WORLD** — the analysis/synthesis vocoder family ReIm follows.
- **DIO** — zero-crossing-based Fo *candidate* generator; **SRH** (summation of
  residual harmonics) scores the candidates to pick Fo; **StoneMask** is the
  instantaneous-frequency Fo refinement.
- **CheapTrick** — the spectral-envelope estimator. **D4C** — the band-aperiodicity
  estimator; its **LoveTrain** step is a high/low band-energy voiced/unvoiced test.
- **velvet noise** — sparse signed impulses used as the aperiodic excitation.
