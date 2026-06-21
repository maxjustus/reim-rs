# reim-rs

A Rust port of [ReIm](./reim), a real-time WORLD-like speech vocoder. Analyzes a
mono signal into fundamental frequency (Fo), aperiodicity (Ap), and spectral
envelope (Sp), then resynthesizes it. The analysis order is Silence -> Fo -> Ap
-> Sp.

The vocoder is a dependency-free library in `src/lib.rs`; `src/main.rs` is a thin
CLI over it. Add the crate as a dependency and `use reim::Reim`, or run the
`reim` binary directly.

## Design

- Allocation-free steady state: `Reim::process_sample` does no heap allocation;
  all working buffers are owned by the analyzer/synthesizer state and reused.
- Self-contained: hand-written radix-2 FFT, xorshift RNG, and WAV I/O.
- Faithful to the C reference: ported function by function. Two deliberate
  deviations from the C are documented in the source — the silence RMS skips a
  one-element out-of-bounds read present in the C, and the Ap range guard mirrors
  the C's exact branch semantics (including NaN handling).

## Use as a library

The whole API is the `Reim` type: construct it once, then push samples through
`process_sample` (one in, one out) or `process_block`.

```rust
// Offline / block: analyze + resynthesize a buffer.
let mut reim = Reim::with_defaults(48_000.0); // fs; period 5 ms, fftsize 2048, fo 71-800 Hz
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
cargo clippy --release        # clean
```

## CLI

```
reim process <in.wav> <out.wav>           analyze + resynthesize a mono WAV
reim eval <ref.wav> <in.wav> [feat.csv]   compare output against a reference
reim bench [in.wav]                       throughput + per-stage latency
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

Measured against the C oracle, the Rust output matches to the f32 storage noise
floor (140-151 dB SNR) with 100% per-frame agreement on silence, voiced/unvoiced,
and Fo (0.0000% relative Fo error), across the bundled 24 kHz voice file and
synthetic 8/16/44.1 kHz signals with vibrato, noise, and silence. The remaining
error is the f32 quantization of the reference WAV; the f64 pipelines agree below
that.

## Pitch (Fo) accuracy

The Fo tracker is DIO zero-crossing candidates over a 2-channels/octave
filterbank, refined by instantaneous frequency and selected by a Summation of
Residual Harmonics (SRH) score, with the previous frame's Fo seeded for
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
not added.

## Performance

`reim bench` on the bundled voice file (24 kHz): ~14x real time, per-frame
analysis p99 ~375 us against a 5 ms frame budget (~13x headroom). Fo analysis
dominates (~52% of frame work). The per-sample synthesis path is the only work
done outside frame boundaries.
