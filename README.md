# reim-rs

A single-file Rust port of [ReIm](./reim), a real-time WORLD-like speech
vocoder. Analyzes a mono signal into fundamental frequency (Fo), aperiodicity
(Ap), and spectral envelope (Sp), then resynthesizes it. The analysis order is
Silence -> Fo -> Ap -> Sp.

The entire library and CLI live in `src/main.rs` with no external dependencies;
it builds with `cargo` or directly with `rustc -O src/main.rs`.

## Design

- Allocation-free steady state: `Reim::process_sample` does no heap allocation;
  all working buffers are owned by the analyzer/synthesizer state and reused.
- Self-contained: hand-written radix-2 FFT, xorshift RNG, and WAV I/O.
- Faithful to the C reference: ported function by function. Two deliberate
  deviations from the C are documented in the source — the silence RMS skips a
  one-element out-of-bounds read present in the C, and the Ap range guard mirrors
  the C's exact branch semantics (including NaN handling).

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

## Performance

`reim bench` on the bundled voice file (24 kHz): ~14x real time, per-frame
analysis p99 ~375 us against a 5 ms frame budget (~13x headroom). Fo analysis
dominates (~52% of frame work). The per-sample synthesis path is the only work
done outside frame boundaries.
