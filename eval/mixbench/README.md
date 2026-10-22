# mixbench: target pitch under noise and competing talkers

mixbench measures how well a pitch tracker follows one target voice when real
background noise or other talkers are present. A seeded manifest defines every
mixture, and SHA-256 fingerprints let a third party rebuild the same audio and
score any tracker with the same rules.

Terms:

- **Target**: the talker whose pitch the tracker must report. The target is
  the clearest voice over time: it is at least as loud as each other talker
  (SIR >= 0 dB). It speaks first in talker mixtures and second in ahead and
  babble mixtures.
- **Interferer**: any other talker in the mixture.
- **SNR / SIR**: signal-to-noise and signal-to-interferer ratio, in dB.
- **Active span**: the samples from 16 ms before the first voiced reference
  frame to 16 ms after the last one (half of the 32 ms RAPT window).
- **Reference grid**: every 10 ms frame of the target's reference pitch file
  (silent frames included), as listed in the target track of `.refs.json`.
- **Capture**: the tracker reports an interferer's pitch instead of the
  target's.

## Explanation

### Why this benchmark

The earlier evaluation in `eval/pitch_rpa.py` adds white noise and mains hum.
Those signals do not contain other voices, and they are stationary. mixbench
uses recorded environments (cafeteria, metro, street, kitchen and others) and
real competing speech. Each interferer has its own laryngograph reference, so
the scorer can tell a wrong pitch from the pitch of the wrong person.

### Sources

| corpus | content | ground truth | license |
| --- | --- | --- | --- |
| [PTDB-TUG](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html) | 4718 read sentences, 20 speakers (10 F, 10 M), 48 kHz | RAPT on the laryngograph signal, 10 ms | ODbL 1.0 / DbCL 1.0 |
| [DEMAND](https://zenodo.org/records/1227121) | 18 recorded noise environments, 16 mics, 48 kHz, 300 s each | none (noise only) | CC BY 4.0 (Zenodo metadata) or CC BY-SA 3.0 (description) |

The official PTDB-TUG server did not answer during construction. `fetch.py`
uses the Hugging Face mirror `gilkeyio/ptdb-tug` at a fixed revision. The
mirror stores the original WAV bytes. The manifest records the SHA-256 of
every WAV file that a mixture uses, so a copy from the official archive
must produce the same hashes. mixbench uses DEMAND channel 1 only.

### Splits

The two splits share no speaker and no noise environment.

| split | target and interferer speakers | DEMAND environments | targets | mixtures |
| --- | --- | --- | --- | --- |
| dev | F01-F04, M01-M04 | DKITCHEN, NPARK, OOFFICE, PSTATION, STRAFFIC, TBUS | 40 | 440 |
| test | F05-F10, M05-M10 | DLIVING, DWASHING, NFIELD, NRIVER, OHALLWAY, OMEETING, PCAFETER, PRESTO, SCAFE, SPSQUARE, TCAR, TMETRO | 84 | 924 |

Use dev for every design choice and parameter. Score test once, after the
method is frozen. `tune_lock.py` reads dev predictions only.

## Reference

### Mixture rules (`build.py`)

The random generator is `numpy.random.default_rng(20251021)`. One target
utterance produces eleven mixtures:

| condition | content | who speaks first | level |
| --- | --- | --- | --- |
| clean | target only | - | - |
| noise_snr10, noise_snr5, noise_snr0 | target + one DEMAND segment | - | SNR 10, 5, 0 dB |
| talker_sir10, talker_sir5, talker_sir0 | target + one interferer | target | SIR 10, 5, 0 dB |
| ahead_sir10, ahead_sir5 | target + the same interferer | interferer | SIR 10, 5 dB |
| babble_sir5, babble_sir0 | target + three interferers | interferers | SIR 5, 0 dB (all three together) |

There is no SIR below 0 dB: there the interferer is the clearest voice, so
the clearest-voice rule makes it the target.

- **Targets**: 5 (dev) or 7 (test) utterances per speaker. An utterance is
  eligible when its active span is at least 2.5 s, its file is at most 8 s, and
  it has at least 100 voiced reference frames. The target keeps gain 1.
- **Target window**: the target's active span plus 0.25 s each side (cut at
  the file ends). In clean, noise and talker mixtures the window starts at
  sample 0 and the mixture ends with the window. In ahead and babble
  mixtures the interferers start at sample 0 and the window starts 0.5-1.0 s
  later.
- **Noise**: one environment and start sample per target, used for all three
  SNRs. SNR = target active-span power / noise-segment power.
- **Interferers**: the single interferer has the same sex as the target for
  even target indices and the other sex for odd ones (126 of each per talker
  condition in test). The ahead mixtures use the same speaker with a
  different chain of utterances.
  The three babble talkers are three other speakers, never the single
  interferer. Each interferer is a chain of its utterances, cut to their
  active spans, never the target's sentence, separated by 0.1-0.4 s pauses.
  In talker mixtures the chain starts 0.5-1.0 s after the start of the
  target's active span. In ahead and babble mixtures it starts at sample 0.
  It stops when the next utterance would start at or after the mixture end,
  and the end cuts the last utterance. Each talker gets the
  gain that makes its power over its chain equal to target active power /
  (talkers x 10^(SIR/10)).
- **Sum order**: target, then interferer segments in manifest order, then
  noise, all in float64. If the peak exceeds 0.99, `out_gain` scales the sum
  to 0.99 (1 of 924 test mixtures). The WAV holds the float32 result.
- **Fingerprint**: `sha256` in the manifest is the SHA-256 of the float32
  little-endian samples. The manifest stores every gain, so the render step
  does not compute a level. `ref_sha256` is the SHA-256 of a source's
  reference F0 column as float64 little-endian, which a parse of the official
  `.f0` file must reproduce. Do not hash the WAV files: libsndfile writes a
  time stamp into each file's PEAK chunk, so the file bytes change on every
  render.

### Reference tracks

`build.py render` writes `<name>.refs.json` next to each WAV. Reference frame
`i` of a source file sits at `0.0195 + 0.010 * i` s. Each segment (target or
interferer) shifts its frames by `(dst_start - src_start) / 48000` s and keeps
only frames whose time lies inside the segment. `f0 > 0` marks a voiced frame.

The PTDB-TUG thesis (Pirker 2012, section 6.2) gives the RAPT window (32 ms)
and hop (10 ms) but not the time of frame 0. `ref_offset.py` tracks pitch on
the raw laryngograph signal of 231 dev utterances with Praat at a 0.5 ms
step. Relative to `0.016 + 0.010 * i`, the best shift has a median of +3.5 ms
(IQR 2.5-4.5 ms), so frame 0 sits at 19.5 ms. `score.py --ref-shift-ms`
measures how much a result depends on this offset.

### Prediction format

One CSV per mixture, `<name>.csv`, rows `time_seconds,f0_hz`, any hop, any
row order. The scorer skips every row that does not parse as two numbers.
`f0 <= 0` (or NaN replaced by 0) means unvoiced.

### Metrics (`score.py`)

The scorer evaluates each clip on the target reference grid:

1. Round all times to whole microseconds. For each grid time, take the
   prediction frame nearest in time. A tie goes to the earlier frame. If rows
   share a time stamp, the last row counts. If the nearest frame is more than
   10 ms away, the estimate is unvoiced at that time.
2. For each interferer track, take its nearest frame within 5 ms, else
   unvoiced.
3. A frame is a **hit** when the target and the estimate are voiced and
   `|1200 log2(est / target)| <= 50` cents.

With T = target voiced, E = estimate voiced, I = any interferer voiced:

| metric | definition |
| --- | --- |
| `f1` | 2PR / (P + R) with P = `precision`, R = `rpa`; 0 when there is no hit |
| `precision` | hits / E |
| `rpa` | hits / T (raw pitch accuracy) |
| `rpa_overlap` | hits in I / T in I |
| `oa` | (hits + frames with neither T nor E) / all frames |
| `vr` | (T and E) / T |
| `vfa` | (E and not T) / not T |
| `ifa` | (E and not T and I) / (not T and I): voiced reports while only an interferer speaks |
| `gpe` | (T and E and not hit) / (T and E) |
| `capture` | frames with E, no hit, and the estimate within 50 cents of an interferer voiced at that frame (target voiced or not) / I |
| `fpe` | median absolute cents error over hits |

A ratio with a zero denominator is NaN and drops out of the means.
`f1` is the headline metric: it charges a missed target frame and a reported
wrong frame equally, and it ignores frames where both the target and the
estimate are unvoiced. It still depends on the number of interferer-only
frames, because each report on such a frame lowers precision. `ifa` and
`rpa_overlap` give the two regions separately.

### Methods (`run.py`)

Every method searches 71-800 Hz. "Live" means that the method can run on a
stream. The delay column gives the time after a frame's time stamp at which
its value is known.

| method | mode | delay | what it runs |
| --- | --- | --- | --- |
| reim-live | live | 21 ms | `reim f0` with `REIM_VOICING_REFINE=0 REIM_PITCH_REFINE=0` (half of the 2048-point window) |
| reim-live-gate | live | 21 ms | reim-live with `REIM_VOICING_SCORE_MIN=0.6` |
| reim-offline | offline | file | `reim f0` defaults (Viterbi voicing and pitch refinement) |
| praat | offline | file | Praat `to_pitch_ac` (parselmouth 0.4.7), 10 ms step, Viterbi path |
| pyin | offline | file | librosa 1.0.0 pYIN at 16 kHz, 1024-point frame, 10 ms hop |
| swiftf0 | live | 176 ms (its README) | SwiftF0 0.3.0, voiced when confidence >= 0.5 |
| dfn3+reim-live | live | DFN3 + 21 ms | DeepFilterNet3 (`deep-filter` 0.5.6 `-D`, lookahead 2 frames) then reim-live |
| dfn3+reim-live-gate | live | DFN3 + 21 ms | DeepFilterNet3 then reim-live-gate |
| rnnoise+reim-live | live | 10 ms + 21 ms | RNNoise, xiph main 70f1d25, 16-bit, its 480-sample delay removed, then reim-live |
| wiener+reim-live | offline | file | `reim enhance` from branch voice-enhancement 19ace40 (two-pass Wiener) then reim-live |
| `<base>+lock` | as base | base, at least 15 ms | causal target lock (`lock.py`) on reim-live-gate, dfn3+reim-live-gate or swiftf0 |
| `<base>+lock-offline` | offline | file | offline target lock on the same bases |

The target lock is a prototype of the clearest-voice rule. The target level
is the 90th percentile of the voiced-frame levels (30 ms window) over the
last 2 s, and the target pitch is the median pitch of the frames within 3 dB
of that level. A voiced frame starts or joins the target when its level is
within `level_db` of the target level and its pitch is within `range_cents`
of the target pitch. An accepted run continues while the pitch moves at most
100 cents per frame. The offline form takes both statistics from the whole
file and accepts or rejects each voiced run. `tune_lock.py` chose
`level_db` and `range_cents` per base method by mean dev F1, and reports a
leave-one-speaker-out estimate; `run.py` `LOCKS` holds the values.

Known advantages and quirks:

- The fused voicing weights in `src/lib.rs` (used by reim-live-gate,
  reim-offline and every chain built on them) were fit on Vocadito and a
  PTDB-TUG subsample that includes test speakers. Treat reim gate results on
  test as optimistic.
- The SwiftF0 training data is not stated here; it can include PTDB-TUG.
- pYIN reports many voiced frames at the 71 Hz floor while the target is
  silent (14.9% of its voiced dev estimates in the first dev build).

Tools in `$REIM_BENCH_DATA/tools` (SHA-256):

| file | source | sha256 |
| --- | --- | --- |
| deep-filter-0.5.6 | GitHub release `deep-filter-0.5.6-aarch64-apple-darwin` | `4601e7f4e4c03e59a4c5b5000216ef3add3e808799cfccd95e14e83ea4611081` |
| rnnoise_demo-70f1d25 | `gitlab.xiph.org/xiph/rnnoise` main 70f1d25, `./autogen.sh && ./configure --disable-shared && make` (model `rnnoise_data-0a8755f8...`) | `0eef040df9b9408710c9db670cbbe0d2751198502bb3408fccd8e5ce878b245c` |
| reim-enhance-19ace40 | `git archive voice-enhancement`, `cargo build --release` | `48eeb7d36de1aa26a9268c2cc3b90bd52ccd75db2a2267aca079bf9f60d18294` |

The `reim` binary is `target/release/reim` of the commit in each
`.run.json`. Native binaries are arm64 macOS builds. Another platform needs
its own build and can give a different prediction digest.

### Summary statistics

The speakers are the independent units. `score.py --compare` averages the
clips of each target speaker, then gives the mean of the speaker means with a
95% t interval (8 dev or 12 test speakers, df = speakers - 1). A paired delta
joins two methods on the mixture id and applies the same t interval to the
per-speaker mean differences.

`score.py --primary` runs the four comparisons in `PRIMARY` on F1 for each
condition group (clean, noise, talker, ahead, babble): 20 paired t tests,
with Holm-adjusted p values. The list was fixed before the test split was
scored.

## How-to

### Rebuild and verify the audio

```
python3 eval/mixbench/fetch.py --jobs 8          # 11.5 GB, size + SHA-256/MD5 checked
R="uv run --no-project --with-requirements eval/mixbench/requirements.txt python"
$R eval/mixbench/build.py extract
$R eval/mixbench/build.py render                 # checks every source and mixture hash
```

`render` must print `all fingerprints match`. A source mismatch means a
different input file. A mixture mismatch with matching sources means a
different float64 sum; report the Python and NumPy versions.

Do not run `build.py manifest` to verify. It rewrites the benchmark
definition.

### Score a tracker or a device

1. Run the tracker on `$REIM_BENCH_DATA/mix/<split>/*.wav`.
2. Write the predictions to `$REIM_BENCH_DATA/preds/<split>/<method>/`.
3. Score them and compare against the committed results:

```
$R eval/mixbench/score.py --split test <method>
$R eval/mixbench/score.py --split test --compare reim-live-gate <method>
$R eval/mixbench/score.py --split test --ref-shift-ms 5 <method>   # offset sensitivity
```

The scorer needs only the WAV names, the `.refs.json` files, and the CSVs, so
the tracker can run on any device. `score.py --self-test` checks every metric
on synthetic tracks with known answers.

### Reproduce a committed run

```
M="uv run --no-project --with-requirements eval/mixbench/requirements-methods.txt python"
$M eval/mixbench/run.py --split test <method>
```

`results/<split>/<method>.run.json` records the tool hashes, package versions,
platform, real-time factor, and `predictions_sha256`, a digest over every
prediction file. A deterministic method on the same tools must reproduce the
digest. `run.py` expects the tools under `$REIM_BENCH_DATA/tools` (see the
method table for the exact builds).
