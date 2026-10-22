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
- **Reference grid**: the target's 10 ms reference frames from time 0 to
  the end of the mixture, as listed in the target track of `.refs.json`.
  Frames before the target window are unvoiced.
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
| dev | F01-F04, M01-M04 | DKITCHEN, NPARK, OOFFICE, PSTATION, STRAFFIC, TBUS | 40 | 400 |
| test | F05-F10, M05-M10 | DLIVING, DWASHING, NFIELD, NRIVER, OHALLWAY, OMEETING, PCAFETER, PRESTO, SCAFE, SPSQUARE, TCAR, TMETRO | 84 | 840 |

Use dev for every design choice and parameter. Score test once, after the
method is frozen. `tune_lock.py` reads dev predictions only.

## Reference

### Mixture rules (`build.py`)

The random generator is `numpy.random.default_rng(20251021)`. One target
utterance produces ten mixtures:

| condition | content | who speaks first | level |
| --- | --- | --- | --- |
| clean | target only | - | - |
| noise_snr10, noise_snr5, noise_snr0 | target + one DEMAND segment | - | SNR 10, 5, 0 dB |
| talker_sir10, talker_sir5 | target + one interferer | target | SIR 10, 5 dB |
| ahead_sir10, ahead_sir5 | target + the same interferer | interferer | SIR 10, 5 dB |
| babble_sir5, babble_sir0 | target + three interferers | interferers | SIR 5, 0 dB (all three together) |

There is no single-talker mixture at 0 dB or below. At 0 dB neither voice
is clearer, so only the speaking order would name the target; below 0 dB the
clearest-voice rule makes the interferer the target. In babble_sir0 each of
the three talkers is 4.8 dB below the target.

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
  even target indices and the other sex for odd ones (42 of each per talker
  or ahead condition in test). The ahead mixtures use the same speaker with a
  separately drawn chain, which can repeat an utterance of the talker chain.
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
  to 0.99 (2 of 400 dev mixtures, 0 test). The WAV holds the float32 result.
- **Fingerprint**: `sha256` in the manifest is the SHA-256 of the float32
  little-endian samples. The manifest stores every gain, so the render step
  does not compute a level. `ref_sha256` is the SHA-256 of a source's
  reference F0 column as float64 little-endian, which a parse of the official
  `.f0` file must reproduce. Do not hash the WAV files: libsndfile writes a
  time stamp into each file's PEAK chunk, so the file bytes change on every
  render.

### Reference tracks

`build.py render` writes `<name>.refs.json` next to each WAV. Reference frame
`i` of a source file sits at `0.0215 + 0.010 * i` s. Each segment (target or
interferer) shifts its frames by `(dst_start - src_start) / 48000` s and keeps
only frames whose time lies inside the segment. The target track then gets
unvoiced frames on the same 10 ms grid back to time 0. `f0 > 0` marks a
voiced frame.

The PTDB-TUG thesis (Pirker 2012, section 6.2) gives the RAPT window (32 ms)
and hop (10 ms) but not the time of frame 0. `ref_offset.py` tracks pitch with
Praat autocorrelation at a 0.5 ms step on 309 dev utterances. Relative to
`0.016 + 0.010 * i`, the reference matches the laryngograph best at a median
shift of +4.0 ms (IQR 2.5-5.0), and the microphone track lags the
laryngograph track by a median of +1.5 ms (IQR 0.5-2.5), the sound travel to
the microphone. Frame 0 therefore sits at 21.5 ms in microphone time. An
independent audit found +4.25 ms for the laryngograph with a tracker-free
glottal-closure method. Praat cross-correlation gives +8.5 ms and is not
used. `score.py --ref-shift-ms` measures how much a result depends on this
offset.

### Prediction format

One CSV per mixture, `<name>.csv`, rows `time_seconds,f0_hz`, any hop, any
row order. The scorer reads the first two fields of each row and skips a
row whose first two fields do not parse as numbers.
`f0 <= 0` (or NaN replaced by 0) means unvoiced.

### Metrics (`score.py`)

The scorer evaluates each clip on the target reference grid:

1. Round all times (grid, prediction, interferer) to whole microseconds.
   For each grid time, take the prediction frame nearest in time. A tie goes
   to the earlier frame. If rows share a time stamp, the last row counts. If
   the nearest frame is more than 10 ms away (exactly 10 ms counts as
   within), the estimate is unvoiced at that time.
2. For each interferer track, take its nearest frame by the same rule with a
   5 ms limit, else unvoiced.
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
leave-one-speaker-out estimate; `run.py` `LOCKS` holds the values. The other
lock constants (2 s history, 3 dB clear band, 100 cents continuity) were set
by hand after dev experiments, outside that loop, so the test split is the
only clean estimate of the lock. On dev the lock raises F1 only by staying
silent while an interferer speaks alone: it lowers `rpa` on every condition.

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
clips of each target speaker that have a value, then gives the mean of the
speaker means with a 95% t interval (df = number of such speakers - 1; 8 on
dev, 12 on test). If all speaker means are equal, the interval is that value.
A paired delta joins two methods on the mixture id, takes the per-clip
difference (method minus baseline), and applies the same procedure to those
differences. Each test is two-sided.

`score.py --primary` runs the four comparisons in `PRIMARY` on F1 for each
condition group (clean, noise, talker, ahead, babble). A speaker's value in
a group is the mean of all its clips in that group. That gives 20 paired
two-sided t tests, with Holm-adjusted p values. Each row also gives the
change in `rpa` and `ifa`, which shows whether an F1 change comes from the
target's pitch or from silence while only an interferer speaks. The list was fixed before the test split was
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

## Results (test split)

All 16 methods ran on the 840 test mixtures from commit `734e6d0` with a
clean tree, and the test split was scored once after the dev work ended.
`results/test/` holds the per-clip scores and the run records.

### Target-pitch F1 by condition

Mean of the 12 speaker means. rnnoise+reim-live and wiener+reim-live feed
reim-live, which has no voicing gate; compare them with reim-live (0.616
clean, 0.303 at noise_snr0), not with reim-live-gate.

| condition | reim-live-gate | praat | pyin | swiftf0 | rnnoise+reim-live | wiener+reim-live | dfn3+reim-live-gate | dfn3+reim-live-gate+lock | dfn3+reim-live-gate+lock-offline | swiftf0+lock-offline |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.763 | 0.795 | 0.688 | 0.818 | 0.632 | 0.621 | 0.767 | 0.761 | 0.754 | 0.756 |
| noise_snr10 | 0.669 | 0.706 | 0.623 | 0.778 | 0.600 | 0.607 | 0.761 | 0.760 | 0.759 | 0.746 |
| noise_snr5 | 0.541 | 0.634 | 0.546 | 0.758 | 0.565 | 0.560 | 0.752 | 0.753 | 0.753 | 0.743 |
| noise_snr0 | 0.321 | 0.517 | 0.348 | 0.727 | 0.493 | 0.449 | 0.724 | 0.724 | 0.726 | 0.717 |
| talker_sir10 | 0.636 | 0.657 | 0.592 | 0.674 | 0.559 | 0.533 | 0.698 | 0.713 | 0.724 | 0.715 |
| talker_sir5 | 0.561 | 0.589 | 0.520 | 0.618 | 0.500 | 0.484 | 0.627 | 0.632 | 0.644 | 0.648 |
| ahead_sir10 | 0.546 | 0.566 | 0.508 | 0.577 | 0.458 | 0.464 | 0.597 | 0.612 | 0.688 | 0.708 |
| ahead_sir5 | 0.474 | 0.487 | 0.450 | 0.515 | 0.412 | 0.412 | 0.524 | 0.532 | 0.556 | 0.575 |
| babble_sir5 | 0.475 | 0.451 | 0.467 | 0.477 | 0.396 | 0.404 | 0.575 | 0.579 | 0.621 | 0.595 |
| babble_sir0 | 0.329 | 0.338 | 0.339 | 0.380 | 0.306 | 0.303 | 0.419 | 0.421 | 0.427 | 0.405 |

### Primary comparisons

The four comparisons in `PRIMARY`, fixed before test scoring. Holm adjusts
over all 20 rows.

| comparison | group | Δrpa | Δifa | ΔF1 | 95% CI | p | Holm p |
|---|---|---|---|---|---|---|---|
| DeepFilterNet3 front end for live reim: dfn3+reim-live-gate vs reim-live-gate | clean | +0.002 | - | +0.004 | [+0.001, +0.008] | 0.0264 | 0.2111 |
| DeepFilterNet3 front end for live reim: dfn3+reim-live-gate vs reim-live-gate | noise | +0.316 | - | +0.235 | [+0.178, +0.292] | 0.0000 | 0.0000 |
| DeepFilterNet3 front end for live reim: dfn3+reim-live-gate vs reim-live-gate | talker | +0.052 | -0.177 | +0.064 | [+0.029, +0.100] | 0.0022 | 0.0236 |
| DeepFilterNet3 front end for live reim: dfn3+reim-live-gate vs reim-live-gate | ahead | +0.051 | -0.095 | +0.050 | [+0.027, +0.074] | 0.0007 | 0.0085 |
| DeepFilterNet3 front end for live reim: dfn3+reim-live-gate vs reim-live-gate | babble | +0.113 | -0.061 | +0.094 | [+0.055, +0.134] | 0.0003 | 0.0040 |
| causal clearest-voice lock: dfn3+reim-live-gate+lock vs dfn3+reim-live-gate | clean | -0.028 | - | -0.006 | [-0.017, +0.006] | 0.2851 | 1.0000 |
| causal clearest-voice lock: dfn3+reim-live-gate+lock vs dfn3+reim-live-gate | noise | -0.009 | - | -0.000 | [-0.005, +0.004] | 0.9248 | 1.0000 |
| causal clearest-voice lock: dfn3+reim-live-gate+lock vs dfn3+reim-live-gate | talker | -0.015 | -0.100 | +0.010 | [+0.005, +0.014] | 0.0012 | 0.0148 |
| causal clearest-voice lock: dfn3+reim-live-gate+lock vs dfn3+reim-live-gate | ahead | -0.008 | -0.054 | +0.012 | [+0.008, +0.015] | 0.0000 | 0.0001 |
| causal clearest-voice lock: dfn3+reim-live-gate+lock vs dfn3+reim-live-gate | babble | -0.002 | -0.013 | +0.003 | [+0.002, +0.004] | 0.0001 | 0.0016 |
| offline lock over causal lock: dfn3+reim-live-gate+lock-offline vs dfn3+reim-live-gate+lock | clean | -0.012 | - | -0.007 | [-0.015, +0.001] | 0.0738 | 0.5165 |
| offline lock over causal lock: dfn3+reim-live-gate+lock-offline vs dfn3+reim-live-gate+lock | noise | -0.007 | - | +0.000 | [-0.004, +0.005] | 0.8786 | 1.0000 |
| offline lock over causal lock: dfn3+reim-live-gate+lock-offline vs dfn3+reim-live-gate+lock | talker | -0.011 | -0.099 | +0.012 | [+0.003, +0.022] | 0.0172 | 0.1547 |
| offline lock over causal lock: dfn3+reim-live-gate+lock-offline vs dfn3+reim-live-gate+lock | ahead | -0.015 | -0.212 | +0.050 | [+0.035, +0.064] | 0.0000 | 0.0002 |
| offline lock over causal lock: dfn3+reim-live-gate+lock-offline vs dfn3+reim-live-gate+lock | babble | -0.013 | -0.083 | +0.024 | [+0.014, +0.034] | 0.0002 | 0.0033 |
| SwiftF0 over the best live reim chain: swiftf0 vs dfn3+reim-live-gate | clean | +0.074 | - | +0.051 | [+0.023, +0.079] | 0.0021 | 0.0236 |
| SwiftF0 over the best live reim chain: swiftf0 vs dfn3+reim-live-gate | noise | +0.026 | - | +0.009 | [-0.025, +0.043] | 0.5792 | 1.0000 |
| SwiftF0 over the best live reim chain: swiftf0 vs dfn3+reim-live-gate | talker | +0.072 | +0.336 | -0.017 | [-0.049, +0.016] | 0.2822 | 1.0000 |
| SwiftF0 over the best live reim chain: swiftf0 vs dfn3+reim-live-gate | ahead | +0.072 | +0.225 | -0.014 | [-0.051, +0.023] | 0.4218 | 1.0000 |
| SwiftF0 over the best live reim chain: swiftf0 vs dfn3+reim-live-gate | babble | +0.070 | +0.393 | -0.068 | [-0.094, -0.043] | 0.0001 | 0.0016 |

### Dependence on the reference offset

`--ref-shift-ms` at -5, +2 and +5 ms leaves every row of the first three
comparisons within 0.011 F1 of the table above, with the same Holm verdict at
p < 0.05. The SwiftF0 rows depend on the offset: the clean delta is +0.051 at
0 ms, +0.050 at +2 ms and +0.014 at +5 ms (Holm p 1.0). The babble delta stays
between -0.063 and -0.069 with Holm p below 0.005. `results/test/shift*ms/`
holds those scores.

### Findings

- DeepFilterNet3 before reim-live-gate removes most of the noise loss. F1 at
  noise_snr0 rises from 0.321 to 0.724, and the clean F1 does not drop. It
  also helps with other talkers: +0.050 to +0.094 F1 per group, with higher
  `rpa` and lower `ifa`. On sung input it also removes voice, see
  "DeepFilterNet3 on sung mm1-rs recordings".
- RNNoise and the two-pass Wiener filter help less: at noise_snr0 they raise
  reim-live from 0.303 to 0.493 and 0.449.
- SwiftF0 alone handles noise as well as the DeepFilterNet3 chain, and it
  has the highest `rpa` in 7 of 10 conditions (dfn3+reim-live is higher at
  noise_snr5, noise_snr0 and babble_sir5). It also reports the interferer
  most often: `ifa` is 0.22 to 0.39 higher than the DeepFilterNet3 chain, and
  its babble F1 is 0.068 lower.
- The causal clearest-voice lock gives +0.010 to +0.012 F1 with two talkers,
  and +0.003 in babble. It lowers `rpa` in every group (0.002 to 0.028), so
  the gain comes from silence while an interferer speaks alone, not from
  better target pitch.
- The offline lock adds +0.050 F1 over the causal lock when the interferer
  speaks first, and +0.024 in babble, again through lower `ifa`. Whole-file
  analysis can reject an early voice that a live lock must follow.
- No method tracks the target well while both voices speak: the best
  `rpa` at talker_sir5 is 0.706 (SwiftF0) against 0.807 on clean speech.

### DeepFilterNet3 on sung mm1-rs recordings (abandoned)

The DeepFilterNet3 finding led to a trial in mm1-rs. The trial put
DeepFilterNet3 ahead of the offline analysis of each recorded clip, with a
toggle for each clip. We abandoned it before a commit, because DeepFilterNet3
removes sung voice.

The input is the 60 mm1-rs recordings in `data/vocal_examples/*/audio.wav`,
48 kHz mono float32, 61.1 s in total. These files are not in git, and they
have no pitch references. The commands are:

```
deep-filter-0.5.6 -D [-a <dB>] -o <out> <in>/*.wav
reim-8c713e7 f0 <wav> 65 800 2048
```

`-a` is the deep-filter attenuation limit in dB. `reim-8c713e7` has SHA-256
`4eb88025a9bfa0221357c558d25a777a5148ccbc63879b3c55f81d8cd49d0633`, the same
`reim` binary as the test runs. Its arguments match the mm1-rs offline
analysis: 65 to 800 Hz and a 2048-point window.

The metric is the number of frames that reim reports as voiced, summed over
the 60 files. Without pitch references, this count measures lost voice, not
accuracy. A lower count means that DeepFilterNet3 removed voice.

| input | voiced frames | clips that lose more than half of their voiced frames |
| --- | --- | --- |
| raw | 4579 | - |
| deep-filter `-D` (the scored configuration) | 3620 | 8 |
| deep-filter `-D -a 12` | 4595 | 0 |
| deep-filter `-D -a 6` | 4518 | 0 |

The last column counts only clips with at least 20 voiced frames in the raw
analysis.

- At full strength, DeepFilterNet3 removes voice but does not move pitch. In
  frames that both tracks report as voiced, f0 agrees within 50 cents. The
  median clip agrees in 100% of those frames.
- The worst clip, `1783624111539_unlabeled_overdub`, goes from 123 voiced
  frames to 0, and its level drops 36 dB.
- A steady synthetic vowel with no noise keeps 3% of its power (-15 dB). The
  vowel is `synth_sung_vowel` from mm1-rs `src/test_utils.rs`: 180 Hz with
  1% vibrato at 5.5 Hz, formants AH, tilt -9 dB per octave, amplitude 0.1,
  1 s. The model
  treats a steady harmonic tone as noise. Read speech does not show this, so
  mixbench cannot detect it.

We measured the cost with the deep-filter binary on an Apple M4 Max. The
rate is the time difference between a 300 s run and a 1 s run, divided by
299 s. Each run used one thread.

| model | CPU seconds per audio second | added latency |
| --- | --- | --- |
| DeepFilterNet3 | 0.031 | 30-40 ms |
| DeepFilterNet3 low latency (`DeepFilterNet3_ll_onnx.tar.gz`) | 0.073 | 10-20 ms |
| reim f0, for comparison | 0.042 | - |

The default model delays its output by 1440 samples (STFT and a 2-frame
lookahead). The low-latency model has no lookahead and delays by 480
samples. Both models then wait up to one 480-sample hop for a full input
frame. In Rust, libDF 0.5.6 is not on crates.io, where the newest release is
0.2.5. It needs a git dependency, tract 0.19, and a 7.98 MB embedded model.

Decision: do not use DeepFilterNet3 at full strength on sung input. The 12 dB
limit keeps the voice, but mixbench did not score it, so its noise gain is
unknown. To revisit, first score `deep-filter -D -a 12` ahead of
reim-live-gate on dev. If the noise gain holds, repeat the recording check
above.

### Verification record

- A verifier rebuilt the scorer from this README alone. Twice it matched
  every per-clip metric, every summary statistic and every primary row
  exactly (largest difference 0.0).
- A verifier rendered the benchmark on four Python/NumPy/libsndfile stacks,
  one of them x86_64. All fingerprints matched, and its own reader confirmed
  the levels, the layouts and the reference times.
- Two adversarial audits found the target-first and tail biases, the
  unscored interferer-first region, the 0 dB order problem, the microphone
  delay and the statistics problems; each fix is a commit on this branch.
