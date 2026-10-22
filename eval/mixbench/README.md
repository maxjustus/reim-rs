# mixbench: target pitch under noise and competing talkers

mixbench measures how well a pitch tracker follows one target voice when real
background noise or other talkers are present. A seeded manifest defines every
mixture, and SHA-256 fingerprints let a third party rebuild the same audio and
score any tracker with the same rules.

Terms:

- **Target**: the talker whose pitch the tracker must report. In every mixture
  the target speaks alone for 0.5-1.0 s before any other talker starts, and the
  target is at least as loud as each other talker.
- **Interferer**: any other talker in the mixture.
- **SNR / SIR**: signal-to-noise and signal-to-interferer ratio, in dB.
- **Active span**: the samples from the first to the last voiced reference
  frame of an utterance (each 32 ms analysis window included).
- **Reference grid**: the 10 ms frames of the target's reference pitch.
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
| dev | F01-F04, M01-M04 | DKITCHEN, NPARK, OOFFICE, PSTATION, STRAFFIC, TBUS | 40 | 360 |
| test | F05-F10, M05-M10 | DLIVING, DWASHING, NFIELD, NRIVER, OHALLWAY, OMEETING, PCAFETER, PRESTO, SCAFE, SPSQUARE, TCAR, TMETRO | 84 | 756 |

Use dev for every design choice and parameter. Score test once, after the
method is frozen. `tune_lock.py` reads dev predictions only.

## Reference

### Mixture rules (`build.py`)

The random generator is `numpy.random.default_rng(20251021)`. One target
utterance produces nine mixtures:

| condition | content | level |
| --- | --- | --- |
| clean | target only | - |
| noise_snr10, noise_snr5, noise_snr0 | target + one DEMAND segment | SNR 10, 5, 0 dB |
| talker_sir10, talker_sir5, talker_sir0 | target + one interferer | SIR 10, 5, 0 dB |
| babble_sir5, babble_sir0 | target + three interferers | SIR 5, 0 dB (all three together) |

- **Targets**: 5 (dev) or 7 (test) utterances per speaker. An utterance is
  eligible when its active span is at least 2.5 s, its file is at most 8 s, and
  it has at least 100 voiced reference frames. The mixture has the length of
  the target file. The target keeps gain 1.
- **Noise**: one environment and start sample per target, used for all three
  SNRs. SNR = target active-span power / noise-segment power.
- **Interferers**: the single interferer has the same sex as the target for
  even target indices and the other sex for odd ones (126 of each in test).
  The three babble talkers are other speakers. Each interferer is a chain of
  its utterances, cut to their active spans, never the target's sentence,
  separated by 0.1-0.4 s pauses. The chain starts 0.5-1.0 s after the target's
  first voiced frame and runs to the end of the mixture. Each talker gets the
  gain that makes its power over its chain equal to target active power /
  (talkers x 10^(SIR/10)).
- **Sum order**: target, then interferer segments in manifest order, then
  noise, all in float64. If the peak exceeds 0.99, `out_gain` scales the sum
  to 0.99 (22 of 756 test mixtures). The WAV holds the float32 result.
- **Fingerprint**: `sha256` in the manifest is the SHA-256 of the float32
  little-endian samples. The manifest stores every gain, so the render step
  does not compute a level.

### Reference tracks

`build.py render` writes `<name>.refs.json` next to each WAV. Reference frame
`i` of a source file sits at `0.016 + 0.010 * i` s, the center of the 32 ms
RAPT window. The target starts at sample 0 of the mixture. An interferer
segment shifts its frames by `(dst_start - src_start) / 48000` s and keeps only
frames whose center lies inside the segment. `f0 > 0` marks a voiced frame.

A sweep of the reference offset from -30 to +30 ms on dev clean speech gave
the best agreement at +5 ms for all four independent trackers tested (reim,
Praat, pYIN, SwiftF0). The gain over the documented offset was at most 0.007
overall accuracy, so the scorer keeps the documented 16 ms.

### Prediction format

One CSV per mixture, `<name>.csv`, rows `time_seconds,f0_hz`, any hop, any
row order, header optional. `f0 <= 0` (or NaN replaced by 0) means unvoiced.

### Metrics (`score.py`)

The scorer evaluates each clip on the target reference grid:

1. For each grid time, take the prediction frame nearest in time (ties go to
   the earlier frame). If that frame is more than 10 ms away, the estimate is
   unvoiced at that time.
2. For each interferer track, take its nearest frame within 5 ms, else
   unvoiced.
3. A frame is a **hit** when the target and the estimate are voiced and
   `|1200 log2(est / target)| <= 50` cents.

With T = target voiced, E = estimate voiced, I = any interferer voiced:

| metric | definition |
| --- | --- |
| `f1` | 2PR / (P + R) with P = `precision`, R = `rpa`; 0 when E is empty |
| `precision` | hits / E |
| `rpa` | hits / T (raw pitch accuracy) |
| `rpa_overlap` | hits in I / T in I |
| `oa` | (hits + frames with neither T nor E) / all frames |
| `vr` | (T and E) / T |
| `vfa` | (E and not T) / not T |
| `gpe` | (T and E and not hit) / (T and E) |
| `capture` | (E, not hit, estimate within 50 cents of a voiced interferer) / I |
| `fpe` | median absolute cents error over hits |

A ratio with a zero denominator is NaN and drops out of the means.
`f1` is the headline metric: it charges a missed target frame and a reported
wrong frame equally, independent of the share of unvoiced frames.

Summary tables (`score.py --compare`) give the mean over clips for each
condition. The 95% interval comes from 2000 bootstrap resamples of target
speakers (seed 0), which keeps the clips of one speaker together. A paired
delta against the first method uses the same resampling on per-clip
differences.

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
