use reim::segment::{
    cents_to_hz, clean_contour, contour_svg, decompose_contour, hz_to_cents, render, segment,
    NoteEdit, SegmentConfig, SegmentKind,
};
use reim::{write_wav, Analyzer, Frame, Reim, Synthesizer, WavData};
use rustfft::FftPlanner;

const FS: f64 = 24_000.0;
const N: usize = 12_000; // 0.5 s

fn synthetic_input() -> Vec<f64> {
    let mut x = vec![0.0; N];
    let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
    let mut noise = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let pi = std::f64::consts::PI;
    for (i, s) in x.iter_mut().enumerate() {
        let t = i as f64 / FS;
        if i < 4_000 {
            let f = 180.0 * (1.0 + 0.03 * (2.0 * pi * 5.0 * t).sin());
            let ph = 2.0 * pi * f * t;
            *s = 0.5 * (ph.sin() + 0.4 * (2.0 * ph).sin() + 0.2 * (3.0 * ph).sin());
        } else if i < 8_000 {
            let ph = 2.0 * pi * 220.0 * t;
            *s = 0.3 * ph.sin() + 0.25 * noise();
        }
    }
    x
}

fn snr_db(reference: &[f64], test: &[f64]) -> f64 {
    let n = reference.len().min(test.len());
    let mut sig = 0.0;
    let mut err = 0.0;
    for i in 0..n {
        sig += reference[i] * reference[i];
        let d = reference[i] - test[i];
        err += d * d;
    }
    10.0 * (sig / (err + 1e-20)).log10()
}

#[test]
fn frame_identity_roundtrip() {
    let input = synthetic_input();

    // Reference: normal process_block path
    let mut reim = Reim::with_defaults(FS);
    let mut reference = vec![0.0; input.len()];
    reim.process_block(&input, &mut reference);

    // Test: analyze_to_frames -> synthesize_frames
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    assert!(!frames.is_empty(), "should produce frames");

    let mut synth = Synthesizer::with_defaults(FS);
    let test_output = synth.synthesize_frames(&frames);

    // The outputs should match at >120 dB SNR (floating-point noise only)
    let n = reference.len().min(test_output.len());
    assert!(n > 0, "outputs should not be empty");
    let snr = snr_db(&reference[..n], &test_output[..n]);
    assert!(
        snr > 120.0,
        "identity round-trip SNR too low: {snr:.1} dB (need >120 dB)"
    );
}

#[test]
fn clean_contour_suppresses_octave_jump() {
    let frames: Vec<Frame> = (0..10)
        .map(|i| Frame {
            fo: if i == 5 { 400.0 } else { 200.0 },
            voiced: true,
            silence: false,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        })
        .collect();
    let cleaned = clean_contour(&frames, 5);
    for (i, &val) in cleaned.iter().enumerate() {
        assert!(
            (val - 200.0).abs() < 1.0,
            "frame {i}: expected ~200, got {val}"
        );
    }
}

#[test]
fn clean_contour_passes_constant_pitch() {
    let frames: Vec<Frame> = (0..20)
        .map(|_| Frame {
            fo: 300.0,
            voiced: true,
            silence: false,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        })
        .collect();
    let cleaned = clean_contour(&frames, 5);
    for &val in &cleaned {
        assert!((val - 300.0).abs() < 0.01);
    }
}

#[test]
fn clean_contour_unvoiced_zero() {
    let frames: Vec<Frame> = (0..5)
        .map(|_| Frame {
            fo: 0.0,
            voiced: false,
            silence: true,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        })
        .collect();
    let cleaned = clean_contour(&frames, 5);
    for &val in &cleaned {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn hz_cents_roundtrip() {
    let test_freqs = [100.0, 200.0, 440.0, 880.0, 1000.0];
    for &freq in &test_freqs {
        let frames = vec![Frame {
            fo: freq,
            voiced: true,
            silence: false,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        }];
        let cleaned = clean_contour(&frames, 1);
        assert!((cleaned[0] - freq).abs() < 0.01, "freq {freq}");
    }
}

#[test]
fn segment_constant_pitch_single_note() {
    let frames: Vec<Frame> = (0..50)
        .map(|_| Frame {
            fo: 200.0,
            voiced: true,
            silence: false,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        })
        .collect();
    let segs = segment(&frames, 200.0, &SegmentConfig::default());
    assert_eq!(segs.len(), 1);
    assert!(matches!(segs[0].kind, SegmentKind::Note(_)));
    assert_eq!(segs[0].frames, 0..50);
}

#[test]
fn segment_step_change_two_notes() {
    let frames: Vec<Frame> = (0..100)
        .map(|i| {
            let fo = if i < 50 { 200.0 } else { 300.0 };
            Frame {
                fo,
                voiced: true,
                silence: false,
                aperiodicity: vec![],
                spectral_envelope: vec![],
            }
        })
        .collect();
    let segs = segment(&frames, 200.0, &SegmentConfig::default());
    let notes: Vec<_> = segs
        .iter()
        .filter(|s| matches!(s.kind, SegmentKind::Note(_)))
        .collect();
    assert!(
        notes.len() >= 2,
        "expected at least 2 notes, got {}",
        notes.len()
    );
}

#[test]
fn segment_silence_is_unvoiced() {
    let frames: Vec<Frame> = (0..20)
        .map(|_| Frame {
            fo: 0.0,
            voiced: false,
            silence: true,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        })
        .collect();
    let segs = segment(&frames, 200.0, &SegmentConfig::default());
    assert_eq!(segs.len(), 1);
    assert!(matches!(segs[0].kind, SegmentKind::Unvoiced));
}

#[test]
fn segment_voiced_silence_voiced() {
    let mut frames = Vec::new();
    for _ in 0..30 {
        frames.push(Frame {
            fo: 200.0,
            voiced: true,
            silence: false,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        });
    }
    for _ in 0..10 {
        frames.push(Frame {
            fo: 0.0,
            voiced: false,
            silence: true,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        });
    }
    for _ in 0..30 {
        frames.push(Frame {
            fo: 200.0,
            voiced: true,
            silence: false,
            aperiodicity: vec![],
            spectral_envelope: vec![],
        });
    }
    let segs = segment(&frames, 200.0, &SegmentConfig::default());
    assert_eq!(segs.len(), 3, "expected Note, Unvoiced, Note");
    assert!(matches!(segs[0].kind, SegmentKind::Note(_)));
    assert!(matches!(segs[1].kind, SegmentKind::Unvoiced));
    assert!(matches!(segs[2].kind, SegmentKind::Note(_)));
}

// --- decompose_contour tests ---

const FRAME_RATE: f64 = 200.0;

fn make_config() -> SegmentConfig {
    SegmentConfig::default()
}

#[test]
fn decompose_constant_pitch() {
    let fo: Vec<f64> = vec![440.0; 200];
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config(), &mut FftPlanner::new());
    assert!(
        contour.center_cents.abs() < 1.0,
        "center should be ~0 cents from A4"
    );
    let drift_rms =
        (contour.drift.iter().map(|d| d * d).sum::<f64>() / contour.drift.len() as f64).sqrt();
    assert!(drift_rms < 1.0, "drift RMS should be near zero");
    assert!(contour.vibrato_rate_hz < 0.1, "no vibrato expected");
}

#[test]
fn decompose_known_vibrato() {
    let pi = std::f64::consts::PI;
    let fo: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / FRAME_RATE;
            let cents_deviation = 30.0 * (2.0 * pi * 6.0 * t).sin();
            440.0 * 2.0_f64.powf(cents_deviation / 1200.0)
        })
        .collect();
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config(), &mut FftPlanner::new());

    assert!(
        (contour.vibrato_rate_hz - 6.0).abs() < 0.5,
        "vibrato rate: expected ~6Hz, got {}",
        contour.vibrato_rate_hz
    );

    let edge = 20;
    let inner_amp: Vec<f64> = contour.vibrato_amp[edge..contour.vibrato_amp.len() - edge].to_vec();
    let mean_amp = inner_amp.iter().sum::<f64>() / inner_amp.len() as f64;
    assert!(
        (mean_amp - 30.0).abs() < 5.0,
        "vibrato amplitude: expected ~30 cents, got {mean_amp}"
    );

    let residual_rms = (contour.residual.iter().map(|r| r * r).sum::<f64>()
        / contour.residual.len() as f64)
        .sqrt();
    assert!(residual_rms < 5.0, "residual RMS too high: {residual_rms}");
}

#[test]
fn decompose_known_drift() {
    let fo: Vec<f64> = (0..200)
        .map(|i| {
            let cents_drift = 40.0 * i as f64 / 199.0;
            440.0 * 2.0_f64.powf(cents_drift / 1200.0)
        })
        .collect();
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config(), &mut FftPlanner::new());

    let true_drift: Vec<f64> = (0..200).map(|i| 40.0 * i as f64 / 199.0 - 20.0).collect();
    let drift_centered: Vec<f64> = {
        let mean = contour.drift.iter().sum::<f64>() / contour.drift.len() as f64;
        contour.drift.iter().map(|d| d - mean).collect()
    };
    let true_centered: Vec<f64> = {
        let mean = true_drift.iter().sum::<f64>() / true_drift.len() as f64;
        true_drift.iter().map(|d| d - mean).collect()
    };
    let corr = correlation(&drift_centered, &true_centered);
    assert!(
        corr > 0.95,
        "drift correlation with true ramp: {corr} (need >0.95)"
    );

    let vib_rms = (contour.vibrato_amp.iter().map(|a| a * a).sum::<f64>()
        / contour.vibrato_amp.len() as f64)
        .sqrt();
    assert!(
        contour.vibrato_rate_hz < 0.1 || vib_rms < 2.0,
        "no vibrato expected on pure drift"
    );
}

#[test]
fn decompose_drift_plus_vibrato() {
    let pi = std::f64::consts::PI;
    let fo: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / FRAME_RATE;
            let drift = 30.0 * i as f64 / 199.0;
            let vib = 20.0 * (2.0 * pi * 5.0 * t).sin();
            440.0 * 2.0_f64.powf((drift + vib) / 1200.0)
        })
        .collect();
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config(), &mut FftPlanner::new());

    assert!(
        (contour.vibrato_rate_hz - 5.0).abs() < 1.0,
        "vibrato rate: expected ~5Hz, got {}",
        contour.vibrato_rate_hz
    );
}

#[test]
fn decompose_no_vibrato_jitter() {
    let mut lcg: u64 = 12345;
    let fo: Vec<f64> = (0..200)
        .map(|_| {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let jitter = ((lcg >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 10.0;
            440.0 * 2.0_f64.powf(jitter / 1200.0)
        })
        .collect();
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config(), &mut FftPlanner::new());

    let mean_vib_amp = contour.vibrato_amp.iter().sum::<f64>() / contour.vibrato_amp.len() as f64;
    assert!(
        contour.vibrato_rate_hz < 0.1 || mean_vib_amp < 2.0,
        "false positive vibrato: rate={}, amp={mean_vib_amp}",
        contour.vibrato_rate_hz
    );
}

#[test]
fn decompose_reconstruction_accuracy() {
    let pi = std::f64::consts::PI;
    let fo: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / FRAME_RATE;
            let drift = 20.0 * i as f64 / 199.0;
            let vib = 25.0 * (2.0 * pi * 6.0 * t).sin();
            440.0 * 2.0_f64.powf((drift + vib) / 1200.0)
        })
        .collect();
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config(), &mut FftPlanner::new());

    for i in 0..fo.len() {
        let reconstructed_cents = contour.center_cents
            + contour.drift[i]
            + contour.vibrato_amp[i] * contour.vibrato_phase[i].sin()
            + contour.residual[i];
        let original_cents = 1200.0 * (fo[i] / 440.0).log2();
        let err = (reconstructed_cents - original_cents).abs();
        assert!(err < 0.5, "frame {i}: reconstruction error {err} cents");
    }
}

#[test]
fn decompose_ramped_vibrato() {
    let pi = std::f64::consts::PI;
    let fo: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / FRAME_RATE;
            let ramp = 30.0 * i as f64 / 199.0;
            let vib = ramp * (2.0 * pi * 6.0 * t).sin();
            440.0 * 2.0_f64.powf(vib / 1200.0)
        })
        .collect();
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config(), &mut FftPlanner::new());

    let true_amp: Vec<f64> = (0..200).map(|i| 30.0 * i as f64 / 199.0).collect();
    let edge = 20;
    let inner_detected: Vec<f64> = contour.vibrato_amp[edge..200 - edge].to_vec();
    let inner_true: Vec<f64> = true_amp[edge..200 - edge].to_vec();
    let corr = correlation(&inner_detected, &inner_true);
    assert!(
        corr > 0.85,
        "ramped vibrato amplitude correlation: {corr} (need >0.85)"
    );
}

#[test]
fn contour_svg_structural() {
    let frames: Vec<Frame> = (0..100)
        .map(|i| {
            if i < 40 {
                Frame {
                    fo: 220.0,
                    voiced: true,
                    silence: false,
                    aperiodicity: vec![],
                    spectral_envelope: vec![],
                }
            } else if i < 60 {
                Frame {
                    fo: 0.0,
                    voiced: false,
                    silence: true,
                    aperiodicity: vec![],
                    spectral_envelope: vec![],
                }
            } else {
                Frame {
                    fo: 330.0,
                    voiced: true,
                    silence: false,
                    aperiodicity: vec![],
                    spectral_envelope: vec![],
                }
            }
        })
        .collect();
    let config = SegmentConfig::default();
    let segs = segment(&frames, 200.0, &config);
    let svg = contour_svg(&frames, &segs, 200.0, None);

    assert!(svg.starts_with("<svg"), "should start with <svg tag");
    assert!(svg.contains("</svg>"), "should have closing svg tag");
    assert!(svg.contains("circle"), "should have Fo dots (circles)");
    assert!(svg.contains("rect"), "should have unvoiced shading (rects)");
    assert!(
        svg.contains("polyline") || svg.contains("line"),
        "should have contour lines"
    );
}

// --- glide detection tests ---

fn voiced_frame(fo: f64) -> Frame {
    Frame {
        fo,
        voiced: true,
        silence: false,
        aperiodicity: vec![0.1; 10],
        spectral_envelope: vec![1.0; 10],
    }
}

/// Hold at f_a, cosine-eased cents ramp over `glide` frames, hold at f_b.
fn glide_input(f_a: f64, f_b: f64, hold: usize, glide: usize) -> Vec<Frame> {
    let ca = hz_to_cents(f_a);
    let cb = hz_to_cents(f_b);
    let pi = std::f64::consts::PI;
    let mut v = Vec::new();
    for _ in 0..hold {
        v.push(voiced_frame(f_a));
    }
    for j in 0..glide {
        let x = (j + 1) as f64 / (glide + 1) as f64;
        let c = ca + (cb - ca) * 0.5 * (1.0 - (pi * x).cos());
        v.push(voiced_frame(cents_to_hz(c)));
    }
    for _ in 0..hold {
        v.push(voiced_frame(f_b));
    }
    v
}

fn note_contours(segs: &[reim::segment::Segment]) -> Vec<&reim::segment::NoteContour> {
    segs.iter()
        .filter_map(|s| match &s.kind {
            SegmentKind::Note(nc) => Some(nc),
            SegmentKind::Unvoiced => None,
        })
        .collect()
}

#[test]
fn glide_detected_between_notes() {
    let frames = glide_input(200.0, 300.0, 60, 20);
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let notes = note_contours(&segs);
    assert_eq!(notes.len(), 2, "expected 2 notes, got {}", notes.len());
    assert!(notes[0].onset_glide.is_empty(), "first note has no glide");
    let g = notes[1].onset_glide.len();
    assert!((12..=24).contains(&g), "glide len {g}, expected ~20");
    let depth = notes[1].onset_glide_depth_cents;
    assert!(
        (depth - 702.0).abs() < 100.0,
        "depth {depth}, expected ~702 cents"
    );
    // First note ends where the glide starts: its last frame is still near 200 Hz.
    let a_end = segs[0].frames.end;
    let last_cents = hz_to_cents(frames[a_end - 1].fo) - hz_to_cents(200.0);
    assert!(
        last_cents.abs() < 60.0,
        "first note's last frame {last_cents} cents from 200 Hz"
    );
}

#[test]
fn hard_step_has_no_glide() {
    let frames = glide_input(200.0, 300.0, 60, 0);
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let notes = note_contours(&segs);
    assert_eq!(notes.len(), 2);
    for nc in &notes {
        assert!(nc.onset_glide.is_empty(), "hard step must not have a glide");
    }
}

#[test]
fn glide_with_vibrato_not_swallowed() {
    // 6 Hz, 40-cent vibrato on both notes, 20-frame glide between them.
    let pi = std::f64::consts::PI;
    let base = glide_input(200.0, 300.0, 150, 20);
    let frames: Vec<Frame> = base
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let t = i as f64 / FRAME_RATE;
            let vib = 40.0 * (2.0 * pi * 6.0 * t).sin();
            voiced_frame(f.fo * 2.0_f64.powf(vib / 1200.0))
        })
        .collect();
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let notes = note_contours(&segs);
    assert_eq!(notes.len(), 2, "expected 2 notes, got {}", notes.len());
    let g = notes[1].onset_glide.len();
    assert!(
        (8..=40).contains(&g),
        "glide len {g}: must not swallow vibrato cycles (true len 20)"
    );
    // Vibrato survives in the second note's core decomposition.
    let nc = notes[1];
    let mean_amp: f64 = nc.vibrato_amp.iter().sum::<f64>() / nc.vibrato_amp.len() as f64;
    assert!(
        (mean_amp - 40.0).abs() < 20.0,
        "vibrato amp mean {mean_amp}, expected ~40"
    );
    // Regression guard: excluding glide frames from decomposition keeps residual small.
    let res_rms =
        (nc.residual.iter().map(|r| r * r).sum::<f64>() / nc.residual.len() as f64).sqrt();
    assert!(
        res_rms < 15.0,
        "residual RMS {res_rms}, expected < 15 cents"
    );
}

#[test]
fn shallow_step_no_glide() {
    // 25-cent step: below glide_min_cents (and below note-split threshold).
    let f_b = 200.0 * 2.0_f64.powf(25.0 / 1200.0);
    let frames = glide_input(200.0, f_b, 60, 10);
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    for nc in note_contours(&segs) {
        assert!(
            nc.onset_glide.is_empty(),
            "25-cent move must not be a glide"
        );
    }
}

#[test]
fn long_glide_capped() {
    // 500 ms glide (100 frames) gets capped at max_glide_frames.
    let config = SegmentConfig::default();
    let frames = glide_input(200.0, 300.0, 60, 100);
    let segs = segment(&frames, FRAME_RATE, &config);
    let notes = note_contours(&segs);
    let max_glide = notes.iter().map(|nc| nc.onset_glide.len()).max().unwrap();
    assert!(
        max_glide <= config.max_glide_frames,
        "glide {max_glide} exceeds cap {}",
        config.max_glide_frames
    );
    assert!(
        max_glide >= config.max_glide_frames / 2,
        "glide {max_glide}: long portamento should still be substantially detected"
    );
}

#[test]
fn render_identity_with_glide() {
    let frames = glide_input(200.0, 300.0, 60, 20);
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let rendered = render(&frames, &segs, &[]);
    assert_eq!(rendered.len(), frames.len());
    for (i, (orig, rend)) in frames.iter().zip(rendered.iter()).enumerate() {
        let err = (hz_to_cents(orig.fo) - hz_to_cents(rend.fo)).abs();
        assert!(err < 1e-6, "frame {i}: {err} cents off under identity");
    }
}

#[test]
fn contour_svg_draws_glide() {
    let frames = glide_input(200.0, 300.0, 60, 20);
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let svg = contour_svg(&frames, &segs, FRAME_RATE, None);
    assert!(
        svg.contains("#d2a"),
        "SVG should contain the magenta glide polyline"
    );
}

// --- scoop-from-silence tests ---

/// 20 unvoiced frames, a 15-frame linear rise from 150 cents below the
/// target into a 100-frame hold at 300 Hz.
fn scoop_input() -> Vec<Frame> {
    let mut frames: Vec<Frame> = (0..20)
        .map(|_| Frame {
            fo: 0.0,
            voiced: false,
            silence: true,
            aperiodicity: vec![0.1; 10],
            spectral_envelope: vec![1.0; 10],
        })
        .collect();
    let tc = hz_to_cents(300.0);
    for j in 0..15 {
        let c = tc - 150.0 * (1.0 - (j as f64 + 1.0) / 16.0);
        frames.push(voiced_frame(cents_to_hz(c)));
    }
    for _ in 0..100 {
        frames.push(voiced_frame(300.0));
    }
    frames
}

#[test]
fn scoop_from_silence_detected() {
    let frames = scoop_input();
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let notes = note_contours(&segs);
    assert_eq!(notes.len(), 1, "expected 1 note, got {}", notes.len());
    let nc = notes[0];
    let g = nc.onset_glide.len();
    assert!((8..=20).contains(&g), "scoop len {g}, expected ~13");
    assert!(
        (nc.onset_glide_depth_cents - 150.0).abs() < 50.0,
        "scoop depth {}, expected ~150 cents",
        nc.onset_glide_depth_cents
    );
}

#[test]
fn scoop_depth_preserved_under_pitch_edit() {
    let frames = scoop_input();
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let (note_idx, nc) = segs
        .iter()
        .enumerate()
        .find_map(|(i, s)| match &s.kind {
            SegmentKind::Note(nc) => Some((i, nc)),
            SegmentKind::Unvoiced => None,
        })
        .unwrap();
    let g = nc.onset_glide.len();
    assert!(g >= 2, "fixture must produce a scoop");
    let note_start = segs[note_idx].frames.start;
    let old_entry = hz_to_cents(frames[note_start + g].fo);

    let mut edit = NoteEdit::identity(note_idx);
    edit.target_cents = Some(nc.center_cents + 300.0);
    let rendered = render(&frames, &segs, &[edit]);

    // With no previous voiced pitch, the scoop shifts with the note,
    // preserving its depth below the (edited) entry pitch.
    let first = hz_to_cents(rendered[note_start].fo);
    let expected = old_entry + 300.0 - nc.onset_glide_depth_cents;
    assert!(
        (first - expected).abs() < 1e-6,
        "scoop start {first}, expected {expected}"
    );
}

// --- glide edit tests ---

/// (frames, segments, a_len, glide_len, entry_cents) for the standard
/// two-note glide fixture. a_len is the first note's frame count.
fn glide_fixture() -> (Vec<Frame>, Vec<reim::segment::Segment>, usize, usize, f64) {
    let frames = glide_input(200.0, 300.0, 60, 20);
    let segs = segment(&frames, FRAME_RATE, &SegmentConfig::default());
    let notes = note_contours(&segs);
    assert_eq!(notes.len(), 2);
    let a_len = segs[0].frames.len();
    let g = notes[1].onset_glide.len();
    assert!(g >= 2, "fixture must produce a glide");
    let entry = hz_to_cents(frames[segs[1].frames.start + g].fo);
    (frames, segs, a_len, g, entry)
}

#[test]
fn render_glide_retargets_after_pitch_edit() {
    let (frames, segs, a_len, g, old_entry) = glide_fixture();
    let edit = NoteEdit {
        target_cents: Some(note_contours(&segs)[1].center_cents + 100.0),
        ..NoteEdit::identity(1)
    };
    let rendered = render(&frames, &segs, &[edit]);
    assert_eq!(rendered.len(), frames.len());

    let exit = hz_to_cents(rendered[a_len - 1].fo);
    let entry = hz_to_cents(rendered[a_len + g].fo);
    assert!(
        (entry - (old_entry + 100.0)).abs() < 1e-6,
        "entry {entry} should be old entry + 100"
    );
    // The glide must connect exit to the NEW entry without a jump.
    let depth = entry - exit;
    let max_step = (depth.abs() / g as f64) * 3.0 + 1.0;
    for i in a_len..=a_len + g {
        let step = hz_to_cents(rendered[i].fo) - hz_to_cents(rendered[i - 1].fo);
        assert!(
            step.abs() < max_step,
            "jump of {step} cents at frame {i} (max {max_step})"
        );
    }
}

#[test]
fn render_glide_scale_zero_is_hard_step() {
    let (frames, segs, a_len, g, entry) = glide_fixture();
    let mut edit = NoteEdit::identity(1);
    edit.glide_scale = 0.0;
    let rendered = render(&frames, &segs, &[edit]);
    assert_eq!(rendered.len(), frames.len(), "duration unchanged");
    for (i, f) in rendered.iter().enumerate().take(a_len + g).skip(a_len) {
        let c = hz_to_cents(f.fo);
        assert!(
            (c - entry).abs() < 1e-9,
            "glide frame {i} at {c}, expected entry pitch {entry}"
        );
    }
}

#[test]
fn render_glide_time_scale_stretches_glide() {
    let (frames, segs, a_len, g, entry) = glide_fixture();
    let mut edit = NoteEdit::identity(1);
    edit.glide_time_scale = 2.0;
    let rendered = render(&frames, &segs, &[edit]);
    assert_eq!(
        rendered.len(),
        frames.len() + g,
        "glide doubles, core unchanged"
    );
    let last_glide = hz_to_cents(rendered[a_len + 2 * g - 1].fo);
    assert!(
        (last_glide - entry).abs() < 80.0,
        "stretched glide should still land near entry, got {last_glide} vs {entry}"
    );
}

#[test]
fn render_glide_time_scale_zero_drops_glide() {
    let (frames, segs, a_len, g, entry) = glide_fixture();
    let mut edit = NoteEdit::identity(1);
    edit.glide_time_scale = 0.0;
    let rendered = render(&frames, &segs, &[edit]);
    assert_eq!(
        rendered.len(),
        frames.len() - g,
        "note shortens by glide_len"
    );
    let first = hz_to_cents(rendered[a_len].fo);
    assert!(
        (first - entry).abs() < 1e-9,
        "note now starts at entry pitch, got {first} vs {entry}"
    );
}

#[test]
fn render_glide_survives_time_stretch() {
    let (frames, segs, a_len, g, _) = glide_fixture();
    let src_len = segs[1].frames.len();
    let mut edit = NoteEdit::identity(1);
    edit.out_len = Some(2 * src_len);
    let rendered = render(&frames, &segs, &[edit]);
    assert_eq!(rendered.len(), a_len + 2 * src_len);
    // Continuity through the stretched glide: half the per-frame slope.
    let depth = 702.0;
    let max_step = depth / (2 * g) as f64 * 3.0 + 1.0;
    for i in a_len..a_len + 2 * g {
        let step = hz_to_cents(rendered[i].fo) - hz_to_cents(rendered[i - 1].fo);
        assert!(
            step.abs() < max_step,
            "jump of {step} cents at frame {i} (max {max_step})"
        );
    }
}

#[test]
fn render_glide_connects_two_edited_notes() {
    let (frames, segs, a_len, g, old_entry) = glide_fixture();
    let centers: Vec<f64> = note_contours(&segs)
        .iter()
        .map(|nc| nc.center_cents)
        .collect();
    let edit_a = NoteEdit {
        target_cents: Some(centers[0] - 100.0),
        ..NoteEdit::identity(0)
    };
    let edit_b = NoteEdit {
        target_cents: Some(centers[1] + 100.0),
        ..NoteEdit::identity(1)
    };
    let rendered = render(&frames, &segs, &[edit_a, edit_b]);

    let exit = hz_to_cents(rendered[a_len - 1].fo);
    let entry = hz_to_cents(rendered[a_len + g].fo);
    assert!(
        (entry - (old_entry + 100.0)).abs() < 1e-6,
        "entry should follow note B's edit"
    );
    let first_glide = hz_to_cents(rendered[a_len].fo);
    assert!(
        (first_glide - exit).abs() < 80.0,
        "glide should start from note A's edited exit: {first_glide} vs {exit}"
    );
}

// --- render tests ---

#[test]
fn render_identity_preserves_frames() {
    let frames: Vec<Frame> = (0..50)
        .map(|i| Frame {
            fo: 200.0 + i as f64,
            voiced: true,
            silence: false,
            aperiodicity: vec![0.1; 10],
            spectral_envelope: vec![1.0; 10],
        })
        .collect();
    let config = SegmentConfig::default();
    let segs = segment(&frames, 200.0, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, _)| NoteEdit::identity(i))
        .collect();

    let rendered = render(&frames, &segs, &edits);
    assert_eq!(rendered.len(), frames.len());

    for (i, (orig, rend)) in frames.iter().zip(rendered.iter()).enumerate() {
        let err = (orig.fo - rend.fo).abs();
        assert!(err < 0.1, "frame {i}: fo mismatch {err}");
    }
}

#[test]
fn render_pitch_correction() {
    let frames: Vec<Frame> = (0..50)
        .map(|_| Frame {
            fo: 220.0,
            voiced: true,
            silence: false,
            aperiodicity: vec![0.1; 10],
            spectral_envelope: vec![1.0; 10],
        })
        .collect();
    let config = SegmentConfig::default();
    let segs = segment(&frames, 200.0, &config);

    let edits = vec![NoteEdit {
        target_cents: Some(0.0), // A4
        drift_scale: 0.0,
        ..NoteEdit::identity(0)
    }];

    let rendered = render(&frames, &segs, &edits);
    for f in &rendered {
        if f.voiced {
            let cents = 1200.0 * (f.fo / 440.0).log2();
            assert!(cents.abs() < 5.0, "should be near A4, got {} cents", cents);
        }
    }
}

#[test]
fn render_vibrato_removal() {
    let pi = std::f64::consts::PI;
    let frames: Vec<Frame> = (0..100)
        .map(|i| {
            let t = i as f64 / 200.0;
            let vib = 30.0 * (2.0 * pi * 6.0 * t).sin();
            Frame {
                fo: 440.0 * 2.0_f64.powf(vib / 1200.0),
                voiced: true,
                silence: false,
                aperiodicity: vec![0.1; 10],
                spectral_envelope: vec![1.0; 10],
            }
        })
        .collect();
    let config = SegmentConfig::default();
    let segs = segment(&frames, 200.0, &config);

    let edits = vec![NoteEdit {
        vibrato_scale: 0.0,
        ..NoteEdit::identity(0)
    }];

    let rendered = render(&frames, &segs, &edits);
    let cents: Vec<f64> = rendered
        .iter()
        .filter(|f| f.voiced)
        .map(|f| 1200.0 * (f.fo / 440.0).log2())
        .collect();
    if cents.len() > 2 {
        let mean = cents.iter().sum::<f64>() / cents.len() as f64;
        let var = cents.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / cents.len() as f64;
        let std_dev = var.sqrt();
        assert!(
            std_dev < 10.0,
            "vibrato should be reduced, std_dev={std_dev}"
        );
    }
}

#[test]
fn render_time_stretch() {
    let frames: Vec<Frame> = (0..50)
        .map(|_| Frame {
            fo: 300.0,
            voiced: true,
            silence: false,
            aperiodicity: vec![0.1; 10],
            spectral_envelope: vec![1.0; 10],
        })
        .collect();
    let config = SegmentConfig::default();
    let segs = segment(&frames, 200.0, &config);

    let edits = vec![NoteEdit {
        out_len: Some(100),
        ..NoteEdit::identity(0)
    }];

    let rendered = render(&frames, &segs, &edits);
    assert_eq!(
        rendered.len(),
        100,
        "time stretch 2x should double frame count"
    );
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum_ab = 0.0;
    let mut sum_a2 = 0.0;
    let mut sum_b2 = 0.0;
    for i in 0..n {
        sum_ab += a[i] * b[i];
        sum_a2 += a[i] * a[i];
        sum_b2 += b[i] * b[i];
    }
    sum_ab / (sum_a2.sqrt() * sum_b2.sqrt() + 1e-20)
}

// --- end-to-end integration tests ---

#[test]
fn e2e_segment_boundaries() {
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let has_note = segs.iter().any(|s| matches!(s.kind, SegmentKind::Note(_)));
    let has_unvoiced = segs.iter().any(|s| matches!(s.kind, SegmentKind::Unvoiced));
    assert!(has_note, "should detect at least one note");
    assert!(has_unvoiced, "should detect silence as unvoiced");

    assert!(
        matches!(segs.last().unwrap().kind, SegmentKind::Unvoiced),
        "last segment should be unvoiced (silence)"
    );
}

#[test]
fn e2e_identity_through_segment_render() {
    let input = synthetic_input();

    let mut reim = Reim::with_defaults(FS);
    let mut reference = vec![0.0; input.len()];
    reim.process_block(&input, &mut reference);

    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, _)| NoteEdit::identity(i))
        .collect();

    let rendered = render(&frames, &segs, &edits);
    assert_eq!(
        rendered.len(),
        frames.len(),
        "identity render should preserve frame count"
    );

    let mut synth = Synthesizer::with_defaults(FS);
    let test_output = synth.synthesize_frames(&rendered);

    let n = reference.len().min(test_output.len());
    let snr = snr_db(&reference[..n], &test_output[..n]);
    assert!(
        snr > 80.0,
        "full pipeline identity SNR too low: {snr:.1} dB (need >80 dB)"
    );
}

#[test]
fn e2e_pitch_correction() {
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let target = -1200.0;
    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, _)| NoteEdit {
            target_cents: Some(target),
            drift_scale: 0.0,
            ..NoteEdit::identity(i)
        })
        .collect();

    let rendered = render(&frames, &segs, &edits);

    let voiced_cents: Vec<f64> = rendered
        .iter()
        .filter(|f| f.voiced)
        .map(|f| 1200.0 * (f.fo / 440.0).log2())
        .collect();

    if !voiced_cents.is_empty() {
        let mean = voiced_cents.iter().sum::<f64>() / voiced_cents.len() as f64;
        assert!(
            (mean - target).abs() < 50.0,
            "mean pitch should be near A3 ({target} cents), got {mean} cents"
        );
    }
}

#[test]
fn e2e_vibrato_removal() {
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, _)| NoteEdit {
            vibrato_scale: 0.0,
            ..NoteEdit::identity(i)
        })
        .collect();

    let rendered = render(&frames, &segs, &edits);

    // Walk segments with an output offset to index rendered correctly.
    // (Identity edits preserve lengths, but this pattern stays correct
    // if future edits change segment lengths.)
    let mut out_offset = 0;
    for (seg_idx, seg) in segs.iter().enumerate() {
        let edit = edits.iter().find(|e| e.segment_index == seg_idx);
        let out_len = edit.and_then(|e| e.out_len).unwrap_or(seg.frames.len());

        if matches!(seg.kind, SegmentKind::Note(_)) {
            let original_cents: Vec<f64> = frames[seg.frames.clone()]
                .iter()
                .filter(|f| f.voiced)
                .map(|f| 1200.0 * (f.fo / 440.0).log2())
                .collect();
            let rendered_cents: Vec<f64> = rendered[out_offset..out_offset + out_len]
                .iter()
                .filter(|f| f.voiced)
                .map(|f| 1200.0 * (f.fo / 440.0).log2())
                .collect();

            if original_cents.len() > 2 && rendered_cents.len() > 2 {
                let var = |v: &[f64]| {
                    let mean = v.iter().sum::<f64>() / v.len() as f64;
                    v.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / v.len() as f64
                };
                let orig_var = var(&original_cents);
                let rend_var = var(&rendered_cents);
                assert!(
                    rend_var <= orig_var,
                    "vibrato removal should reduce variance: orig={orig_var:.1}, rendered={rend_var:.1}"
                );
            }
        }
        out_offset += out_len;
    }
}

#[test]
fn e2e_time_stretch() {
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, s)| NoteEdit {
            out_len: Some(s.frames.len() * 2),
            ..NoteEdit::identity(i)
        })
        .collect();

    let rendered = render(&frames, &segs, &edits);
    assert!(
        rendered.len() > frames.len(),
        "stretched output should be longer"
    );

    let mut synth = Synthesizer::with_defaults(FS);
    let output = synth.synthesize_frames(&rendered);
    assert!(
        output.len() > input.len(),
        "synthesized stretched output should be longer"
    );
}

#[test]
fn e2e_achieved_vs_intended_identity() {
    // Use a longer clean signal (2s voiced vibrato tone) so the pitch tracker
    // has enough material after the fftsize ramp-up.
    let long_n: usize = 48_000;
    let long_input = {
        let pi = std::f64::consts::PI;
        (0..long_n)
            .map(|i| {
                let t = i as f64 / FS;
                let f = 220.0 * (1.0 + 0.02 * (2.0 * pi * 5.5 * t).sin());
                let ph = 2.0 * pi * f * t;
                0.5 * (ph.sin() + 0.3 * (2.0 * ph).sin())
            })
            .collect::<Vec<f64>>()
    };

    let mut a1 = Analyzer::with_defaults(FS);
    let orig_frames = a1.analyze_to_frames(&long_input);

    let config2 = SegmentConfig::default();
    let segs2 = segment(&orig_frames, FRAME_RATE, &config2);
    let edits2: Vec<NoteEdit> = segs2
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, _)| NoteEdit::identity(i))
        .collect();
    let rendered2 = render(&orig_frames, &segs2, &edits2);

    let mut synth2 = Synthesizer::with_defaults(FS);
    let out2 = synth2.synthesize_frames(&rendered2);

    let mut a2 = Analyzer::with_defaults(FS);
    let re2 = a2.analyze_to_frames(&out2);

    // Compare the MEDIAN pitch of the re-analyzed output against the intended
    // median. Frame-by-frame comparison is unreliable because the pitch tracker
    // can lock onto different harmonics on resynthesized audio, producing large
    // per-frame errors even when the perceived pitch is correct. Median-to-median
    // comparison filters out these outlier frames and catches real pitch shifts.
    let orig_voiced_cents: Vec<f64> = orig_frames
        .iter()
        .filter(|f| f.voiced && f.fo > 0.0)
        .map(|f| 1200.0 * (f.fo / 440.0).log2())
        .collect();
    let re_voiced_cents: Vec<f64> = re2
        .iter()
        .filter(|f| f.voiced && f.fo > 0.0)
        .map(|f| 1200.0 * (f.fo / 440.0).log2())
        .collect();
    assert!(
        orig_voiced_cents.len() >= 10 && re_voiced_cents.len() >= 10,
        "need voiced frames: orig={}, re={}",
        orig_voiced_cents.len(),
        re_voiced_cents.len()
    );

    let median_of = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let orig_median = median_of(&orig_voiced_cents);
    let re_median = median_of(&re_voiced_cents);
    let median_shift = (re_median - orig_median).abs();
    eprintln!(
        "identity achieved-vs-intended: orig_median={orig_median:.1} cents, \
         re_median={re_median:.1} cents, shift={median_shift:.1} cents"
    );

    // The identity round-trip should not shift the perceived pitch center by
    // more than ~50 cents (half semitone). The vocoder's minimum-phase pulse
    // synthesis can shift the pitch tracker's estimate by tens of cents due to
    // harmonic structure changes. A real bug (wrong octave, contour corruption)
    // would show shifts of hundreds of cents.
    assert!(
        median_shift < 50.0,
        "identity pitch center shifted: {median_shift:.1} cents (need <50)"
    );
}

// --- listening tests (write output files for manual inspection) ---

fn ensure_output_dir() {
    std::fs::create_dir_all("tests/output").ok();
}

#[test]
#[ignore]
fn write_identity_roundtrip() {
    ensure_output_dir();
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let mut synth = Synthesizer::with_defaults(FS);
    let output = synth.synthesize_frames(&frames);

    let wav = WavData {
        sample_rate: FS as u32,
        samples: output,
    };
    write_wav("tests/output/identity.wav", &wav).unwrap();
    eprintln!("wrote tests/output/identity.wav");
}

#[test]
#[ignore]
fn write_pitch_correction() {
    ensure_output_dir();
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if let SegmentKind::Note(c) = &s.kind {
                let nearest_semitone = (c.center_cents / 100.0).round() * 100.0;
                Some(NoteEdit {
                    target_cents: Some(nearest_semitone),
                    drift_scale: 0.0,
                    ..NoteEdit::identity(i)
                })
            } else {
                None
            }
        })
        .collect();

    let rendered = render(&frames, &segs, &edits);
    let mut synth = Synthesizer::with_defaults(FS);
    let output = synth.synthesize_frames(&rendered);

    let wav = WavData {
        sample_rate: FS as u32,
        samples: output.clone(),
    };
    write_wav("tests/output/pitch_corrected.wav", &wav).unwrap();

    let intended_fo: Vec<f64> = rendered.iter().map(|f| f.fo).collect();
    let mut re_analyzer = Analyzer::with_defaults(FS);
    let re_frames = re_analyzer.analyze_to_frames(&output);
    let re_segs = segment(&re_frames, FRAME_RATE, &config);
    let svg = contour_svg(&re_frames, &re_segs, FRAME_RATE, Some(&intended_fo));
    std::fs::write("tests/output/pitch_corrected.svg", &svg).unwrap();

    eprintln!("wrote tests/output/pitch_corrected.wav and .svg");
}

#[test]
#[ignore]
fn write_vibrato_removal() {
    ensure_output_dir();
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, _)| NoteEdit {
            vibrato_scale: 0.0,
            ..NoteEdit::identity(i)
        })
        .collect();

    let rendered = render(&frames, &segs, &edits);
    let mut synth = Synthesizer::with_defaults(FS);
    let output = synth.synthesize_frames(&rendered);

    let wav = WavData {
        sample_rate: FS as u32,
        samples: output.clone(),
    };
    write_wav("tests/output/no_vibrato.wav", &wav).unwrap();

    let intended_fo: Vec<f64> = rendered.iter().map(|f| f.fo).collect();
    let mut re_analyzer = Analyzer::with_defaults(FS);
    let re_frames = re_analyzer.analyze_to_frames(&output);
    let re_segs = segment(&re_frames, FRAME_RATE, &config);
    let svg = contour_svg(&re_frames, &re_segs, FRAME_RATE, Some(&intended_fo));
    std::fs::write("tests/output/no_vibrato.svg", &svg).unwrap();

    eprintln!("wrote tests/output/no_vibrato.wav and .svg");
}

#[test]
#[ignore]
fn write_transpose() {
    ensure_output_dir();
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if let SegmentKind::Note(c) = &s.kind {
                Some(NoteEdit {
                    target_cents: Some(c.center_cents + 300.0),
                    ..NoteEdit::identity(i)
                })
            } else {
                None
            }
        })
        .collect();

    let rendered = render(&frames, &segs, &edits);
    let mut synth = Synthesizer::with_defaults(FS);
    let output = synth.synthesize_frames(&rendered);

    let wav = WavData {
        sample_rate: FS as u32,
        samples: output,
    };
    write_wav("tests/output/transposed.wav", &wav).unwrap();
    eprintln!("wrote tests/output/transposed.wav");
}

#[test]
#[ignore]
fn write_time_stretch() {
    ensure_output_dir();
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    let edits: Vec<NoteEdit> = segs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SegmentKind::Note(_)))
        .map(|(i, s)| NoteEdit {
            out_len: Some((s.frames.len() as f64 * 1.5) as usize),
            ..NoteEdit::identity(i)
        })
        .collect();

    let rendered = render(&frames, &segs, &edits);
    let mut synth = Synthesizer::with_defaults(FS);
    let output = synth.synthesize_frames(&rendered);

    let wav = WavData {
        sample_rate: FS as u32,
        samples: output,
    };
    write_wav("tests/output/stretched.wav", &wav).unwrap();
    eprintln!("wrote tests/output/stretched.wav");
}

#[test]
#[ignore]
fn write_contour_csv() {
    ensure_output_dir();
    let input = synthetic_input();
    let mut analyzer = Analyzer::with_defaults(FS);
    let frames = analyzer.analyze_to_frames(&input);
    let config = SegmentConfig::default();
    let segs = segment(&frames, FRAME_RATE, &config);

    for (seg_idx, seg) in segs.iter().enumerate() {
        if let SegmentKind::Note(c) = &seg.kind {
            let mut csv = String::from("frame,original_cents,center,drift,vibrato,residual\n");
            for (j, frame_idx) in seg.frames.clone().enumerate() {
                let orig_cents = if frames[frame_idx].voiced && frames[frame_idx].fo > 0.0 {
                    1200.0 * (frames[frame_idx].fo / 440.0).log2()
                } else {
                    0.0
                };
                let vib = if j < c.vibrato_amp.len() && j < c.vibrato_phase.len() {
                    c.vibrato_amp[j] * c.vibrato_phase[j].sin()
                } else {
                    0.0
                };
                let drift = if j < c.drift.len() { c.drift[j] } else { 0.0 };
                let residual = if j < c.residual.len() {
                    c.residual[j]
                } else {
                    0.0
                };
                csv.push_str(&format!(
                    "{frame_idx},{orig_cents:.2},{:.2},{drift:.2},{vib:.2},{residual:.2}\n",
                    c.center_cents
                ));
            }
            let path = format!("tests/output/contour_{seg_idx}.csv");
            std::fs::write(&path, &csv).unwrap();
            eprintln!("wrote {path}");
        }
    }
}
