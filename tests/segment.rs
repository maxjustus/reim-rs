use reim::segment::{
    clean_contour, contour_svg, decompose_contour, segment, SegmentConfig, SegmentKind,
};
use reim::{Analyzer, Frame, Reim, Synthesizer};

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
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config());
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
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config());

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
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config());

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
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config());

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
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config());

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
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config());

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
    let contour = decompose_contour(&fo, FRAME_RATE, &make_config());

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
