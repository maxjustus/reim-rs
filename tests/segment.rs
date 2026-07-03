use reim::segment::{clean_contour, segment, SegmentConfig, SegmentKind};
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
    let segs = segment(&frames, &SegmentConfig::default());
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
    let segs = segment(&frames, &SegmentConfig::default());
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
    let segs = segment(&frames, &SegmentConfig::default());
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
    let segs = segment(&frames, &SegmentConfig::default());
    assert_eq!(segs.len(), 3, "expected Note, Unvoiced, Note");
    assert!(matches!(segs[0].kind, SegmentKind::Note(_)));
    assert!(matches!(segs[1].kind, SegmentKind::Unvoiced));
    assert!(matches!(segs[2].kind, SegmentKind::Note(_)));
}
