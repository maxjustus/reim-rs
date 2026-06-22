//! Round-trip snapshot regression.
//!
//! The C oracle (`reim eval`) validates the analysis side (silence/voiced/Fo) and,
//! today, the synthesis waveform to the f32 noise floor. Once the aperiodicity
//! stage stops being a binary placeholder, the synthesis waveform will diverge
//! from the C by design and the oracle's waveform check no longer applies. This
//! test is the replacement net for that waveform: it freezes reim's own output on
//! a fixed synthetic signal and fails if the round trip changes unexpectedly.
//!
//! It compares against a committed golden via SNR (tolerant to floating-point
//! noise, sensitive to real behavior changes), the same methodology as the oracle.
//! Regenerate the golden deliberately after an intended synthesis change:
//!
//!     REGEN_SNAPSHOT=1 cargo test --release --test snapshot
//!
//! The analysis-side decisions stay faithful to the C, so keep using `reim eval`
//! for those; this only guards the synthesized waveform.

use reim::Reim;

const FS: f64 = 24_000.0;
const N: usize = 12_000; // 0.5 s
const MIN_SNR_DB: f64 = 120.0; // fp-noise passes (~150 dB); any real change fails hard

/// Deterministic input: a vibrato voiced tone, then a breathy (tone+noise) region
/// where binary vs continuous aperiodicity differs most, then silence.
fn synthetic_input() -> Vec<f64> {
    let mut x = vec![0.0; N];
    let mut lcg: u64 = 0x2545_F491_4F6C_DD1D; // fixed seed -> deterministic "noise"
    let mut noise = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f64 / (1u64 << 31) as f64) - 1.0 // ~[-1, 1)
    };
    let pi = std::f64::consts::PI;
    for (i, s) in x.iter_mut().enumerate() {
        let t = i as f64 / FS;
        if i < 4_000 {
            // vibrato voiced tone with a few harmonics
            let f = 180.0 * (1.0 + 0.03 * (2.0 * pi * 5.0 * t).sin());
            let ph = 2.0 * pi * f * t;
            *s = 0.5 * (ph.sin() + 0.4 * (2.0 * ph).sin() + 0.2 * (3.0 * ph).sin());
        } else if i < 8_000 {
            // breathy: a tone plus strong aperiodic energy
            let ph = 2.0 * pi * 220.0 * t;
            *s = 0.3 * ph.sin() + 0.25 * noise();
        }
        // i >= 8_000: silence
    }
    x
}

fn snr_db(golden: &[f64], test: &[f64]) -> f64 {
    let n = golden.len().min(test.len());
    let mut sig = 0.0;
    let mut err = 0.0;
    for i in 0..n {
        sig += golden[i] * golden[i];
        let d = golden[i] - test[i];
        err += d * d;
    }
    10.0 * (sig / (err + 1e-20)).log10()
}

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/roundtrip_24k.f32")
}

fn write_golden(path: &std::path::Path, out: &[f64]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut bytes = Vec::with_capacity(out.len() * 4);
    for &s in out {
        bytes.extend_from_slice(&(s as f32).to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

fn read_golden(path: &std::path::Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
}

#[test]
fn roundtrip_matches_golden() {
    let input = synthetic_input();
    let mut reim = Reim::with_defaults(FS);
    let mut out = vec![0.0; input.len()];
    reim.process_block(&input, &mut out);
    assert!(out.iter().all(|x| x.is_finite()), "round trip produced non-finite samples");

    let path = golden_path();
    if std::env::var("REGEN_SNAPSHOT").is_ok() || !path.exists() {
        write_golden(&path, &out);
        eprintln!("snapshot: wrote golden {} ({} samples)", path.display(), out.len());
        return;
    }

    let golden = read_golden(&path);
    assert_eq!(golden.len(), out.len(), "golden length changed; regenerate if intended");
    let snr = snr_db(&golden, &out);
    assert!(
        snr > MIN_SNR_DB,
        "round-trip output diverged from the golden: {snr:.1} dB SNR (< {MIN_SNR_DB} dB).\n\
         If this change to the synthesis path is intended, regenerate with:\n\
         REGEN_SNAPSHOT=1 cargo test --release --test snapshot"
    );
}
