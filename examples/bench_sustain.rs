use std::time::Instant;

use reim::segment::{segment, SegmentConfig};
use reim::Frame;

// Worst case: one long sustained note with vibrato — the boundary detector's
// running median and decompose's autocorrelation both see n = all frames.
fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(24_000); // 2 minutes at 200 Hz frame rate
    let pi = std::f64::consts::PI;
    let frames: Vec<Frame> = (0..n)
        .map(|i| {
            let t = i as f64 / 200.0;
            let vib = 30.0 * (2.0 * pi * 5.5 * t).sin();
            Frame {
                fo: 220.0 * 2.0_f64.powf(vib / 1200.0),
                voiced: true,
                silence: false,
                voicing_score: 1.0,
                aperiodicity: vec![0.1; 8],
                spectral_envelope: vec![1.0; 8],
            }
        })
        .collect();
    let t0 = Instant::now();
    let segs = segment(&frames, 200.0, &SegmentConfig::default());
    println!(
        "{} frames, {} segments: {:.2?}",
        n,
        segs.len(),
        t0.elapsed()
    );
}
