use crate::Frame;

const A4_HZ: f64 = 440.0;

pub fn hz_to_cents(hz: f64) -> f64 {
    1200.0 * (hz / A4_HZ).log2()
}

pub fn cents_to_hz(cents: f64) -> f64 {
    A4_HZ * 2.0_f64.powf(cents / 1200.0)
}

/// Median-filter the Fo contour. Unvoiced frames -> 0.0, voiced -> median-filtered Fo in Hz.
/// Kills single-frame octave jumps.
pub fn clean_contour(frames: &[Frame], median_window: usize) -> Vec<f64> {
    let half = median_window / 2;
    let mut result = vec![0.0; frames.len()];
    for i in 0..frames.len() {
        if !frames[i].voiced {
            continue;
        }
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(frames.len());
        let mut window_vals: Vec<f64> = (start..end)
            .filter(|&j| frames[j].voiced)
            .map(|j| frames[j].fo)
            .collect();
        if window_vals.is_empty() {
            continue;
        }
        window_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result[i] = window_vals[window_vals.len() / 2];
    }
    result
}
