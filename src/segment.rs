use std::f64::consts::PI;
use std::ops::Range;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

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

pub struct SegmentConfig {
    pub stability_cents: f64,
    pub min_note_frames: usize,
    pub median_window: usize,
    pub drift_cutoff_hz: f64,
    pub vibrato_min_hz: f64,
    pub vibrato_max_hz: f64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        SegmentConfig {
            stability_cents: 50.0,
            min_note_frames: 6,
            median_window: 5,
            drift_cutoff_hz: 2.0,
            vibrato_min_hz: 4.0,
            vibrato_max_hz: 8.0,
        }
    }
}

pub struct NoteContour {
    pub center_cents: f64,
    pub drift: Vec<f64>,
    pub vibrato_rate_hz: f64,
    pub vibrato_amp: Vec<f64>,
    pub vibrato_phase: Vec<f64>,
    pub residual: Vec<f64>,
}

pub enum SegmentKind {
    Note(NoteContour),
    Unvoiced,
}

pub struct Segment {
    pub frames: Range<usize>,
    pub kind: SegmentKind,
}

fn median_cents(vals: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn gaussian_lowpass(signal: &[f64], sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    if radius == 0 || signal.len() <= 1 {
        return signal.to_vec();
    }

    // Build normalized Gaussian kernel
    let kernel_len = 2 * radius + 1;
    let mut kernel = vec![0.0; kernel_len];
    let mut sum = 0.0;
    for i in 0..kernel_len {
        let x = i as f64 - radius as f64;
        let val = (-0.5 * (x / sigma).powi(2)).exp();
        kernel[i] = val;
        sum += val;
    }
    for k in &mut kernel {
        *k /= sum;
    }

    // Reflect-pad the signal
    let n = signal.len();
    let mut padded = vec![0.0; n + 2 * radius];
    for i in 0..radius {
        // Reflect: index radius-1-i maps to signal[radius-i] clamped
        padded[i] = signal[(radius - i).min(n - 1)];
    }
    padded[radius..radius + n].copy_from_slice(signal);
    for i in 0..radius {
        padded[radius + n + i] = signal[n.saturating_sub(2 + i)];
    }

    // Convolve
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..kernel_len {
            acc += padded[i + j] * kernel[j];
        }
        out[i] = acc;
    }
    out
}

fn autocorrelation(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut result = vec![0.0; n];
    for lag in 0..n {
        let mut sum = 0.0;
        for i in 0..n - lag {
            sum += signal[i] * signal[i + lag];
        }
        result[lag] = sum;
    }
    result
}

fn bandpass_fft(signal: &[f64], center_hz: f64, half_width_hz: f64, sample_rate: f64) -> Vec<f64> {
    let n = signal.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut buf: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buf);

    let lo = center_hz - half_width_hz;
    let hi = center_hz + half_width_hz;

    for (i, c) in buf.iter_mut().enumerate() {
        let freq = if i <= n / 2 {
            i as f64 * sample_rate / n as f64
        } else {
            (n - i) as f64 * sample_rate / n as f64
        };
        if freq < lo || freq > hi {
            *c = Complex::new(0.0, 0.0);
        }
    }

    ifft.process(&mut buf);
    buf.iter().map(|c| c.re / n as f64).collect()
}

fn hilbert_analytic(signal: &[f64]) -> Vec<Complex<f64>> {
    let n = signal.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut buf: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buf);

    // Zero negative frequencies, double positive, keep DC and Nyquist
    // DC (index 0): unchanged
    // Positive frequencies: indices 1..n/2 -> multiply by 2
    // Nyquist (index n/2 if n is even): unchanged
    // Negative frequencies: indices n/2+1..n -> zero
    for i in 1..n {
        if i < n.div_ceil(2) {
            buf[i] *= 2.0;
        } else if i > n / 2 {
            buf[i] = Complex::new(0.0, 0.0);
        }
        // i == n/2 (Nyquist for even n): unchanged
    }

    ifft.process(&mut buf);
    for c in &mut buf {
        *c /= n as f64;
    }
    buf
}

fn unwrap_phase(phase: &mut [f64]) {
    for i in 1..phase.len() {
        let mut diff = phase[i] - phase[i - 1];
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }
        phase[i] = phase[i - 1] + diff;
    }
}

pub fn decompose_contour(fo_hz: &[f64], frame_rate_hz: f64, config: &SegmentConfig) -> NoteContour {
    let n = fo_hz.len();
    let cents: Vec<f64> = fo_hz.iter().map(|&f| hz_to_cents(f)).collect();

    // Center: median
    let center = median_cents(&cents);

    // Drift: Gaussian lowpass of (cents - center)
    let deviation: Vec<f64> = cents.iter().map(|&c| c - center).collect();
    let sigma = frame_rate_hz / (2.0 * PI * config.drift_cutoff_hz);
    let drift = gaussian_lowpass(&deviation, sigma);

    // Detrended signal
    let detrended: Vec<f64> = (0..n).map(|i| deviation[i] - drift[i]).collect();

    // Vibrato extraction
    let min_frames = (2.0 * frame_rate_hz / config.vibrato_min_hz).ceil() as usize;
    if n < min_frames {
        let residual = detrended;
        return NoteContour {
            center_cents: center,
            drift,
            vibrato_rate_hz: 0.0,
            vibrato_amp: vec![0.0; n],
            vibrato_phase: vec![0.0; n],
            residual,
        };
    }

    // Autocorrelation to find vibrato rate
    let acorr = autocorrelation(&detrended);
    let lag_min = (frame_rate_hz / config.vibrato_max_hz).floor() as usize;
    let lag_max = (frame_rate_hz / config.vibrato_min_hz).ceil() as usize;
    let lag_max = lag_max.min(n - 1);

    if acorr[0] < 1e-10 {
        return NoteContour {
            center_cents: center,
            drift,
            vibrato_rate_hz: 0.0,
            vibrato_amp: vec![0.0; n],
            vibrato_phase: vec![0.0; n],
            residual: detrended,
        };
    }

    let threshold = 0.3 * acorr[0];
    let mut best_lag = 0;
    let mut best_val = f64::NEG_INFINITY;

    if lag_min <= lag_max {
        for lag in lag_min..=lag_max {
            if acorr[lag] > best_val {
                best_val = acorr[lag];
                best_lag = lag;
            }
        }
    }

    if best_val <= threshold || best_lag == 0 {
        // No vibrato detected
        return NoteContour {
            center_cents: center,
            drift,
            vibrato_rate_hz: 0.0,
            vibrato_amp: vec![0.0; n],
            vibrato_phase: vec![0.0; n],
            residual: detrended,
        };
    }

    let vibrato_rate = frame_rate_hz / best_lag as f64;

    // Bandpass around detected rate ±1Hz, then Hilbert
    let bandpassed = bandpass_fft(&detrended, vibrato_rate, 1.0, frame_rate_hz);
    let analytic = hilbert_analytic(&bandpassed);

    let vibrato_amp: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();
    // Shift phase by pi/2 so that amp * sin(phase) = real part of analytic signal
    // (the real part is amp * cos(atan2(im,re)), and sin(x + pi/2) = cos(x))
    let mut vibrato_phase: Vec<f64> = analytic
        .iter()
        .map(|c| c.im.atan2(c.re) + PI / 2.0)
        .collect();
    unwrap_phase(&mut vibrato_phase);

    // Residual: original deviation - drift - vibrato reconstruction
    let residual: Vec<f64> = (0..n)
        .map(|i| deviation[i] - drift[i] - vibrato_amp[i] * vibrato_phase[i].sin())
        .collect();

    NoteContour {
        center_cents: center,
        drift,
        vibrato_rate_hz: vibrato_rate,
        vibrato_amp,
        vibrato_phase,
        residual,
    }
}

pub fn segment(frames: &[Frame], frame_rate_hz: f64, config: &SegmentConfig) -> Vec<Segment> {
    if frames.is_empty() {
        return Vec::new();
    }

    let cleaned = clean_contour(frames, config.median_window);

    // Split into voiced/unvoiced runs.
    struct Run {
        start: usize,
        end: usize,
        voiced: bool,
    }

    let mut runs: Vec<Run> = Vec::new();
    let mut i = 0;
    while i < frames.len() {
        let voiced = cleaned[i] > 0.0;
        let start = i;
        while i < frames.len() && (cleaned[i] > 0.0) == voiced {
            i += 1;
        }
        runs.push(Run {
            start,
            end: i,
            voiced,
        });
    }

    // For each voiced run, detect note boundaries. Unvoiced runs pass through.
    let mut segments: Vec<Segment> = Vec::new();

    for run in &runs {
        if !run.voiced {
            segments.push(Segment {
                frames: run.start..run.end,
                kind: SegmentKind::Unvoiced,
            });
            continue;
        }

        let run_cents: Vec<f64> = (run.start..run.end)
            .map(|j| hz_to_cents(cleaned[j]))
            .collect();

        // Detect note boundaries within this voiced run.
        let mut note_boundaries: Vec<usize> = vec![0]; // offsets within the run
        let mut current_center = run_cents[0];
        let mut departed_count = 0;
        let mut first_departed = 0;

        // Track running median of the current note for center pitch.
        let mut current_note_cents: Vec<f64> = vec![run_cents[0]];

        for j in 1..run_cents.len() {
            let diff = (run_cents[j] - current_center).abs();
            if diff > config.stability_cents {
                if departed_count == 0 {
                    first_departed = j;
                }
                departed_count += 1;
                if departed_count >= config.min_note_frames {
                    // New note starts at first_departed.
                    note_boundaries.push(first_departed);
                    // Reset center to the median of the new note's frames so far.
                    current_note_cents.clear();
                    for k in first_departed..=j {
                        current_note_cents.push(run_cents[k]);
                    }
                    current_center = median_cents(&current_note_cents);
                    departed_count = 0;
                }
            } else {
                departed_count = 0;
                current_note_cents.push(run_cents[j]);
                current_center = median_cents(&current_note_cents);
            }
        }

        // Convert note boundaries to segments, folding short notes.
        let mut note_ranges: Vec<Range<usize>> = Vec::new();
        for b in 0..note_boundaries.len() {
            let start = note_boundaries[b] + run.start;
            let end = if b + 1 < note_boundaries.len() {
                note_boundaries[b + 1] + run.start
            } else {
                run.end
            };
            note_ranges.push(start..end);
        }

        // Fold short notes into adjacent notes.
        let mut merged: Vec<Range<usize>> = Vec::new();
        for range in note_ranges {
            let len = range.end - range.start;
            if len < config.min_note_frames {
                if let Some(prev) = merged.last_mut() {
                    prev.end = range.end;
                } else {
                    // Will try to attach to next; park it for now.
                    merged.push(range);
                }
            } else {
                // If the previous entry was short (parked), merge it into this one.
                if let Some(prev) = merged.last() {
                    if prev.end - prev.start < config.min_note_frames {
                        let prev_start = prev.start;
                        merged.last_mut().unwrap().start = prev_start;
                        merged.last_mut().unwrap().end = range.end;
                        continue;
                    }
                }
                merged.push(range);
            }
        }

        // Any remaining short notes at the end: fold into previous if possible,
        // otherwise mark unvoiced.
        if merged.len() > 1 {
            let last_len = merged.last().unwrap().end - merged.last().unwrap().start;
            if last_len < config.min_note_frames {
                let last_end = merged.last().unwrap().end;
                let len = merged.len();
                merged[len - 2].end = last_end;
                merged.pop();
            }
        }

        for range in merged {
            let len = range.end - range.start;
            if len < config.min_note_frames {
                segments.push(Segment {
                    frames: range,
                    kind: SegmentKind::Unvoiced,
                });
            } else {
                let fo_slice: Vec<f64> = (range.start..range.end).map(|j| frames[j].fo).collect();
                segments.push(Segment {
                    frames: range,
                    kind: SegmentKind::Note(decompose_contour(&fo_slice, frame_rate_hz, config)),
                });
            }
        }
    }

    segments
}
