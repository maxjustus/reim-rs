use std::f64::consts::PI;
use std::fmt::Write;
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

#[derive(Clone, Debug)]
pub struct SegmentConfig {
    pub stability_cents: f64,
    pub min_note_frames: usize,
    pub median_window: usize,
    pub drift_cutoff_hz: f64,
    pub vibrato_min_hz: f64,
    pub vibrato_max_hz: f64,
    pub glide_slope_cents_per_sec: f64,
    pub glide_min_cents: f64,
    pub max_glide_frames: usize,
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
            glide_slope_cents_per_sec: 400.0,
            glide_min_cents: 40.0,
            max_glide_frames: 60,
        }
    }
}

/// The segment covers `onset_glide.len()` glide frames followed by the core;
/// `drift`/`vibrato_*`/`residual` are core-only.
#[derive(Clone, Debug)]
pub struct NoteContour {
    pub center_cents: f64,
    /// Normalized glide shape (~0 at the previous pitch, ~1 at this note's
    /// entry pitch, may overshoot). Empty = no onset glide.
    pub onset_glide: Vec<f64>,
    pub onset_glide_depth_cents: f64,
    pub drift: Vec<f64>,
    pub vibrato_rate_hz: f64,
    pub vibrato_amp: Vec<f64>,
    pub vibrato_phase: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Clone, Debug)]
pub enum SegmentKind {
    Note(NoteContour),
    Unvoiced,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub frames: Range<usize>,
    pub kind: SegmentKind,
}

fn median_cents(vals: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

#[derive(Clone, Copy, PartialEq)]
struct TotalF64(f64);
impl Eq for TotalF64 {}
impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Incremental median over a growing set: O(log n) push. Matches
/// `median_cents` (upper median for even counts).
#[derive(Default)]
struct RunningMedian {
    lo: std::collections::BinaryHeap<TotalF64>,
    hi: std::collections::BinaryHeap<std::cmp::Reverse<TotalF64>>,
}

impl RunningMedian {
    fn push(&mut self, x: f64) {
        match self.hi.peek() {
            Some(&std::cmp::Reverse(top)) if x < top.0 => self.lo.push(TotalF64(x)),
            _ => self.hi.push(std::cmp::Reverse(TotalF64(x))),
        }
        if self.hi.len() > self.lo.len() + 1 {
            let std::cmp::Reverse(v) = self.hi.pop().unwrap();
            self.lo.push(v);
        } else if self.lo.len() > self.hi.len() {
            let v = self.lo.pop().unwrap();
            self.hi.push(std::cmp::Reverse(v));
        }
    }

    fn median(&self) -> f64 {
        self.hi.peek().unwrap().0 .0
    }

    fn clear(&mut self) {
        self.lo.clear();
        self.hi.clear();
    }
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

fn autocorrelation(signal: &[f64], max_lag: usize) -> Vec<f64> {
    let n = signal.len();
    let mut result = vec![0.0; max_lag.min(n - 1) + 1];
    for (lag, r) in result.iter_mut().enumerate() {
        let mut sum = 0.0;
        for i in 0..n - lag {
            sum += signal[i] * signal[i + lag];
        }
        *r = sum;
    }
    result
}

fn bandpass_fft(
    signal: &[f64],
    center_hz: f64,
    half_width_hz: f64,
    sample_rate: f64,
    planner: &mut FftPlanner<f64>,
) -> Vec<f64> {
    let n = signal.len();
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

fn hilbert_analytic(signal: &[f64], planner: &mut FftPlanner<f64>) -> Vec<Complex<f64>> {
    let n = signal.len();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut buf: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buf);

    for i in 1..n {
        if i < n.div_ceil(2) {
            buf[i] *= 2.0;
        } else if i > n / 2 {
            buf[i] = Complex::new(0.0, 0.0);
        }
    }

    ifft.process(&mut buf);
    for c in &mut buf {
        *c /= n as f64;
    }
    buf
}

/// A sustained, mostly-monotone pitch movement — a stair-step chunk the note
/// boundary detector carves out of a portamento. Vibrato fails the monotone
/// test (net movement ~0 over a chunk), genuine short notes fail the slope
/// test, and a note holding then gliding away fails the plateau test (its
/// median sits at the held pitch, next to an endpoint).
fn is_transition(cents: &[f64], per_frame_slope_th: f64, band: f64) -> bool {
    if cents.len() < 2 {
        return false;
    }
    let net = cents[cents.len() - 1] - cents[0];
    let mean_slope = net.abs() / (cents.len() - 1) as f64;
    if mean_slope < per_frame_slope_th {
        return false;
    }
    let m = median_cents(cents);
    if (m - cents[0]).abs() <= band || (m - cents[cents.len() - 1]).abs() <= band {
        return false;
    }
    let sum_abs: f64 = cents.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    net.abs() / sum_abs.max(1e-12) >= 0.8
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

pub fn decompose_contour(
    fo_hz: &[f64],
    frame_rate_hz: f64,
    config: &SegmentConfig,
    planner: &mut FftPlanner<f64>,
) -> NoteContour {
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

    let no_vibrato = |drift: Vec<f64>, residual: Vec<f64>| NoteContour {
        center_cents: center,
        onset_glide: Vec::new(),
        onset_glide_depth_cents: 0.0,
        drift,
        vibrato_rate_hz: 0.0,
        vibrato_amp: vec![0.0; n],
        vibrato_phase: vec![0.0; n],
        residual,
    };

    // Vibrato extraction
    let min_frames = (2.0 * frame_rate_hz / config.vibrato_min_hz).ceil() as usize;
    if n < min_frames {
        return no_vibrato(drift, detrended);
    }

    // Autocorrelation to find vibrato rate; only lags up to the slowest
    // vibrato period are ever inspected.
    let lag_min = (frame_rate_hz / config.vibrato_max_hz).floor() as usize;
    let lag_max = (frame_rate_hz / config.vibrato_min_hz).ceil() as usize;
    let lag_max = lag_max.min(n - 1);
    let acorr = autocorrelation(&detrended, lag_max);

    if acorr[0] < 1e-10 {
        return no_vibrato(drift, detrended);
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
        return no_vibrato(drift, detrended);
    }

    let vibrato_rate = frame_rate_hz / best_lag as f64;

    // Bandpass around detected rate ±1Hz, then Hilbert
    let bandpassed = bandpass_fft(&detrended, vibrato_rate, 1.0, frame_rate_hz, planner);
    let analytic = hilbert_analytic(&bandpassed, planner);

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
        onset_glide: Vec::new(),
        onset_glide_depth_cents: 0.0,
        drift,
        vibrato_rate_hz: vibrato_rate,
        vibrato_amp,
        vibrato_phase,
        residual,
    }
}

/// Fold stair-stepped transition chunks into the following range, then refine
/// each note onset into (range, onset_glide_len) pairs by expanding along
/// steep monotone slopes. Ranges stay contiguous; a glide's frames belong to
/// the following note. `cents` is the cleaned contour in cents, indexed
/// absolutely.
fn attach_glides(
    merged: &[Range<usize>],
    cents: &[f64],
    run_end: usize,
    frame_rate_hz: f64,
    config: &SegmentConfig,
) -> Vec<(Range<usize>, usize)> {
    let th = config.glide_slope_cents_per_sec / frame_rate_hz;
    let band = config.stability_cents / 2.0;
    // Centered slope at j: single-frame diffs are brittle (the median filter
    // can leave one flat pair mid-glide, which would halt expansion).
    let run_start = merged.first().map_or(0, |r| r.start);
    let slope = |j: usize| {
        let a = j.saturating_sub(1).max(run_start);
        let b = (j + 1).min(run_end - 1);
        (cents[b] - cents[a]) / (b - a).max(1) as f64
    };

    // Monotone transition chunks merge into the next stable range as its
    // pre-attached glide (a slow portamento gets carved into several
    // stair-step "notes" by the boundary detector; they are really one glide).
    let mut ranges: Vec<(Range<usize>, usize)> = Vec::new();
    let mut chain_start: Option<usize> = None;
    for range in merged {
        let len = range.end - range.start;
        if len >= config.min_note_frames && is_transition(&cents[range.start..range.end], th, band)
        {
            chain_start.get_or_insert(range.start);
            continue;
        }
        match chain_start.take() {
            Some(cs) if len >= config.min_note_frames => {
                ranges.push((cs..range.end, range.start - cs));
            }
            Some(cs) => {
                ranges.push((cs..range.start, 0));
                ranges.push((range.clone(), 0));
            }
            None => ranges.push((range.clone(), 0)),
        }
    }
    if let Some(cs) = chain_start {
        ranges.push((cs..run_end, 0));
    }

    // Refine the glide extent at each note onset. k == 0 is a scoop out of
    // silence: no previous note to expand back into, and the depth anchors
    // on the glide's own first frame instead of the previous note's last.
    for k in 0..ranges.len() {
        let (b, pre) = ranges[k].clone();
        let core_start = b.start + pre;
        if b.end - core_start < config.min_note_frames {
            continue;
        }
        let c_b = median_cents(&cents[core_start..b.end]);

        let prev = if k > 0 {
            let (a, a_glide) = ranges[k - 1].clone();
            let a_core_start = a.start + a_glide;
            if a.end - a_core_start < config.min_note_frames {
                continue;
            }
            Some((a_core_start, median_cents(&cents[a_core_start..a.end])))
        } else {
            None
        };

        let dir = match prev {
            Some((_, c_a)) => {
                if (c_b - c_a).abs() < config.glide_min_cents {
                    ranges[k].1 = 0;
                    continue;
                }
                (c_b - c_a).signum()
            }
            None => (c_b - cents[b.start]).signum(),
        };

        let mut g_start = b.start;
        let mut g_end = core_start;
        if let Some((a_core_start, c_a)) = prev {
            while g_start > a_core_start + config.min_note_frames
                && g_end - g_start < config.max_glide_frames
                && dir * slope(g_start - 1) >= th
                && dir * (cents[g_start - 1] - c_a) > band
            {
                g_start -= 1;
            }
        }
        while g_end < b.end - config.min_note_frames
            && g_end - g_start < config.max_glide_frames
            && dir * slope(g_end) >= th
            && dir * (c_b - cents[g_end]) > band
        {
            g_end += 1;
        }
        // A pre-attached chain can exceed the cap; trim from the start so the
        // excess frames stay with the previous note.
        if prev.is_some() && g_end - g_start > config.max_glide_frames {
            g_start = g_end - config.max_glide_frames;
        }

        let depth_anchor = if prev.is_some() {
            cents[g_start - 1]
        } else {
            cents[g_start]
        };
        let depth = dir * (cents[g_end] - depth_anchor);
        if g_end - g_start >= 2 && depth >= config.glide_min_cents {
            if prev.is_some() {
                ranges[k - 1].0.end = g_start;
                ranges[k].0.start = g_start;
            }
            ranges[k].1 = g_end - g_start;
        } else {
            ranges[k].1 = 0;
        }
    }

    ranges
}

pub fn segment(frames: &[Frame], frame_rate_hz: f64, config: &SegmentConfig) -> Vec<Segment> {
    if frames.is_empty() {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f64>::new();
    let cleaned = clean_contour(frames, config.median_window);
    // Cleaned contour in cents; 0.0 for unvoiced frames (never read there).
    let cents: Vec<f64> = cleaned
        .iter()
        .map(|&h| if h > 0.0 { hz_to_cents(h) } else { 0.0 })
        .collect();

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

        // Boundary detection runs on a vibrato-suppressed contour: vibrato is
        // a zero-mean oscillation at vibrato_min_hz+, so lowpassing below it
        // leaves only actual pitch movement for the stability test to see.
        let sigma = frame_rate_hz / (2.0 * PI * config.drift_cutoff_hz);
        let run_cents = gaussian_lowpass(&cents[run.start..run.end], sigma);

        // Detect note boundaries within this voiced run.
        let mut note_boundaries: Vec<usize> = vec![0]; // offsets within the run
        let mut current_center = run_cents[0];
        let mut departed_count = 0;
        let mut first_departed = 0;

        // Track running median of the current note for center pitch.
        let mut current_note_median = RunningMedian::default();
        current_note_median.push(run_cents[0]);

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
                    current_note_median.clear();
                    for k in first_departed..=j {
                        current_note_median.push(run_cents[k]);
                    }
                    current_center = current_note_median.median();
                    departed_count = 0;
                }
            } else {
                departed_count = 0;
                current_note_median.push(run_cents[j]);
                current_center = current_note_median.median();
            }
        }

        // The lowpass smears a pitch step over ~±3σ, so the threshold fires
        // late; snap each boundary to the steepest smoothed slope nearby
        // (Gaussian smoothing is zero-phase, so that's the transition center).
        // Stair-step boundaries carved out of one smeared transition all snap
        // to the same steepest frame and dedup away.
        let radius = (3.0 * sigma).ceil() as usize;
        for b in note_boundaries.iter_mut().skip(1) {
            let lo = (*b).saturating_sub(radius).max(1);
            let hi = (*b + radius).min(run_cents.len() - 1);
            *b = (lo..=hi)
                .max_by(|&x, &y| {
                    let dx = (run_cents[x] - run_cents[x - 1]).abs();
                    let dy = (run_cents[y] - run_cents[y - 1]).abs();
                    dx.total_cmp(&dy)
                })
                .unwrap();
        }
        note_boundaries.sort_unstable();
        note_boundaries.dedup();

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

        let ranges = attach_glides(&merged, &cents, run.end, frame_rate_hz, config);

        for (range, glide_len) in ranges {
            let len = range.end - range.start;
            if len < config.min_note_frames {
                segments.push(Segment {
                    frames: range,
                    kind: SegmentKind::Unvoiced,
                });
                continue;
            }
            let core_start = range.start + glide_len;
            let fo_slice: Vec<f64> = (core_start..range.end).map(|j| frames[j].fo).collect();
            let mut nc = decompose_contour(&fo_slice, frame_rate_hz, config, &mut planner);
            if glide_len > 0 {
                // Scoop from silence anchors on its own first frame; a glide
                // between notes anchors on the previous note's last frame.
                let anchor_start = if range.start > 0 && frames[range.start - 1].voiced {
                    hz_to_cents(frames[range.start - 1].fo)
                } else {
                    hz_to_cents(frames[range.start].fo)
                };
                let anchor_end = hz_to_cents(frames[core_start].fo);
                let depth = anchor_end - anchor_start;
                nc.onset_glide = if depth.abs() < 1e-6 {
                    (0..glide_len)
                        .map(|j| j as f64 / glide_len as f64)
                        .collect()
                } else {
                    (range.start..core_start)
                        .map(|j| (hz_to_cents(frames[j].fo) - anchor_start) / depth)
                        .collect()
                };
                nc.onset_glide_depth_cents = depth;
            }
            segments.push(Segment {
                frames: range,
                kind: SegmentKind::Note(nc),
            });
        }
    }

    segments
}

#[derive(Clone, Debug)]
pub struct NoteEdit {
    pub segment_index: usize,
    pub target_cents: Option<f64>,
    pub drift_scale: f64,
    pub vibrato_scale: f64,
    pub vibrato_rate_scale: f64,
    /// Scales glide duration; 0.0 drops the glide frames (note shortens).
    pub glide_time_scale: f64,
    pub out_len: Option<usize>,
    /// Silence frames rendered before the segment (repositioning gap). A gap
    /// severs the note from its predecessor, so an onset glide anchors on its
    /// own depth (scoop) instead of retargeting to the previous exit pitch.
    pub lead_gap_frames: usize,
}

impl NoteEdit {
    pub fn identity(segment_index: usize) -> Self {
        NoteEdit {
            segment_index,
            target_cents: None,
            drift_scale: 1.0,
            vibrato_scale: 1.0,
            vibrato_rate_scale: 1.0,
            glide_time_scale: 1.0,
            out_len: None,
            lead_gap_frames: 0,
        }
    }
}

fn lerp_track(track: &[f64], s: f64) -> f64 {
    if track.is_empty() {
        return 0.0;
    }
    let idx = (s.floor() as usize).min(track.len() - 1);
    let idx1 = (idx + 1).min(track.len() - 1);
    let t = s - s.floor();
    track[idx] * (1.0 - t) + track[idx1] * t
}

/// Interpolate spectral data between two source frames; fo is supplied by the
/// caller's contour model. Spectral envelope in log power, aperiodicity linear.
fn interp_frame(fa: &Frame, fb: &Frame, t: f64, fo: f64) -> Frame {
    // Identity/on-grid resampling lands exactly on a source frame.
    if t == 0.0 || std::ptr::eq(fa, fb) {
        return Frame { fo, ..fa.clone() };
    }
    let floor = 1e-16_f64;
    let spectral_envelope: Vec<f64> = fa
        .spectral_envelope
        .iter()
        .zip(fb.spectral_envelope.iter())
        .map(|(&a, &b)| {
            let la = a.max(floor).ln();
            let lb = b.max(floor).ln();
            (la * (1.0 - t) + lb * t).exp()
        })
        .collect();
    let aperiodicity: Vec<f64> = fa
        .aperiodicity
        .iter()
        .zip(fb.aperiodicity.iter())
        .map(|(&a, &b)| a * (1.0 - t) + b * t)
        .collect();
    let nearest = if t < 0.5 { fa } else { fb };
    Frame {
        fo,
        voiced: nearest.voiced,
        silence: nearest.silence,
        aperiodicity,
        spectral_envelope,
    }
}

/// Source-track position for output frame j when resampling src_len frames
/// to out_len frames.
fn src_pos(j: usize, src_len: usize, out_len: usize) -> f64 {
    if out_len <= 1 {
        0.0
    } else {
        j as f64 * (src_len - 1) as f64 / (out_len - 1) as f64
    }
}

/// (glide_out, core_out) frame counts for a `Note` segment under `edit`, or
/// `None` when the edit drops the whole segment (`out_len: Some(0)`). Shared
/// by `render()` and [`segment_output_len`] so the two can't drift apart.
fn note_render_lengths(
    src_len: usize,
    glide_len: usize,
    edit: &NoteEdit,
) -> Option<(usize, usize)> {
    // Melodyne's transition speed is timing-neutral: the segment's total length
    // is fixed by out_len (or unchanged); glide_time_scale only redistributes
    // frames between the glide and the core.
    let total = match edit.out_len {
        Some(0) => return None,
        Some(x) => x,
        None => src_len,
    };
    let stretch = total as f64 / src_len as f64;
    let glide_out =
        ((glide_len as f64 * edit.glide_time_scale * stretch).round() as usize).min(total - 1);
    Some((glide_out, total - glide_out))
}

/// Number of output frames [`render`] will produce for `seg` given `edit`
/// (`None` = identity edit). Lets callers locate a segment's span within
/// `render`'s flat output without re-deriving its length arithmetic.
pub fn segment_output_len(seg: &Segment, edit: Option<&NoteEdit>) -> usize {
    let src_len = seg.frames.end - seg.frames.start;
    match &seg.kind {
        SegmentKind::Unvoiced => edit.and_then(|e| e.out_len).unwrap_or(src_len),
        SegmentKind::Note(nc) => {
            let identity;
            let edit = match edit {
                Some(e) => e,
                None => {
                    identity = NoteEdit::identity(0);
                    &identity
                }
            };
            let glide_len = nc.onset_glide.len();
            note_render_lengths(src_len, glide_len, edit)
                .map(|(glide_out, core_out)| edit.lead_gap_frames + glide_out + core_out)
                .unwrap_or(0)
        }
    }
}

pub fn render(frames: &[Frame], segments: &[Segment], edits: &[NoteEdit]) -> Vec<Frame> {
    let mut output = Vec::new();
    let mut last_voiced_exit_cents: Option<f64> = None;

    for (seg_idx, seg) in segments.iter().enumerate() {
        let edit = edits.iter().find(|e| e.segment_index == seg_idx);
        let src_len = seg.frames.end - seg.frames.start;

        match &seg.kind {
            SegmentKind::Unvoiced => {
                last_voiced_exit_cents = None;
                let out_len = edit.and_then(|e| e.out_len).unwrap_or(src_len);
                for j in 0..out_len {
                    let s = src_pos(j, src_len, out_len);
                    let idx = (s.floor() as usize).min(src_len - 1);
                    let abs_idx = seg.frames.start + idx;
                    output.push(frames[abs_idx].clone());
                }
            }
            SegmentKind::Note(nc) => {
                let identity;
                let edit = match edit {
                    Some(e) => e,
                    None => {
                        identity = NoteEdit::identity(seg_idx);
                        &identity
                    }
                };
                let glide_len = nc.onset_glide.len();
                let core_len = src_len - glide_len;
                let Some((glide_out, core_out)) = note_render_lengths(src_len, glide_len, edit)
                else {
                    continue;
                };

                if edit.lead_gap_frames > 0 {
                    let gap_frame = Frame {
                        fo: 0.0,
                        voiced: false,
                        silence: true,
                        ..frames[seg.frames.start].clone()
                    };
                    output.extend(std::iter::repeat_n(gap_frame, edit.lead_gap_frames));
                    last_voiced_exit_cents = None;
                }

                let center = edit.target_cents.unwrap_or(nc.center_cents);
                let entry_cents = center
                    + edit.drift_scale * lerp_track(&nc.drift, 0.0)
                    + edit.vibrato_scale
                        * lerp_track(&nc.vibrato_amp, 0.0)
                        * lerp_track(&nc.vibrato_phase, 0.0).sin()
                    + edit.drift_scale * lerp_track(&nc.residual, 0.0);

                if glide_len > 0 && glide_out > 0 {
                    // Retarget: connect the previous note's actual exit pitch
                    // to this note's (post-edit) entry pitch along the stored
                    // normalized glide shape.
                    let start_cents =
                        last_voiced_exit_cents.unwrap_or(entry_cents - nc.onset_glide_depth_cents);
                    for j in 0..glide_out {
                        let s = src_pos(j, glide_len, glide_out);
                        let shape = lerp_track(&nc.onset_glide, s);
                        let cents = start_cents + shape * (entry_cents - start_cents);
                        let idx = (s.floor() as usize).min(glide_len - 1);
                        let idx1 = (idx + 1).min(glide_len - 1);
                        output.push(interp_frame(
                            &frames[seg.frames.start + idx],
                            &frames[seg.frames.start + idx1],
                            s - s.floor(),
                            cents_to_hz(cents),
                        ));
                    }
                }

                let core_origin = seg.frames.start + glide_len;
                let mut phi = lerp_track(&nc.vibrato_phase, 0.0);
                let mut prev_s_phase = phi;
                let mut last_cents = entry_cents;

                for j in 0..core_out {
                    let s = src_pos(j, core_len, core_out);
                    let idx = (s.floor() as usize).min(core_len - 1);
                    let idx1 = (idx + 1).min(core_len - 1);

                    let cur_phase = lerp_track(&nc.vibrato_phase, s);
                    if j == 0 {
                        phi = cur_phase;
                    } else {
                        phi += edit.vibrato_rate_scale * (cur_phase - prev_s_phase);
                    }
                    prev_s_phase = cur_phase;

                    let drift = lerp_track(&nc.drift, s);
                    let vib_amp = lerp_track(&nc.vibrato_amp, s);
                    let residual = lerp_track(&nc.residual, s);

                    let cents_out = center
                        + edit.drift_scale * drift
                        + edit.vibrato_scale * vib_amp * phi.sin()
                        + edit.drift_scale * residual;
                    last_cents = cents_out;

                    output.push(interp_frame(
                        &frames[core_origin + idx],
                        &frames[core_origin + idx1],
                        s - s.floor(),
                        cents_to_hz(cents_out),
                    ));
                }

                last_voiced_exit_cents = Some(last_cents);
            }
        }
    }

    output
}

fn note_name(cents: f64) -> String {
    let midi = (69.0 + cents / 100.0).round() as i32;
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!(
        "{}{}",
        NAMES[midi.rem_euclid(12) as usize],
        midi.div_euclid(12) - 1
    )
}

pub fn variance_explained(contour: &NoteContour) -> (f64, f64, f64) {
    let n = contour.drift.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let total_var: f64 = contour
        .drift
        .iter()
        .zip(contour.vibrato_amp.iter().zip(contour.vibrato_phase.iter()))
        .zip(contour.residual.iter())
        .map(|((d, (a, p)), r)| {
            let v = d + a * p.sin() + r;
            v * v
        })
        .sum::<f64>()
        / n as f64;
    if total_var < 1e-10 {
        return (0.0, 0.0, 0.0);
    }
    let drift_var = contour.drift.iter().map(|d| d * d).sum::<f64>() / n as f64;
    let vib_var = contour
        .vibrato_amp
        .iter()
        .zip(contour.vibrato_phase.iter())
        .map(|(a, p)| {
            let v = a * p.sin();
            v * v
        })
        .sum::<f64>()
        / n as f64;
    let res_var = contour.residual.iter().map(|r| r * r).sum::<f64>() / n as f64;
    let sum = drift_var + vib_var + res_var;
    (drift_var / sum, vib_var / sum, res_var / sum)
}

pub fn contour_svg(
    frames: &[Frame],
    segments: &[Segment],
    frame_rate_hz: f64,
    overlay: Option<&[f64]>,
) -> String {
    let (w, h) = (1200.0_f64, 600.0_f64);
    let (pl, pr, pt, pb) = (60.0, 20.0, 20.0, 40.0);
    let (pw, ph) = (w - pl - pr, h - pt - pb);

    // Y-axis range from voiced frames
    let mut c_min = f64::INFINITY;
    let mut c_max = f64::NEG_INFINITY;
    for f in frames {
        if f.voiced && f.fo > 0.0 {
            let c = hz_to_cents(f.fo);
            c_min = c_min.min(c);
            c_max = c_max.max(c);
        }
    }
    if !c_min.is_finite() {
        c_min = -1200.0;
        c_max = 1200.0;
    }
    c_min -= 200.0;
    c_max += 200.0;
    let x_max = if frames.is_empty() {
        1.0
    } else {
        frames.len() as f64 / frame_rate_hz
    };

    let px = |t: f64| pl + t / x_max * pw;
    let py = |c: f64| pt + (c_max - c) / (c_max - c_min) * ph;

    let mut s = String::with_capacity(16384);
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" \
         viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">\n\
         <rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n"
    );

    // 1. Unvoiced shading + segment boundaries
    for seg in segments {
        let x0 = px(seg.frames.start as f64 / frame_rate_hz);
        let x1 = px(seg.frames.end as f64 / frame_rate_hz);
        if matches!(seg.kind, SegmentKind::Unvoiced) {
            let _ = writeln!(
                s,
                "<rect x=\"{x0:.1}\" y=\"{pt:.1}\" width=\"{:.1}\" \
                 height=\"{ph:.1}\" fill=\"#eee\"/>",
                x1 - x0
            );
        }
        let _ = writeln!(
            s,
            "<line x1=\"{x0:.1}\" y1=\"{pt:.1}\" x2=\"{x0:.1}\" y2=\"{:.1}\" \
             stroke=\"#999\" stroke-dasharray=\"4,4\"/>",
            pt + ph
        );
    }
    if let Some(last) = segments.last() {
        let x = px(last.frames.end as f64 / frame_rate_hz);
        let _ = writeln!(
            s,
            "<line x1=\"{x:.1}\" y1=\"{pt:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" \
             stroke=\"#999\" stroke-dasharray=\"4,4\"/>",
            pt + ph
        );
    }

    // 2. Semitone gridlines with note names
    let first = (c_min / 100.0).ceil() as i32;
    let last = (c_max / 100.0).floor() as i32;
    for semi in first..=last {
        let c = semi as f64 * 100.0;
        let y = py(c);
        let _ = write!(
            s,
            "<line x1=\"{pl:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#ddd\"/>\n\
             <text x=\"{:.0}\" y=\"{y:.1}\" font-size=\"9\" fill=\"#666\" \
             text-anchor=\"end\" dominant-baseline=\"middle\">{}</text>\n",
            pl + pw,
            pl - 4.0,
            note_name(c)
        );
    }

    // 3. Raw Fo dots
    for (i, f) in frames.iter().enumerate() {
        if f.voiced && f.fo > 0.0 {
            let (x, y) = (px(i as f64 / frame_rate_hz), py(hz_to_cents(f.fo)));
            let _ = writeln!(
                s,
                "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"2\" fill=\"#69b\"/>"
            );
        }
    }

    // 4-6. Per-note contour lines
    let mut pts = String::new();
    for seg in segments {
        let SegmentKind::Note(ref nc) = seg.kind else {
            continue;
        };
        let core_start = seg.frames.start + nc.onset_glide.len();
        let t0 = core_start as f64 / frame_rate_hz;
        let t1 = seg.frames.end.saturating_sub(1).max(core_start) as f64 / frame_rate_hz;

        // 4. Center line (green)
        let yc = py(nc.center_cents);
        let _ = writeln!(
            s,
            "<line x1=\"{:.1}\" y1=\"{yc:.1}\" x2=\"{:.1}\" y2=\"{yc:.1}\" \
             stroke=\"#2a2\" stroke-width=\"1.5\"/>",
            px(t0),
            px(t1)
        );

        // 5. Center + drift (orange polyline)
        pts.clear();
        for (j, d) in nc.drift.iter().enumerate() {
            let _ = write!(
                pts,
                "{:.1},{:.1} ",
                px((core_start + j) as f64 / frame_rate_hz),
                py(nc.center_cents + d)
            );
        }
        let _ = writeln!(
            s,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"#c80\" stroke-width=\"1.2\"/>",
            pts.trim_end()
        );

        // 6. Full reconstruction (red polyline)
        pts.clear();
        for j in 0..nc.drift.len() {
            let c = nc.center_cents + nc.drift[j] + nc.vibrato_amp[j] * nc.vibrato_phase[j].sin();
            let _ = write!(
                pts,
                "{:.1},{:.1} ",
                px((core_start + j) as f64 / frame_rate_hz),
                py(c)
            );
        }
        let _ = writeln!(
            s,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"#d22\" stroke-width=\"1\"/>",
            pts.trim_end()
        );

        // 6b. Onset glide reconstruction (magenta)
        if !nc.onset_glide.is_empty() {
            let anchor_end = nc.center_cents
                + nc.drift.first().copied().unwrap_or(0.0)
                + nc.vibrato_amp.first().copied().unwrap_or(0.0)
                    * nc.vibrato_phase.first().copied().unwrap_or(0.0).sin()
                + nc.residual.first().copied().unwrap_or(0.0);
            let anchor_start = anchor_end - nc.onset_glide_depth_cents;
            pts.clear();
            for (j, sh) in nc.onset_glide.iter().enumerate() {
                let c = anchor_start + sh * nc.onset_glide_depth_cents;
                let _ = write!(
                    pts,
                    "{:.1},{:.1} ",
                    px((seg.frames.start + j) as f64 / frame_rate_hz),
                    py(c)
                );
            }
            let _ = writeln!(
                s,
                "<polyline points=\"{}\" fill=\"none\" stroke=\"#d2a\" stroke-width=\"1.5\"/>",
                pts.trim_end()
            );
        }
    }

    // 7. Overlay track (purple)
    if let Some(ov) = overlay {
        pts.clear();
        for (i, &v) in ov.iter().enumerate() {
            if v > 0.0 {
                let _ = write!(
                    pts,
                    "{:.1},{:.1} ",
                    px(i as f64 / frame_rate_hz),
                    py(hz_to_cents(v))
                );
            }
        }
        let trimmed = pts.trim_end();
        if !trimmed.is_empty() {
            let _ = writeln!(
                s,
                "<polyline points=\"{trimmed}\" fill=\"none\" stroke=\"#82d\" stroke-width=\"1\"/>"
            );
        }
    }

    // X-axis time labels
    let step = if x_max <= 1.0 {
        0.1
    } else if x_max <= 5.0 {
        0.5
    } else if x_max <= 20.0 {
        1.0
    } else {
        5.0
    };
    let mut t = 0.0;
    while t <= x_max + 1e-9 {
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.0}\" font-size=\"10\" fill=\"#666\" \
             text-anchor=\"middle\">{t:.1}s</text>",
            px(t),
            h - 5.0
        );
        t += step;
    }

    s.push_str("</svg>\n");
    s
}
