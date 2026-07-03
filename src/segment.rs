use std::ops::Range;

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

fn make_note_contour(fo_hz: &[f64]) -> NoteContour {
    let cents: Vec<f64> = fo_hz.iter().map(|&f| hz_to_cents(f)).collect();
    let center = median_cents(&cents);
    let drift: Vec<f64> = cents.iter().map(|&c| c - center).collect();
    let len = fo_hz.len();
    NoteContour {
        center_cents: center,
        drift,
        vibrato_rate_hz: 0.0,
        vibrato_amp: vec![0.0; len],
        vibrato_phase: vec![0.0; len],
        residual: vec![0.0; len],
    }
}

pub fn segment(frames: &[Frame], config: &SegmentConfig) -> Vec<Segment> {
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
                let fo_slice: Vec<f64> = (range.start..range.end).map(|j| cleaned[j]).collect();
                segments.push(Segment {
                    frames: range,
                    kind: SegmentKind::Note(make_note_contour(&fo_slice)),
                });
            }
        }
    }

    segments
}
