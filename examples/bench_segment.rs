use std::time::Instant;

use reim::segment::{render, segment, NoteEdit, SegmentConfig};
use reim::{Analyzer, Synthesizer};

// 60 s "melody": notes of varying length with vibrato, drift, and glides
// between them — shaped like real singing so the segmenter does real work.
fn synth_input(fs: f64, seconds: f64) -> Vec<f64> {
    let n = (fs * seconds) as usize;
    let pi = std::f64::consts::PI;
    let notes: [f64; 8] = [220.0, 246.9, 261.6, 293.7, 329.6, 293.7, 261.6, 246.9];
    let note_len = 0.8; // seconds
    let glide_len = 0.12;

    let mut x = vec![0.0; n];
    let mut phase = 0.0;
    for (i, s) in x.iter_mut().enumerate() {
        let t = i as f64 / fs;
        let pos = t / note_len;
        let idx = pos as usize % notes.len();
        let frac = pos.fract();
        let f_a = notes[idx];
        let f_b = notes[(idx + 1) % notes.len()];
        // glide at the end of each note
        let base = if frac > 1.0 - glide_len / note_len {
            let g = (frac - (1.0 - glide_len / note_len)) / (glide_len / note_len);
            f_a * (f_b / f_a).powf(0.5 * (1.0 - (pi * g).cos()))
        } else {
            f_a
        };
        let vib = 1.0 + 0.006 * (2.0 * pi * 5.5 * t).sin();
        let drift = 1.0 + 0.003 * (2.0 * pi * 0.7 * t).sin();
        let f = base * vib * drift;
        phase += 2.0 * pi * f / fs;
        *s = 0.4 * (phase.sin() + 0.4 * (2.0 * phase).sin() + 0.15 * (3.0 * phase).sin());
    }
    x
}

fn main() {
    let fs = 24_000.0;
    let seconds: f64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(60.0);
    let input = synth_input(fs, seconds);

    let t0 = Instant::now();
    let mut analyzer = Analyzer::with_defaults(fs);
    let frames = analyzer.analyze_to_frames(&input);
    let t_analyze = t0.elapsed();

    let config = SegmentConfig::default();
    let t0 = Instant::now();
    let segs = segment(&frames, 200.0, &config);
    let t_segment = t0.elapsed();

    let edits: Vec<NoteEdit> = (0..segs.len()).map(NoteEdit::identity).collect();
    let t0 = Instant::now();
    let rendered = render(&frames, &segs, &edits);
    let t_render = t0.elapsed();

    let t0 = Instant::now();
    let mut synth = Synthesizer::with_defaults(fs);
    let out = synth.synthesize_frames(&rendered);
    let t_synth = t0.elapsed();

    println!(
        "input: {:.0}s audio, {} frames, {} segments",
        seconds,
        frames.len(),
        segs.len()
    );
    println!("analyze_to_frames:  {t_analyze:>10.2?}");
    println!("segment:            {t_segment:>10.2?}");
    println!("render:             {t_render:>10.2?}");
    println!("synthesize_frames:  {t_synth:>10.2?}");
    println!("output samples: {}", out.len());
}
