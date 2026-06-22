//! Formant shift: analyze a WAV, warp the spectral-envelope frequency axis, and
//! resynthesize with ReIm's own synthesizer. Pitch (Fo) and aperiodicity are
//! left untouched, so only the formant structure moves -- a "smaller/larger
//! head" effect without changing the perceived pitch.
//!
//!     cargo run --release --example formant_shift -- <in.wav> <out.wav> [ratio]
//!
//! `ratio` > 1 shifts formants up (default 1.15); < 1 shifts them down.

use reim::{read_wav, write_wav_f32, Analyzer, Synthesizer};

/// Resample `src` (a per-bin envelope) onto a frequency axis scaled by `ratio`,
/// into `dst`: `dst[k] = src[k / ratio]` with linear interpolation. Bins past
/// the top of the source axis are clamped to the last bin. Allocation-free.
fn warp_formants(src: &[f64], ratio: f64, dst: &mut [f64]) {
    let last = src.len() - 1;
    for (k, d) in dst.iter_mut().enumerate() {
        let pos = (k as f64 / ratio).clamp(0.0, last as f64);
        let i = pos.floor() as usize;
        let frac = pos - i as f64;
        *d = if i >= last {
            src[last]
        } else {
            src[i] * (1.0 - frac) + src[i + 1] * frac
        };
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!("usage: {} <in.wav> <out.wav> [ratio]", args[0]));
    }
    let ratio: f64 = args.get(3).map_or(1.15, |s| s.parse().unwrap_or(1.15));

    let wav = read_wav(&args[1])?;
    let fs = wav.sample_rate as f64;

    let mut analyzer = Analyzer::with_defaults(fs);
    let mut synth = Synthesizer::with_defaults(fs);
    let mut warped = vec![0.0; analyzer.numbins()];
    let mut out = Vec::with_capacity(wav.samples.len());

    for &x in &wav.samples {
        if analyzer.push_sample(x) {
            warp_formants(analyzer.spectral_envelope(), ratio, &mut warped);
            synth.push_frame(analyzer.fo(), analyzer.voiced(), analyzer.silence(), analyzer.aperiodicity(), &warped);
        }
        out.push(synth.next_sample());
    }

    write_wav_f32(&args[2], &out, wav.sample_rate)?;
    eprintln!("wrote {} ({} samples, formant ratio {ratio})", args[2], out.len());
    Ok(())
}
