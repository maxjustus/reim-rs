//! Command-line front end for the `reim` vocoder library.
//!
//!   reim process <in.wav> <out.wav>           analyze + resynthesize a mono WAV
//!   reim eval <ref.wav> <in.wav> [feat.csv]   compare output against a reference
//!   reim bench [in.wav]                        throughput + per-stage latency
//!   reim f0 <in.wav> [fmin] [fmax] [fftsize]   emit the per-frame Fo contour as CSV

use reim::{default_fftsize, read_wav, write_wav_f32, Reim};

// Optional, EXPERIMENTAL: enable the voicing periodicity gate from the CLI via the
// REIM_VOICING_SCORE_MIN env var — a fused voicing probability threshold in (0,1),
// e.g. =0.5. Default off. See README "Voicing".
fn apply_voicing_env(reim: &mut Reim) {
    if let Some(x) = std::env::var("REIM_VOICING_SCORE_MIN")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        reim.set_voicing_score_min(x);
    }
    // REIM_SOFT_VOICING=1: continuous voicing strength at synthesis (see
    // Reim::set_soft_voicing). Default off.
    if std::env::var("REIM_SOFT_VOICING").as_deref() == Ok("1") {
        reim.set_soft_voicing(true);
    }
}

fn cmd_process(input: &str, output: &str) -> Result<(), String> {
    let wav = read_wav(input)?;
    let mut reim = Reim::with_defaults(wav.sample_rate as f64);
    apply_voicing_env(&mut reim);
    let mut out = vec![0.0; wav.samples.len()];
    reim.process_block(&wav.samples, &mut out);
    write_wav_f32(output, &out, wav.sample_rate)?;
    println!(
        "processed {} samples @ {} Hz -> {}",
        wav.samples.len(),
        wav.sample_rate,
        output
    );
    Ok(())
}

// Returns (signal energy = sum of reference^2, error energy = sum of (reference-test)^2)
// over `range`, accumulated in index order (fixed summation order for reproducibility).
fn sig_err(reference: &[f64], test: &[f64], range: std::ops::Range<usize>) -> (f64, f64) {
    let mut sig = 0.0;
    let mut err = 0.0;
    for j in range {
        sig += reference[j] * reference[j];
        let d = reference[j] - test[j];
        err += d * d;
    }
    (sig, err)
}

fn segmental_snr(reference: &[f64], test: &[f64], seg: usize) -> f64 {
    let n = reference.len().min(test.len());
    let mut total = 0.0;
    let mut count = 0;
    let mut i = 0;
    while i < n {
        let end = (i + seg).min(n);
        let (sig, err) = sig_err(reference, test, i..end);
        if sig > 1e-12 {
            total += 10.0 * (sig / (err + 1e-20)).log10();
            count += 1;
        }
        i = end;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn global_snr(reference: &[f64], test: &[f64]) -> f64 {
    let n = reference.len().min(test.len());
    let (sig, err) = sig_err(reference, test, 0..n);
    10.0 * (sig / (err + 1e-20)).log10()
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

fn cmd_eval(reference: &str, input: &str, feat: Option<&str>) -> Result<(), String> {
    let refwav = read_wav(reference)?;
    let inwav = read_wav(input)?;
    let mut reim = Reim::with_defaults(inwav.sample_rate as f64);
    apply_voicing_env(&mut reim);

    // collect per-frame features as we go
    let mut fo_rust: Vec<(bool, f64, bool)> = Vec::new();
    let mut out = vec![0.0; inwav.samples.len()];
    let mut last_frame = 0u64;
    for (i, &x) in inwav.samples.iter().enumerate() {
        out[i] = reim.process_sample(x);
        if reim.frame_count() != last_frame {
            fo_rust.push((reim.last_silence(), reim.last_fo(), reim.last_voiced()));
            last_frame = reim.frame_count();
        }
    }

    let n = refwav.samples.len().min(out.len());
    let max_err = (0..n)
        .map(|i| (refwav.samples[i] - out[i]).abs())
        .fold(0.0_f64, f64::max);
    let gsnr = global_snr(&refwav.samples, &out);
    let ssnr = segmental_snr(&refwav.samples, &out, 240);

    println!("== waveform ==");
    println!("  ref samples     : {}", refwav.samples.len());
    println!("  rust samples    : {}", out.len());
    println!("  max abs error   : {max_err:.6e}");
    println!("  global SNR      : {gsnr:.2} dB");
    println!("  segmental SNR   : {ssnr:.2} dB (240-sample frames)");

    if let Some(feat_path) = feat {
        let text =
            std::fs::read_to_string(feat_path).map_err(|e| format!("read {feat_path}: {e}"))?;
        let mut sil_match = 0usize;
        let mut vu_match = 0usize;
        let mut fo_close = 0usize;
        let mut fo_err_sum = 0.0;
        let mut fo_err_cnt = 0usize;
        let mut total = 0usize;
        for line in text.lines() {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 4 {
                continue;
            }
            let idx: usize = f[0].parse().unwrap_or(usize::MAX);
            if idx >= fo_rust.len() {
                continue;
            }
            let (rsil, rfo, rvu) = fo_rust[idx];
            let csil = f[1] == "1";
            let cfo: f64 = f[2].parse().unwrap_or(0.0);
            let cvu = f[3] == "1";
            total += 1;
            if rsil == csil {
                sil_match += 1;
            }
            if rvu == cvu {
                vu_match += 1;
            }
            if cfo > 0.0 && rfo > 0.0 {
                let rel = (cfo - rfo).abs() / cfo;
                if rel < 0.02 {
                    fo_close += 1;
                }
                fo_err_sum += rel;
                fo_err_cnt += 1;
            } else if cfo == 0.0 && rfo == 0.0 {
                fo_close += 1;
            }
        }
        println!("== per-frame features (vs {feat_path}) ==");
        println!("  frames compared : {total}");
        println!(
            "  silence match   : {sil_match}/{total} ({:.1}%)",
            pct(sil_match, total)
        );
        println!(
            "  voiced match    : {vu_match}/{total} ({:.1}%)",
            pct(vu_match, total)
        );
        println!(
            "  fo within 2%    : {fo_close}/{total} ({:.1}%)",
            pct(fo_close, total)
        );
        if fo_err_cnt > 0 {
            println!(
                "  mean rel fo err : {:.4}% (over {fo_err_cnt} voiced-ish frames)",
                100.0 * fo_err_sum / fo_err_cnt as f64
            );
        }
    }
    Ok(())
}

fn cmd_bench(input: Option<&str>) -> Result<(), String> {
    // input signal: file, or a 2-second synthetic 220 Hz tone at 24 kHz
    let (samples, fs) = match input {
        Some(p) => {
            let w = read_wav(p)?;
            (w.samples, w.sample_rate as f64)
        }
        None => {
            let fs = 24000.0;
            let n = (fs * 2.0) as usize;
            let s = (0..n)
                .map(|i| 0.3 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / fs).sin())
                .collect();
            (s, fs)
        }
    };

    let p = Reim::profile(&samples, fs);
    let audio_secs = p.samples as f64 / p.fs;
    let rtf = audio_secs / p.elapsed_total;

    println!("== throughput ==");
    println!(
        "  input           : {} samples, {:.2} s @ {} Hz",
        p.samples, audio_secs, p.fs as u32
    );
    println!("  process time    : {:.4} s", p.elapsed_total);
    println!(
        "  real-time factor: {:.1}x   (>1 = faster than real time)",
        rtf
    );
    println!(
        "  per sample      : {:.1} ns",
        p.elapsed_total * 1e9 / p.samples as f64
    );

    let frame_work = p.stage_silence + p.stage_fo + p.stage_ap + p.stage_sp + p.stage_new_frame;
    println!("== per-stage (total over run) ==");
    println!("  silence         : {:8.2} ms", p.stage_silence * 1e3);
    println!(
        "  fo              : {:8.2} ms   ({:.0}% of frame work)",
        p.stage_fo * 1e3,
        100.0 * p.stage_fo / frame_work
    );
    println!("  ap              : {:8.2} ms", p.stage_ap * 1e3);
    println!("  sp              : {:8.2} ms", p.stage_sp * 1e3);
    println!("  synth new_frame : {:8.2} ms", p.stage_new_frame * 1e3);
    println!(
        "  synth next_samp : {:8.2} ms   (per-sample hot path)",
        p.stage_next_sample * 1e3
    );
    println!("== synthesis sub-stages ==");
    println!(
        "  minphase pulse  : {:8.2} ms",
        p.synth_minphase_pulse * 1e3
    );
    println!(
        "  minphase noise  : {:8.2} ms",
        p.synth_minphase_noise * 1e3
    );
    println!("  impulse noise   : {:8.2} ms", p.synth_impulse_noise * 1e3);
    println!(
        "  pulse gen (ifft): {:8.2} ms   ({} calls, {:.1} us/call)",
        p.synth_pulse_gen * 1e3,
        p.synth_pulse_gen_count,
        if p.synth_pulse_gen_count > 0 {
            p.synth_pulse_gen / p.synth_pulse_gen_count as f64 * 1e6
        } else {
            0.0
        }
    );

    let mut lat = p.frame_latencies;
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if lat.is_empty() {
        println!("== per-frame analysis latency ==");
        println!("  frames          : 0 (no full frame in input)");
        return Ok(());
    }
    let q = |x: f64| lat[((lat.len() as f64 - 1.0) * x) as usize] * 1e6;
    let budget_us = p.period_ms * 1000.0;
    println!("== per-frame analysis latency ==");
    println!("  frames          : {}", lat.len());
    println!(
        "  mean            : {:.1} us",
        lat.iter().sum::<f64>() / lat.len().max(1) as f64 * 1e6
    );
    println!(
        "  p50 / p99 / max : {:.1} / {:.1} / {:.1} us",
        q(0.50),
        q(0.99),
        q(1.0)
    );
    println!(
        "  frame budget    : {:.1} us ({} ms period)  -> headroom {:.0}x",
        budget_us,
        p.period_ms,
        budget_us / q(1.0).max(1e-9)
    );
    Ok(())
}

// Emit the per-frame Fo contour as CSV: "time_seconds,fo_hz" (fo 0.0 = unvoiced).
// time is the analysis-window center, so the contour aligns with the audio it
// describes. fftsize defaults to the sample-rate-aware default (1024 at 16 kHz,
// 2048 at 24-48 kHz) and can be overridden as the 4th positional arg.
fn cmd_f0(
    input: &str,
    fmin: Option<&str>,
    fmax: Option<&str>,
    fftarg: Option<&str>,
) -> Result<(), String> {
    let fo_floor: f64 = fmin.unwrap_or("71").parse().map_err(|_| "bad fmin")?;
    let fo_ceil: f64 = fmax.unwrap_or("800").parse().map_err(|_| "bad fmax")?;
    let wav = read_wav(input)?;
    let fs = wav.sample_rate as f64;
    let fftsize: usize = match fftarg {
        Some(s) => s.parse().map_err(|_| "bad fftsize")?,
        None => default_fftsize(fs, fo_floor),
    };
    let mut reim = Reim::new(fs, 5.0, fftsize, fo_floor, fo_ceil);
    apply_voicing_env(&mut reim);
    let half = fftsize as f64 / 2.0;
    let mut last = 0u64;
    let mut out = String::new();
    for (i, &x) in wav.samples.iter().enumerate() {
        reim.process_sample(x);
        if reim.frame_count() != last {
            last = reim.frame_count();
            let t = ((i as f64 - half) / fs).max(0.0);
            // emit pitch only on voiced frames so the contour reflects the full
            // voicing decision (incl. the sub-fundamental rumble guard), not just analyze_fo
            let fo = if reim.last_voiced() {
                reim.last_fo()
            } else {
                0.0
            };
            out.push_str(&format!("{t:.6},{fo:.4}\n"));
        }
    }
    print!("{out}");
    Ok(())
}

// Emit per-frame voicing features as headered CSV for offline analysis and
// fusion-weight fitting. Same framing/timing as `f0`.
fn cmd_features(input: &str, fmin: Option<&str>, fmax: Option<&str>) -> Result<(), String> {
    let fo_floor: f64 = fmin.unwrap_or("71").parse().map_err(|_| "bad fmin")?;
    let fo_ceil: f64 = fmax.unwrap_or("800").parse().map_err(|_| "bad fmax")?;
    let wav = read_wav(input)?;
    let fs = wav.sample_rate as f64;
    let fftsize = default_fftsize(fs, fo_floor);
    let mut reim = Reim::new(fs, 5.0, fftsize, fo_floor, fo_ceil);
    apply_voicing_env(&mut reim);
    let half = fftsize as f64 / 2.0;
    let mut last = 0u64;
    let mut out =
        String::from("frame,time,silence,fo,voiced,score,score_margin,nccf,cpp,prob,strength\n");
    for (i, &x) in wav.samples.iter().enumerate() {
        reim.process_sample(x);
        if reim.frame_count() != last {
            last = reim.frame_count();
            let t = ((i as f64 - half) / fs).max(0.0);
            let f = reim.last_voicing_features();
            out.push_str(&format!(
                "{},{t:.6},{},{:.4},{},{:.6e},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                last - 1,
                reim.last_silence() as u8,
                reim.last_fo(),
                reim.last_voiced() as u8,
                f.score,
                f.score_margin,
                f.nccf,
                f.cpp,
                reim.last_voicing_score(),
                reim.last_voicing_strength(),
            ));
        }
    }
    print!("{out}");
    Ok(())
}

// Dump per-frame aperiodicity as raw little-endian f64: frame_count rows of
// (fftsize/2 + 1) bins each. For comparing the analyzer against a reference.
fn cmd_ap(input: &str, output: &str) -> Result<(), String> {
    let wav = read_wav(input)?;
    let mut reim = Reim::with_defaults(wav.sample_rate as f64);
    apply_voicing_env(&mut reim);
    let mut bytes = Vec::new();
    let mut last = 0u64;
    for &x in &wav.samples {
        reim.process_sample(x);
        if reim.frame_count() != last {
            last = reim.frame_count();
            for &a in reim.last_aperiodicity() {
                bytes.extend_from_slice(&a.to_le_bytes());
            }
        }
    }
    std::fs::write(output, &bytes).map_err(|e| format!("write {output}: {e}"))?;
    println!("wrote aperiodicity: {last} frames -> {output}");
    Ok(())
}

fn cmd_segment(input: &str, svg_output: Option<&str>) -> Result<(), String> {
    let wav = read_wav(input)?;
    let fs = wav.sample_rate as f64;
    let mut analyzer = reim::Analyzer::with_defaults(fs);
    let frames = analyzer.analyze_to_frames(&wav.samples);
    let config = reim::segment::SegmentConfig::default();
    let period_ms = 5.0;
    let frame_rate = 1000.0 / period_ms;
    let segments = reim::segment::segment(&frames, frame_rate, &config);

    println!("segments: {}", segments.len());
    println!(
        "{:<8} {:<8} {:<8} {:<12} {:<10} {:<8} {:<8} {:<8} {:<8}",
        "start", "end", "type", "center", "vib_rate", "vib_amp", "drift%", "vib%", "res%"
    );

    for seg in &segments {
        let start_ms = seg.frames.start as f64 * period_ms;
        let end_ms = seg.frames.end as f64 * period_ms;
        match &seg.kind {
            reim::segment::SegmentKind::Note(c) => {
                let (d_frac, v_frac, r_frac) = reim::segment::variance_explained(c);
                let mean_amp = if c.vibrato_amp.is_empty() {
                    0.0
                } else {
                    c.vibrato_amp.iter().sum::<f64>() / c.vibrato_amp.len() as f64
                };
                let midi = 69.0 + c.center_cents / 100.0;
                let midi_round = midi.round() as i32;
                let names = [
                    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                ];
                let note = ((midi_round % 12) + 12) % 12;
                let oct = midi_round / 12 - 1;
                let cents_off = c.center_cents - (midi_round as f64 - 69.0) * 100.0;
                let note_str = format!("{}{}{:+.0}c", names[note as usize], oct, cents_off);
                println!(
                    "{:<8.0} {:<8.0} {:<8} {:<12} {:<10.1} {:<8.1} {:<8.0} {:<8.0} {:<8.0}",
                    start_ms,
                    end_ms,
                    "Note",
                    note_str,
                    c.vibrato_rate_hz,
                    mean_amp,
                    d_frac * 100.0,
                    v_frac * 100.0,
                    r_frac * 100.0
                );
            }
            reim::segment::SegmentKind::Unvoiced => {
                println!(
                    "{:<8.0} {:<8.0} {:<8} {:<12} {:<10} {:<8} {:<8} {:<8} {:<8}",
                    start_ms, end_ms, "Unvoiced", "-", "-", "-", "-", "-", "-"
                );
            }
        }
    }

    if let Some(svg_path) = svg_output {
        let svg = reim::segment::contour_svg(&frames, &segments, frame_rate, None);
        std::fs::write(svg_path, &svg).map_err(|e| format!("write {svg_path}: {e}"))?;
        println!("wrote contour SVG: {svg_path}");
    }
    Ok(())
}

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  reim process <in.wav> <out.wav>");
    eprintln!("  reim eval <ref.wav> <in.wav> [feat.csv]");
    eprintln!("  reim bench [in.wav]");
    eprintln!("  reim f0 <in.wav> [fmin] [fmax] [fftsize]");
    eprintln!("  reim features <in.wav> [fmin] [fmax]");
    eprintln!("  reim ap <in.wav> <out.f64>");
    eprintln!("  reim segment <in.wav> [contour.svg]");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(|s| s.as_str()) {
        Some("process") if args.len() == 4 => cmd_process(&args[2], &args[3]),
        Some("eval") if args.len() >= 4 => {
            cmd_eval(&args[2], &args[3], args.get(4).map(|s| s.as_str()))
        }
        Some("bench") => cmd_bench(args.get(2).map(|s| s.as_str())),
        Some("f0") if args.len() >= 3 => cmd_f0(
            &args[2],
            args.get(3).map(|s| s.as_str()),
            args.get(4).map(|s| s.as_str()),
            args.get(5).map(|s| s.as_str()),
        ),
        Some("features") if args.len() >= 3 => cmd_features(
            &args[2],
            args.get(3).map(|s| s.as_str()),
            args.get(4).map(|s| s.as_str()),
        ),
        Some("ap") if args.len() == 4 => cmd_ap(&args[2], &args[3]),
        Some("segment") if args.len() >= 3 => {
            cmd_segment(&args[2], args.get(3).map(|s| s.as_str()))
        }
        _ => usage(),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
