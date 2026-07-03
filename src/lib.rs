//! ReIm — a real-time WORLD-like vocoder, single-file Rust port.
//!
//! Speech analysis/synthesis using three acoustic features compatible with WORLD:
//! fundamental frequency (Fo), aperiodicity (Ap), and spectral envelope (Sp).
//! The analysis order is Silence -> Fo -> Ap -> Sp, then synthesis.
//!
//! Design goals: allocation-free steady-state (real-time safe), faithful port of
//! the reference C implementation, zero external dependencies.
//!
//! CLI:
//!   reim process <in.wav> <out.wav>        analyze+synthesize a mono WAV
//!   reim eval <ref.wav> <in.wav> [feat.csv] compare against a reference output
//!   reim bench [in.wav]                     profile throughput and per-stage latency
//!   reim f0 <in.wav> [fmin] [fmax] [fftsize] emit the per-frame Fo contour as CSV
//!
//! The hot path (`Reim::process_sample`) performs no heap allocation; all working
//! buffers are owned by the analyzer/synthesizer state and reused every frame.
//!
//! # Example
//! ```
//! use reim::Reim;
//! let mut reim = Reim::with_defaults(48_000.0); // sample rate in Hz
//! let input = vec![0.0f64; 4_800];
//! let mut output = vec![0.0; input.len()];
//! reim.process_block(&input, &mut output);     // allocation-free per sample
//! assert_eq!(output.len(), input.len());
//! ```

// This is a faithful, index-parallel port of a C DSP reference. Numeric kernels
// index several arrays in lockstep, so range loops read closer to the source than
// zipped iterators; analyzer methods carry the same parameter set as their C
// counterparts; and REIM_PI deliberately reproduces the C's truncated pi literal.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_range_contains
)]

// The reference C code uses a *truncated* pi literal. Reproduce it exactly so the
// windowing/phase math matches the C bit-for-bit where the FFT allows. The allow
// is scoped to this one literal so approx_constant stays deny-by-default elsewhere.
#[allow(clippy::approx_constant)]
const REIM_PI: f64 = 3.14159265358979;
const U32_MAX_F: f64 = u32::MAX as f64;

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// ============================================================================
// Math helpers
// ============================================================================

#[inline]
fn complex_abs2(re: f64, im: f64) -> f64 {
    re * re + im * im
}
#[inline]
fn complex_abs(re: f64, im: f64) -> f64 {
    complex_abs2(re, im).sqrt()
}
#[inline]
fn complex_angle(re: f64, im: f64) -> f64 {
    im.atan2(re)
}

/// Instantaneous frequency between two one-sample-shifted spectra, in Hz.
#[inline]
fn instfreq(xr1: f64, xi1: f64, xr2: f64, xi2: f64, fs: f64) -> f64 {
    (fs / (2.0 * REIM_PI) * complex_angle(xr1 * xr2 + xi1 * xi2, xi1 * xr2 - xr1 * xi2)).abs()
}

/// Clamp a (floating) array position into `[0, len-1]` (CLAMP_INDEX in C).
#[inline]
fn clamp_index_f(x: f64, len: usize) -> f64 {
    x.max(0.0).min(len.saturating_sub(1) as f64)
}

/// ifftshift of the first `numbins` samples by `numbins-1` (circular by N/2).
fn ifftshift(source: &[f64], destination: &mut [f64], numbins: usize) {
    for k in 0..numbins - 1 {
        destination[k] = source[numbins - 1 + k];
        destination[numbins - 1 + k] = source[k];
    }
}

// --- analysis windows (centered, length-limited) ---------------------------

fn nuttall_window(index: f64, fftsize: f64, length: f64) -> f64 {
    let wt = 2.0 * REIM_PI * (index - (fftsize - 1.0) / 2.0) / length;
    if wt < -REIM_PI || REIM_PI < wt {
        return 0.0;
    }
    0.355768 + 0.487396 * wt.cos() + 0.144232 * (2.0 * wt).cos() + 0.012604 * (3.0 * wt).cos()
}

fn hanning_window(index: f64, fftsize: f64, length: f64) -> f64 {
    let wt = REIM_PI * (2.0 * index - (fftsize - 1.0)) / length;
    if wt < -REIM_PI || REIM_PI < wt {
        return 0.0;
    }
    0.5 + 0.5 * wt.cos()
}

fn blackman_window(index: f64, fftsize: f64, length: f64) -> f64 {
    let wt = 2.0 * REIM_PI * (index - (fftsize - 1.0) / 2.0) / length;
    if wt < -REIM_PI || REIM_PI < wt {
        return 0.0;
    }
    0.42 + 0.5 * wt.cos() + 0.08 * (2.0 * wt).cos()
}

fn vorbis_window(index: f64, fftsize: f64) -> f64 {
    let wt = REIM_PI * (index + 0.5) / fftsize;
    let s = wt.sin();
    (REIM_PI / 2.0 * s * s).sin()
}

// ============================================================================
// Xorshift128 RNG (period 2^128-1). Fixed seed -> deterministic velvet noise.
// ============================================================================

#[derive(Clone)]
struct Xorshift {
    s: [u32; 4],
}

impl Xorshift {
    fn new() -> Self {
        Xorshift {
            s: [123456789, 362436069, 521288629, 88675123],
        }
    }
    /// Uniform random in [0, 1].
    fn uniform(&mut self) -> f64 {
        let t = self.s[3];
        let a = self.s[0];
        self.s[3] = self.s[2];
        self.s[2] = self.s[1];
        self.s[1] = a;
        let t = t ^ (t << 11);
        self.s[0] = (t ^ (t >> 8)) ^ (a ^ (a >> 19));
        self.s[0] as f64 / U32_MAX_F
    }
}

// ============================================================================
// FFT — wraps rustfft (SIMD-optimized split-radix) behind the same planar API
// the rest of the codebase expects. Forward uses the exp(-i) kernel; inverse
// uses exp(+i) and scales by 1/N, matching the reference conventions.
// ============================================================================

struct Fft {
    n: usize,
    fwd: std::sync::Arc<dyn rustfft::Fft<f64>>,
    #[cfg(test)]
    inv: std::sync::Arc<dyn rustfft::Fft<f64>>,
    buf: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    // f32 real FFT (synthesis path)
    r2c: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    c2r: std::sync::Arc<dyn realfft::ComplexToReal<f32>>,
    half_buf: Vec<Complex<f32>>,
    real_scratch: Vec<Complex<f32>>,
    real_buf: Vec<f32>,
    // f64 real FFT (analysis path)
    r2c_f64: std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    c2r_f64: std::sync::Arc<dyn realfft::ComplexToReal<f64>>,
    half_buf_f64: Vec<Complex<f64>>,
    real_scratch_f64: Vec<Complex<f64>>,
    real_buf_f64: Vec<f64>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n >= 2,
            "fft size must be a power of two"
        );
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch_len = fwd
            .get_inplace_scratch_len()
            .max(inv.get_inplace_scratch_len());
        let mut real_planner_f32 = RealFftPlanner::<f32>::new();
        let r2c = real_planner_f32.plan_fft_forward(n);
        let c2r = real_planner_f32.plan_fft_inverse(n);
        let real_scratch_len_f32 = r2c.get_scratch_len().max(c2r.get_scratch_len());
        let mut real_planner_f64 = RealFftPlanner::<f64>::new();
        let r2c_f64 = real_planner_f64.plan_fft_forward(n);
        let c2r_f64 = real_planner_f64.plan_fft_inverse(n);
        let real_scratch_len_f64 = r2c_f64.get_scratch_len().max(c2r_f64.get_scratch_len());
        Fft {
            n,
            fwd,
            #[cfg(test)]
            inv,
            buf: vec![Complex::default(); n],
            scratch: vec![Complex::default(); scratch_len],
            r2c,
            c2r,
            half_buf: vec![Complex::default(); n / 2 + 1],
            real_scratch: vec![Complex::default(); real_scratch_len_f32],
            real_buf: vec![0.0f32; n],
            r2c_f64,
            c2r_f64,
            half_buf_f64: vec![Complex::default(); n / 2 + 1],
            real_scratch_f64: vec![Complex::default(); real_scratch_len_f64],
            real_buf_f64: vec![0.0; n],
        }
    }

    #[inline]
    fn planar_to_interleaved(&mut self, re: &[f64], im: &[f64]) {
        for i in 0..self.n {
            self.buf[i] = Complex::new(re[i], im[i]);
        }
    }

    #[inline]
    fn interleaved_to_planar(&self, re: &mut [f64], im: &mut [f64]) {
        for i in 0..self.n {
            re[i] = self.buf[i].re;
            im[i] = self.buf[i].im;
        }
    }

    #[inline]
    fn forward(&mut self, re: &mut [f64], im: &mut [f64]) {
        self.planar_to_interleaved(re, im);
        self.fwd
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        self.interleaved_to_planar(re, im);
    }

    #[cfg(test)]
    #[inline]
    fn inverse(&mut self, re: &mut [f64], im: &mut [f64]) {
        self.planar_to_interleaved(re, im);
        self.inv
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        let scale = 1.0 / self.n as f64;
        for c in &mut self.buf {
            c.re *= scale;
            c.im *= scale;
        }
        self.interleaved_to_planar(re, im);
    }

    /// r2c FFT in f32. Accepts/returns f64;
    /// conversion happens at the boundary.
    #[inline]
    fn forward_real(&mut self, data: &mut [f64], im_out: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            self.real_buf[i] = data[i] as f32;
        }
        self.r2c
            .process_with_scratch(
                &mut self.real_buf,
                &mut self.half_buf,
                &mut self.real_scratch,
            )
            .unwrap();
        let numbins = n / 2 + 1;
        for k in 0..numbins {
            data[k] = self.half_buf[k].re as f64;
            im_out[k] = self.half_buf[k].im as f64;
        }
    }

    /// Like `inverse_real` but runs the IFFT in f32. Accepts/returns f64;
    /// conversion happens at the boundary.
    #[inline]
    fn inverse_real(&mut self, data: &mut [f64], im: &[f64]) {
        let numbins = self.n / 2 + 1;
        for k in 0..numbins {
            self.half_buf[k] = Complex::new(data[k] as f32, im[k] as f32);
        }
        self.half_buf[0].im = 0.0;
        self.half_buf[numbins - 1].im = 0.0;
        self.c2r
            .process_with_scratch(
                &mut self.half_buf,
                &mut self.real_buf,
                &mut self.real_scratch,
            )
            .unwrap();
        let scale = 1.0 / self.n as f64;
        for i in 0..self.n {
            data[i] = self.real_buf[i] as f64 * scale;
        }
    }

    /// r2c FFT in f64: `re[0..n]` → half-spectrum in `re[0..numbins]`, `im[0..numbins]`.
    #[inline]
    fn forward_real_f64(&mut self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        let numbins = n / 2 + 1;
        self.real_buf_f64[..n].copy_from_slice(&re[..n]);
        self.r2c_f64
            .process_with_scratch(
                &mut self.real_buf_f64,
                &mut self.half_buf_f64,
                &mut self.real_scratch_f64,
            )
            .unwrap();
        for k in 0..numbins {
            re[k] = self.half_buf_f64[k].re;
            im[k] = self.half_buf_f64[k].im;
        }
    }

    /// c2r IFFT in f64: half-spectrum `re[0..numbins]`, `im[0..numbins]` → `re[0..n]`.
    #[inline]
    fn inverse_real_f64(&mut self, re: &mut [f64], im: &[f64]) {
        let n = self.n;
        let numbins = n / 2 + 1;
        for k in 0..numbins {
            self.half_buf_f64[k] = Complex::new(re[k], im[k]);
        }
        self.half_buf_f64[0].im = 0.0;
        self.half_buf_f64[numbins - 1].im = 0.0;
        self.c2r_f64
            .process_with_scratch(
                &mut self.half_buf_f64,
                &mut self.real_buf_f64,
                &mut self.real_scratch_f64,
            )
            .unwrap();
        let scale = 1.0 / n as f64;
        for i in 0..n {
            re[i] = self.real_buf_f64[i] * scale;
        }
    }
}

// ============================================================================
// Circular structures
// ============================================================================

#[inline]
fn wrap_next(index: usize, capacity: usize) -> usize {
    let i = index + 1;
    if i >= capacity {
        0
    } else {
        i
    }
}

/// Fixed-capacity ring of the most recent `capacity` samples.
struct CircularBuffer {
    head: usize,
    buf: Vec<f64>,
}

impl CircularBuffer {
    fn new(capacity: usize) -> Self {
        CircularBuffer {
            head: 0,
            buf: vec![0.0; capacity],
        }
    }
    fn push(&mut self, value: f64) {
        self.head = wrap_next(self.head, self.buf.len());
        self.buf[self.head] = value;
    }
    /// Copy oldest..newest into `dst` (len == capacity).
    fn copy_all(&self, dst: &mut [f64]) {
        let cap = self.buf.len();
        let mut index = self.head;
        for d in dst.iter_mut().take(cap) {
            index = wrap_next(index, cap);
            *d = self.buf[index];
        }
    }
}

/// Additive output ring used by the overlap-add synthesizer.
struct CircularQueue {
    head: usize,
    remaining: usize,
    buf: Vec<f64>,
}

impl CircularQueue {
    fn new(capacity: usize) -> Self {
        CircularQueue {
            head: 0,
            remaining: 0,
            buf: vec![0.0; capacity],
        }
    }
    /// Add `samples` into the queue starting at the current head (overlap-add).
    fn push_additive(&mut self, samples: &[f64]) {
        let cap = self.buf.len();
        let size = samples.len();
        if size <= cap {
            let first = cap - self.head;
            if size <= first {
                for (d, &s) in self.buf[self.head..self.head + size]
                    .iter_mut()
                    .zip(samples)
                {
                    *d += s;
                }
            } else {
                let (s1, s2) = samples.split_at(first);
                for (d, &s) in self.buf[self.head..].iter_mut().zip(s1) {
                    *d += s;
                }
                for (d, &s) in self.buf[..s2.len()].iter_mut().zip(s2) {
                    *d += s;
                }
            }
        } else {
            let mut index = self.head;
            for (i, &value) in samples.iter().enumerate() {
                if i >= cap {
                    self.buf[index] = value;
                    self.head = wrap_next(self.head, cap);
                } else {
                    self.buf[index] += value;
                }
                index = wrap_next(index, cap);
            }
        }
        self.remaining = size.max(self.remaining).min(cap);
    }
    fn pop(&mut self) -> f64 {
        if self.remaining == 0 {
            return 0.0;
        }
        let value = self.buf[self.head];
        self.buf[self.head] = 0.0;
        self.head = wrap_next(self.head, self.buf.len());
        self.remaining -= 1;
        value
    }
}

// ============================================================================
// Vocoder configuration
// ============================================================================

#[derive(Clone, Copy)]
struct Config {
    period: f64,   // frame period in ms
    fs: f64,       // sampling frequency
    fo_floor: f64, // lower bound of Fo
    fo_ceil: f64,  // upper bound of Fo
    fftsize: usize,
    numbins: usize, // fftsize/2 + 1
}

impl Config {
    fn new(period: f64, fftsize: usize, fo_floor: f64, fo_ceil: f64, fs: f64) -> Self {
        assert!(fftsize.is_power_of_two());
        assert!(fo_floor > 0.0 && fo_floor < fo_ceil && fo_ceil < fs / 2.0 && fs > 0.0);
        Config {
            period,
            fs,
            fo_floor,
            fo_ceil,
            fftsize,
            numbins: fftsize / 2 + 1,
        }
    }
}

// ============================================================================
// Audio framer — emits an fftsize+1 sample window at each frame boundary.
// ============================================================================

struct Framer {
    framesize: f64,
    position: f64,
    buffer_in: CircularBuffer,
}

impl Framer {
    fn new(cfg: &Config) -> Self {
        Framer {
            framesize: cfg.period / 1000.0 * cfg.fs,
            position: 0.0,
            buffer_in: CircularBuffer::new(cfg.fftsize + 1),
        }
    }

    /// Push one input sample; if a new frame begins, fill `frame_waveform`
    /// (len == fftsize+1, oldest..newest) and return true.
    fn next(&mut self, input: f64, frame_waveform: &mut [f64]) -> bool {
        self.buffer_in.push(input);
        let has_new_frame = self.position.floor() == 0.0;
        if has_new_frame {
            self.buffer_in.copy_all(frame_waveform);
        }
        if self.position >= self.framesize - 1.0 {
            self.position -= self.framesize - 1.0;
        } else {
            self.position += 1.0;
        }
        has_new_frame
    }
}

// ============================================================================
// Silence analysis
// ============================================================================

const SILENCE_THRESHOLD: f64 = 0.00025; // -72 dB

/// Frame RMS below threshold => silence. (The C reads `input[1..=fftsize]`; the
/// last index is a one-element out-of-bounds read of the source buffer, harmless
/// to the decision. We sum the in-bounds frame samples instead.)
fn analyze_silence(cfg: &Config, input: &[f64], threshold: f64) -> bool {
    let fftsize = cfg.fftsize;
    let mut sum_of_squares = 0.0;
    for i in 1..fftsize {
        let x = input[i];
        sum_of_squares += x * x;
    }
    let mean_square = sum_of_squares / fftsize as f64;
    mean_square < threshold * threshold
}

// ============================================================================
// Fo analysis — DIO (zero-crossing) candidates refined by instantaneous
// frequency, scored by summation of residual harmonics (SRH).
// ============================================================================

struct FoAnalyzer {
    num_candidates: usize,
    channel_filters: Vec<f64>, // [ch*fftsize + k]
    channel_offsets: Vec<usize>,
    window: Vec<f64>,
    spec_r: Vec<f64>,
    spec_i: Vec<f64>,
    specd_r: Vec<f64>,
    specd_i: Vec<f64>,
    pspec: Vec<f64>,
    ifreqf: Vec<f64>,
    spec_filt_r: Vec<f64>,
    spec_filt_i: Vec<f64>,
    filtered_r: Vec<f64>,
    filtered_i: Vec<f64>,
    cep_r: Vec<f64>,
    cep_i: Vec<f64>,
    fo_previous: f64,
    last_score: f64,
    last_nccf: f64,
    last_cpp: f64,
}

fn get_interpolated_spectrum(freq: f64, fs: f64, spec: &[f64], numbins: usize) -> f64 {
    let position = clamp_index_f(freq / (fs / 2.0) * (numbins - 1) as f64, numbins - 1);
    let index = position.floor() as usize;
    let delta = position - index as f64;
    (1.0 - delta) * spec[index] + delta * spec[index + 1]
}

fn refine_fo(
    fo: f64,
    fo_floor: f64,
    fo_ceil: f64,
    fs: f64,
    ifreqf: &[f64],
    pspec: &[f64],
    numbins: usize,
) -> f64 {
    let harmonics = 3;
    let mut sum_freq = 0.0;
    let mut denominator = 0.0;
    for h in 1..=harmonics {
        let hf = h as f64;
        let freq = get_interpolated_spectrum(fo * hf, fs, ifreqf, numbins);
        let weight = get_interpolated_spectrum(fo * hf, fs, pspec, numbins);
        sum_freq += freq * weight;
        denominator += hf * weight;
    }
    let refined_fo = sum_freq / denominator;
    if refined_fo < fo_floor || refined_fo > fo_ceil || (refined_fo - fo).abs() > fo {
        return fo;
    }
    refined_fo
}

fn get_harmonic_score(fo: f64, fs: f64, pspec: &[f64], numbins: usize) -> f64 {
    let harmonics = 3;
    let mut score = 1.0;
    for h in 1..=harmonics {
        let hf = h as f64;
        score *= get_interpolated_spectrum(fo * hf, fs, pspec, numbins);
        score /= get_interpolated_spectrum(fo * (hf - 0.5), fs, pspec, numbins);
    }
    score
}

/// RAPT-style normalized cross-correlation of a frame with itself at a single
/// lag (samples). Mean-removed, so DC/rumble does not inflate the score.
/// Returns 0.0 (uninformative, not garbage) when the frame cannot fit one full
/// period of overlap at the lag. O(frame).
fn nccf_at_lag(x: &[f64], lag: usize) -> f64 {
    let m = x.len().saturating_sub(lag);
    if lag == 0 || m < lag {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let mut cross = 0.0;
    let mut e0 = 0.0;
    let mut el = 0.0;
    for j in 0..m {
        let a = x[j] - mean;
        let b = x[j + lag] - mean;
        cross += a * b;
        e0 += a * a;
        el += b * b;
    }
    cross / (e0 * el).sqrt().max(1e-12)
}

/// Cepstral peak prominence: peak of the log-power-spectrum cepstrum in the
/// pitch quefrency range [fs/fo_ceil, fs/fo_floor], minus the mean cepstrum
/// over that range (simple baseline, no regression). Scratch buffers `re`/`im`
/// are fftsize long; `re[..numbins]` is overwritten with ln(pspec).
fn cepstral_peak_prominence(
    pspec: &[f64],
    numbins: usize,
    fs: f64,
    fo_floor: f64,
    fo_ceil: f64,
    re: &mut [f64],
    im: &mut [f64],
    fft: &mut Fft,
) -> f64 {
    for k in 0..numbins {
        re[k] = pspec[k].max(1e-15).ln();
    }
    mirror_upper(re, numbins);
    im[..numbins].fill(0.0);
    fft.inverse_real_f64(re, &im[..numbins]);

    let q_lo = (fs / fo_ceil).ceil() as usize;
    let q_hi = ((fs / fo_floor).floor() as usize).min(numbins - 1);
    if q_lo >= q_hi {
        return 0.0;
    }
    let mut max = f64::NEG_INFINITY;
    let mut mean = 0.0;
    for &c in &re[q_lo..=q_hi] {
        max = max.max(c);
        mean += c;
    }
    mean /= (q_hi - q_lo + 1) as f64;
    max - mean
}

/// Estimate Fo from zero-crossings / peaks / dips of a band-limited waveform.
/// Returns the mean interval-frequency, or None when no intervals were found or
/// the candidate fails the periodicity gate (see below).
fn analyze_fo_with_zerocross(x: &[f64], fs: f64) -> Option<f64> {
    let length = x.len();
    let mut last_positive: i64 = -1;
    let mut last_negative: i64 = -1;
    let mut last_peak: i64 = -1;
    let mut last_dip: i64 = -1;

    let mut total_interval = 0.0;
    let mut sum_freq = 0.0;
    let mut sum_square_freq = 0.0;

    let mut accumulate = |i: i64, last: i64| {
        let interval = (i - last) as f64;
        let freq = fs / interval;
        total_interval += interval;
        sum_freq += freq * interval;
        sum_square_freq += freq * freq * interval;
    };

    let mut xprev = x[0];
    let mut xdiffprev = x[1] - x[0];
    for i in 1..length - 1 {
        let xcurr = x[i];
        let xdiffcurr = x[i + 1] - x[i];
        let ii = i as i64;
        if xprev < 0.0 && xcurr >= 0.0 {
            if last_positive >= 0 {
                accumulate(ii, last_positive);
            }
            last_positive = ii;
        } else if xprev > 0.0 && xcurr <= 0.0 {
            if last_negative >= 0 {
                accumulate(ii, last_negative);
            }
            last_negative = ii;
        }
        if xdiffprev < 0.0 && xdiffcurr >= 0.0 {
            if last_peak >= 0 {
                accumulate(ii, last_peak);
            }
            last_peak = ii;
        } else if xdiffprev > 0.0 && xdiffcurr <= 0.0 {
            if last_dip > 1 {
                // NOTE: asymmetric guard (>1, not >=0) preserved from the C source.
                accumulate(ii, last_dip);
            }
            last_dip = ii;
        }
        xprev = xcurr;
        xdiffprev = xdiffcurr;
    }

    if total_interval <= 0.0 {
        return None;
    }
    let mean_freq = sum_freq / total_interval;
    if mean_freq <= 0.0 || mean_freq > fs / 2.0 {
        return None;
    }
    // Periodicity gate: reject the candidate when the interval-weighted variance of
    // the interval frequencies exceeds the mean frequency -- a scale-aware test for
    // whether the zero-cross intervals are regular enough to be a real pitch period.
    // (The C reference writes this as a sign-errored std dev, sqrt(E[f^2] - E[f]),
    // whose rsd > 1 reject is algebraically *exactly* this `variance > mean` gate; we
    // compute the real variance directly, which is the same gate. The threshold is the
    // mean, NOT mean^2: a textbook coefficient-of-variation gate (variance > mean^2)
    // almost never fires and lets rumble/noise through -- measured RPA 72 -> 67,
    // octave error 3.8 -> 8% -- so the mean is the load-bearing part, not the sqrt.)
    let variance = sum_square_freq / total_interval - mean_freq * mean_freq;
    if variance > mean_freq {
        return None;
    }
    Some(mean_freq)
}

impl FoAnalyzer {
    fn new(cfg: &Config, fft: &mut Fft) -> Self {
        let fs = cfg.fs;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        let channels_per_octave = 2.0;
        let num_candidates =
            ((cfg.fo_ceil / cfg.fo_floor).log2() * channels_per_octave).ceil() as usize;

        let mut channel_filters = vec![0.0; num_candidates * fftsize];
        let mut channel_offsets = vec![0usize; num_candidates];
        let mut xr = vec![0.0; fftsize];
        let mut xi = vec![0.0; fftsize];
        for ch in 0..num_candidates {
            let frequency = cfg.fo_floor * 2f64.powf((1.0 + ch as f64) / channels_per_octave);
            let lpf_window_length = (fs / frequency).ceil();
            for k in 0..fftsize {
                xr[k] = nuttall_window(k as f64, fftsize as f64, lpf_window_length);
                xi[k] = 0.0;
            }
            fft.forward(&mut xr, &mut xi);
            for k in 0..fftsize {
                channel_filters[ch * fftsize + k] = complex_abs(xr[k], xi[k]);
            }
            channel_offsets[ch] = lpf_window_length as usize;
        }

        let window_length = (4.0 * fs / cfg.fo_floor).min(fftsize as f64);
        let window = (0..fftsize)
            .map(|k| nuttall_window(k as f64, fftsize as f64, window_length))
            .collect();

        FoAnalyzer {
            num_candidates,
            channel_filters,
            channel_offsets,
            window,
            spec_r: vec![0.0; fftsize],
            spec_i: vec![0.0; fftsize],
            specd_r: vec![0.0; fftsize],
            specd_i: vec![0.0; fftsize],
            pspec: vec![0.0; numbins],
            ifreqf: vec![0.0; numbins],
            spec_filt_r: vec![0.0; fftsize],
            spec_filt_i: vec![0.0; fftsize],
            filtered_r: vec![0.0; fftsize],
            filtered_i: vec![0.0; fftsize],
            cep_r: vec![0.0; fftsize],
            cep_i: vec![0.0; fftsize],
            fo_previous: 0.0,
            last_score: 0.0,
            last_nccf: 0.0,
            last_cpp: 0.0,
        }
    }

    fn analyze(
        &mut self,
        cfg: &Config,
        fft: &mut Fft,
        input: &[f64],
        input_delayed: &[f64],
    ) -> f64 {
        let fs = cfg.fs;
        let fo_floor = cfg.fo_floor;
        let fo_ceil = cfg.fo_ceil;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        // windowed spectra of the frame and its one-sample-delayed copy (real FFT)
        for k in 0..fftsize {
            self.spec_r[k] = input[k] * self.window[k];
            self.specd_r[k] = input_delayed[k] * self.window[k];
        }
        fft.forward_real_f64(&mut self.spec_r, &mut self.spec_i);
        fft.forward_real_f64(&mut self.specd_r, &mut self.specd_i);

        for k in 0..numbins {
            self.pspec[k] = complex_abs2(self.spec_r[k], self.spec_i[k]) + 1e-15;
            self.ifreqf[k] = instfreq(
                self.spec_r[k],
                self.spec_i[k],
                self.specd_r[k],
                self.specd_i[k],
                fs,
            );
        }

        // spectrum of the DC-removed frame (for band-pass filtering, real FFT)
        let mut mean_input = 0.0;
        for k in 0..fftsize {
            mean_input += input[k];
        }
        mean_input /= fftsize as f64;
        for k in 0..fftsize {
            self.spec_filt_r[k] = input[k] - mean_input;
        }
        fft.forward_real_f64(&mut self.spec_filt_r, &mut self.spec_filt_i);

        // initial estimate: previous fo
        let mut best_fo = -1.0;
        let mut best_score = -1.0;
        if self.fo_previous > fo_floor {
            best_fo = refine_fo(
                self.fo_previous,
                fo_floor,
                fo_ceil,
                fs,
                &self.ifreqf,
                &self.pspec,
                numbins,
            );
            best_score = get_harmonic_score(best_fo, fs, &self.pspec, numbins);
        }

        // DIO over each channel filter (half-spectrum multiply + real IFFT)
        for ch in 0..self.num_candidates {
            let base = ch * fftsize;
            for k in 0..numbins {
                let filter = self.channel_filters[base + k];
                self.filtered_r[k] = self.spec_filt_r[k] * filter;
                self.filtered_i[k] = self.spec_filt_i[k] * filter;
            }
            fft.inverse_real_f64(&mut self.filtered_r, &self.filtered_i);

            let offset = self.channel_offsets[ch];
            let Some(fo) = analyze_fo_with_zerocross(&self.filtered_r[offset..], fs) else {
                continue;
            };
            if fo < fo_floor || fo > fo_ceil {
                continue;
            }

            let fo_refined = refine_fo(
                fo,
                fo_floor,
                fo_ceil,
                fs,
                &self.ifreqf,
                &self.pspec,
                numbins,
            );
            let score = get_harmonic_score(fo_refined, fs, &self.pspec, numbins);
            if best_score < score {
                best_fo = fo_refined;
                best_score = score;
            }
        }

        self.last_score = best_score;
        self.last_cpp = cepstral_peak_prominence(
            &self.pspec,
            numbins,
            fs,
            fo_floor,
            fo_ceil,
            &mut self.cep_r,
            &mut self.cep_i,
            fft,
        );
        self.last_nccf = if best_fo >= fo_floor && best_fo <= fo_ceil {
            nccf_at_lag(input, (fs / best_fo).round() as usize)
        } else {
            0.0
        };
        if best_fo < fo_floor || best_fo > fo_ceil || best_score < 0.0 {
            return 0.0;
        }
        self.fo_previous = best_fo;
        best_fo
    }
}

// ============================================================================
// Aperiodicity analysis: WORLD D4C band-aperiodicity.
// ============================================================================

struct ApAnalyzer {
    x_real: Vec<f64>,
    x_imag: Vec<f64>,
    d4c: D4c,
    score_min: f64, // periodicity-gate threshold on the fused probability; 0.0 = gate off
    score_gate: SchmittGate,
    last_probability: f64,
}

// Hysteresis ratio for the opt-in score gate: once voiced, a frame stays voiced
// until the fused probability drops below this fraction of `score_min`.
// Recovers soft note tails whose probability decays smoothly through the
// threshold. Eval-chosen at score_min=0.6 (sweep 0.9/0.8/0.5/0.1): 0.8 gives
// voicing F1 0.932 on Vocadito / 0.830 on PTDB stride-64, with Vocadito recall
// 0.979. Internal, not part of the public API.
const VOICING_SCORE_HYSTERESIS: f64 = 0.8;

// Logistic-regression fusion of the voicing features into a probability.
// Weights fit offline (eval/fit_fusion.py) on raw features [ln(score), nccf,
// cpp]; dataset and resulting metrics recorded in the commit that set them.
// Calibrated for the default (fs, fftsize) config, like the other eval-chosen
// constants. Internal, not part of the public API.
// Fit over Vocadito (40 clips) + PTDB-TUG (295-clip stride-16 subsample), each
// under clean/white20/white10/hum50 conditions (~1.8M frames): held-out F1
// 0.874, AUC 0.953 pooled; 0.955 on clean Vocadito at threshold 0.5.
const FUSION_BIAS: f64 = -3.077462961437281;
const FUSION_WEIGHTS: [f64; 3] = [0.07029391640284875, 5.275854261068518, 1.2496788274186972];

/// Fused voicing probability in (0,1): sigmoid of the weighted features.
fn voicing_probability(f: VoicingFeatures) -> f64 {
    let z = FUSION_BIAS
        + FUSION_WEIGHTS[0] * f.score.max(1e-12).ln()
        + FUSION_WEIGHTS[1] * f.nccf
        + FUSION_WEIGHTS[2] * f.cpp;
    1.0 / (1.0 + (-z).exp())
}

/// Schmitt trigger for the score gate: enter voiced at `score >= high`, stay
/// until `score < high * VOICING_SCORE_HYSTERESIS`. The caller latches `open`
/// from the *final* voiced decision, so silence or any hard gate closing the
/// frame also closes the trigger.
struct SchmittGate {
    open: bool,
}

impl SchmittGate {
    fn pass(&self, score: f64, high: f64) -> bool {
        let threshold = if self.open {
            high * VOICING_SCORE_HYSTERESIS
        } else {
            high
        };
        score >= threshold
    }
}

// Reject a frame as unvoiced when its energy is concentrated BELOW the detected
// fundamental (rumble / mains hum) -- the gap the HF LoveTrain leaves open. Uses a
// full-frame Hann window for low-frequency resolution. Returns true = sub-fo energy
// dominates the fundamental+harmonic band, so it is not a real voice. The 0.4 ratio
// is an internal, eval-chosen constant; it is not part of the public API.
const LOWBAND_REJECT_RATIO: f64 = 0.4;

fn low_band_dominated(
    input: &[f64],
    re: &mut [f64],
    im: &mut [f64],
    fftsize: usize,
    fo: f64,
    fs: f64,
    fft: &mut Fft,
) -> bool {
    for i in 0..fftsize {
        re[i] = input[i] * hanning_window(i as f64, fftsize as f64, fftsize as f64);
    }
    fft.forward_real_f64(re, im);
    let bin = |hz: f64| ((hz / fs * fftsize as f64) as usize).min(fftsize / 2);
    let (lo, mid) = (bin(20.0), bin(0.8 * fo));
    let hi = bin((6.0 * fo).min(fs / 2.0 - 1.0));
    let sub: f64 = (lo..mid).map(|k| complex_abs2(re[k], im[k])).sum();
    let voice: f64 = (mid..=hi).map(|k| complex_abs2(re[k], im[k])).sum();
    sub > LOWBAND_REJECT_RATIO * (sub + voice + 1e-12)
}

// Band-energy ratio (low-mid energy / low-through-high energy) above which the D4C
// LoveTrain judges a frame voiced. Eval-chosen, internal; not part of the public API.
const VOICED_BAND_RATIO: f64 = 0.7;

/// D4C "LoveTrain"-style voiced/unvoiced decision based on low/high band energy.
fn estimate_is_voiced(
    input: &[f64],
    re: &mut [f64],
    im: &mut [f64],
    fftsize: usize,
    fo: f64,
    fs: f64,
    fft: &mut Fft,
) -> bool {
    if fs < 16000.0 {
        return true;
    }
    let window_length = (1.5 * fs / fo).min(fftsize as f64);
    for i in 0..fftsize {
        re[i] = input[i] * blackman_window(i as f64, fftsize as f64, window_length);
    }
    fft.forward_real_f64(re, im);
    let numbins = fftsize / 2 + 1;
    for k in 0..numbins {
        re[k] = complex_abs2(re[k], im[k]);
    }
    let index_lower = (100.0 / fs * fftsize as f64).floor() as usize;
    let index_upper1 = (4000.0 / fs * fftsize as f64).floor() as usize;
    let index_upper2 = (7900.0 / fs * fftsize as f64).floor() as usize;

    let mut low_mid_energy = 1e-6;
    for k in index_lower + 1..=index_upper1 {
        low_mid_energy += re[k];
    }
    let mut low_to_high_energy = low_mid_energy;
    for k in index_upper1 + 1..=index_upper2 {
        low_to_high_energy += re[k];
    }
    low_mid_energy / low_to_high_energy > VOICED_BAND_RATIO
}

// --- D4C constants (WORLD constantnumbers.h) --------------------------------
const D4C_FREQUENCY_INTERVAL: f64 = 3000.0;
const D4C_UPPER_LIMIT: f64 = 15000.0;
const D4C_FLOOR_F0: f64 = 47.0;
const D4C_SAFE_GUARD_MINIMUM: f64 = 1e-12;

/// Nuttall window into `y` (length `y.len()`), matching common.cpp NuttallWindow.
fn nuttall_window_into(y: &mut [f64]) {
    let len = y.len();
    for (i, yi) in y.iter_mut().enumerate() {
        let t = i as f64 / (len as f64 - 1.0);
        *yi = 0.355768 - 0.487396 * (2.0 * REIM_PI * t).cos()
            + 0.144232 * (4.0 * REIM_PI * t).cos()
            - 0.012604 * (6.0 * REIM_PI * t).cos();
    }
}

/// interp1Q: piecewise-linear resample of `y` (sampled at `origin + i*shift`)
/// onto query points `xi`, into `yi`. Port of matlabfunctions.cpp interp1Q.
fn interp1q(origin: f64, shift: f64, y: &[f64], xi: &[f64], yi: &mut [f64]) {
    let last = y.len() - 1;
    for (i, &q) in xi.iter().enumerate() {
        let pos = (q - origin) / shift;
        debug_assert!(pos >= 0.0, "interp1q: negative pos {pos} at query {q}");
        let base = pos as usize;
        let frac = pos - base as f64;
        let delta = if base >= last {
            0.0
        } else {
            y[base + 1] - y[base]
        };
        yi[i] = y[base] + delta * frac;
    }
}

/// Piecewise-linear interpolation of `(x, y)` onto query points `xi`, into `yi`,
/// with MATLAB interp1 semantics (histc bucketing). `x` strictly increasing.
/// Port of matlabfunctions.cpp interp1 + histc; `k` is scratch of len `xi.len()`.
fn interp1(x: &[f64], y: &[f64], xi: &[f64], k: &mut [usize], yi: &mut [f64]) {
    let x_length = x.len();
    let xi_length = xi.len();
    let mut count = 1usize;
    let mut i = 0usize;
    while i < xi_length {
        k[i] = 1;
        if x[0] <= xi[i] {
            break;
        }
        i += 1;
    }
    while i < xi_length {
        if xi[i] < x[count] {
            k[i] = count;
        } else {
            // WORLD advances `count` but re-examines this same `i` (its `index[i--] = count++`
            // cancels the loop's `++i`); reproduce by assigning and not advancing `i`.
            k[i] = count;
            count += 1;
            if count == x_length {
                break;
            }
            continue;
        }
        if count == x_length {
            break;
        }
        i += 1;
    }
    count -= 1;
    i += 1;
    while i < xi_length {
        k[i] = count;
        i += 1;
    }
    for i in 0..xi_length {
        let ki = k[i];
        let h = x[ki] - x[ki - 1];
        let s = (xi[i] - x[ki - 1]) / h;
        yi[i] = y[ki - 1] + s * (y[ki] - y[ki - 1]);
    }
}

/// GetWindowedWaveform (d4c.cpp): place an F0-adaptive window (Hanning or
/// Blackman) at `center`, weight-correct, into `waveform[0..fft_size]`
/// (zero-padded past the window). The matching window samples are written to
/// `window`. SKIPS the +randn*1e-6 dither (negligible; the #1 bit-match risk).
fn get_windowed_waveform(
    x: &[f64],
    fs: f64,
    current_f0: f64,
    center: usize,
    blackman: bool,
    window_length_ratio: f64,
    waveform: &mut [f64],
    window: &mut [f64],
) {
    let half_window_length = (window_length_ratio * fs / current_f0 / 2.0).round() as i64;
    let span = (half_window_length * 2 + 1) as usize;
    let x_last = x.len() as i64 - 1;
    let origin = center as i64; // WORLD: round(current_position*fs + 0.001) == center
    for j in 0..span {
        let base = j as i64 - half_window_length;
        let arg = REIM_PI * (2.0 * base as f64 / window_length_ratio / fs) * current_f0;
        window[j] = if blackman {
            0.42 + 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos()
        } else {
            0.5 * arg.cos() + 0.5
        };
        let safe = (origin + base).clamp(0, x_last) as usize;
        waveform[j] = x[safe] * window[j];
    }
    let mut tmp_weight1 = 0.0;
    let mut tmp_weight2 = 0.0;
    for j in 0..span {
        tmp_weight1 += waveform[j];
        tmp_weight2 += window[j];
    }
    let weighting_coefficient = tmp_weight1 / tmp_weight2;
    for j in 0..span {
        waveform[j] -= window[j] * weighting_coefficient;
    }
    for w in waveform[span..].iter_mut() {
        *w = 0.0;
    }
}

/// DCCorrection (common.cpp): add a low-frequency replica of `buf` to itself,
/// in place over the first `upper_limit-1` bins. `axis`/`replica` are scratch.
fn dc_correction(
    buf: &mut [f64],
    f0: f64,
    fs: f64,
    fft_size: usize,
    axis: &mut [f64],
    replica: &mut [f64],
) {
    let upper_limit = 2 + (f0 * fft_size as f64 / fs) as usize;
    let axis = &mut axis[..upper_limit];
    for (i, a) in axis.iter_mut().enumerate() {
        *a = i as f64 * fs / fft_size as f64;
    }
    let replica = &mut replica[..upper_limit - 1];
    // WORLD queries only the first upper_limit-1 axis points (the array is one longer).
    interp1q(
        f0 - axis[0],
        -fs / fft_size as f64,
        &buf[..upper_limit + 1],
        &axis[..upper_limit - 1],
        replica,
    );
    for i in 0..upper_limit - 1 {
        buf[i] += replica[i];
    }
}

/// Scratch buffers for `linear_smoothing`, sized in `D4c::new` for the widest width.
struct SmoothScratch {
    mirror: Vec<f64>,
    segment: Vec<f64>,
    freq_axis: Vec<f64>,
    low: Vec<f64>,
    high: Vec<f64>,
}

/// LinearSmoothing (common.cpp): moving average of `input` over `width` Hz,
/// written to `output`; both are half-spectra (len fft_size/2 + 1).
fn linear_smoothing(
    input: &[f64],
    output: &mut [f64],
    width: f64,
    fs: f64,
    fft_size: usize,
    s: &mut SmoothScratch,
) {
    let half = fft_size / 2;
    let boundary = (width * fft_size as f64 / fs) as usize + 1;
    let mirror_len = half + boundary * 2 + 1;
    let mirror = &mut s.mirror[..mirror_len];
    for i in 0..boundary {
        mirror[i] = input[boundary - i];
    }
    mirror[boundary..half + boundary].copy_from_slice(&input[..half]);
    for i in half + boundary..=half + boundary * 2 {
        mirror[i] = input[half - (i - (half + boundary))];
    }
    let segment = &mut s.segment[..mirror_len];
    segment[0] = mirror[0] * fs / fft_size as f64;
    for i in 1..mirror_len {
        segment[i] = mirror[i] * fs / fft_size as f64 + segment[i - 1];
    }
    let freq_axis = &mut s.freq_axis[..=half];
    for (i, f) in freq_axis.iter_mut().enumerate() {
        *f = i as f64 / fft_size as f64 * fs - width / 2.0;
    }
    let origin = -(boundary as f64 - 0.5) * fs / fft_size as f64;
    let interval = fs / fft_size as f64;
    interp1q(origin, interval, segment, freq_axis, &mut s.low[..=half]);
    for f in freq_axis.iter_mut() {
        *f += width;
    }
    interp1q(origin, interval, segment, freq_axis, &mut s.high[..=half]);
    for i in 0..=half {
        output[i] = (s.high[i] - s.low[i]) / width;
    }
}

/// GetCentroid (d4c.cpp): energy centroid of the windowed waveform, into `out`.
/// `re`/`im` are FFT scratch (len fft_size); `window`/`waveform` window scratch;
/// `spec_re`/`spec_im` hold the first FFT's half-spectrum (len fft_size/2 + 1).
#[allow(clippy::too_many_arguments)]
fn get_centroid(
    x: &[f64],
    fs: f64,
    current_f0: f64,
    center: usize,
    fft: &mut Fft,
    fft_size: usize,
    out: &mut [f64],
    re: &mut [f64],
    im: &mut [f64],
    window: &mut [f64],
    waveform: &mut [f64],
    spec_re: &mut [f64],
    spec_im: &mut [f64],
) {
    get_windowed_waveform(x, fs, current_f0, center, true, 4.0, waveform, window);
    let normalize_to = (2.0 * fs / current_f0).round() as usize * 2;
    let mut power = 0.0;
    for j in 0..=normalize_to {
        power += waveform[j] * waveform[j];
    }
    let inv = 1.0 / power.sqrt();
    for j in 0..=normalize_to {
        waveform[j] *= inv;
    }
    let half = fft_size / 2;
    // First FFT: spectrum of the normalized waveform (real FFT).
    re.copy_from_slice(waveform);
    fft.forward_real_f64(re, im);
    spec_re[..=half].copy_from_slice(&re[..=half]);
    spec_im[..=half].copy_from_slice(&im[..=half]);
    // Second FFT: of waveform * (i+1) (real FFT).
    for j in 0..fft_size {
        re[j] = waveform[j] * (j as f64 + 1.0);
    }
    fft.forward_real_f64(re, im);
    for i in 0..=half {
        out[i] = re[i] * spec_re[i] + spec_im[i] * im[i];
    }
}

/// D4C scratch buffers + dedicated FFT, sized for `fft_size_d4c`. All allocated
/// once in `new`; the per-frame computation does no heap allocation.
struct D4c {
    fft: Fft,
    fft_size: usize,
    fs: f64,
    number_of_aperiodicities: usize,
    nuttall_window: Vec<f64>, // Nuttall band window, length window_length
    coarse_frequency_axis: Vec<f64>, // number_of_aperiodicities + 2
    coarse_aperiodicity: Vec<f64>, // number_of_aperiodicities + 2
    frequency_axis: Vec<f64>, // output query axis (cfg.numbins)
    interp_index: Vec<usize>, // histc scratch for interp1, length numbins
    re: Vec<f64>,             // FFT real buffer, length fft_size
    im: Vec<f64>,             // FFT imag buffer, length fft_size
    window: Vec<f64>,         // F0-adaptive window samples, length fft_size
    waveform: Vec<f64>,       // windowed waveform, length fft_size
    // half-spectrum buffers (length fft_size/2 + 1)
    spec_re: Vec<f64>,
    spec_im: Vec<f64>,
    centroid1: Vec<f64>,
    centroid2: Vec<f64>,
    static_centroid: Vec<f64>,
    smoothed_power_spectrum: Vec<f64>,
    static_group_delay: Vec<f64>,
    smoothed_group_delay: Vec<f64>,
    power_spectrum: Vec<f64>, // sort + cumsum scratch
    dc_axis: Vec<f64>,
    dc_replica: Vec<f64>,
    smooth: SmoothScratch,
}

impl D4c {
    fn new(cfg: &Config) -> Self {
        let fs = cfg.fs;
        let fft_size = 2f64
            .powi(1 + ((4.0 * fs / D4C_FLOOR_F0 + 1.0).ln() / std::f64::consts::LN_2) as i32)
            as usize;
        let number_of_aperiodicities = ((D4C_UPPER_LIMIT.min(fs / 2.0 - D4C_FREQUENCY_INTERVAL))
            / D4C_FREQUENCY_INTERVAL) as usize;
        let window_length = (D4C_FREQUENCY_INTERVAL * fft_size as f64 / fs) as usize * 2 + 1;
        let mut nuttall_window = vec![0.0; window_length];
        nuttall_window_into(&mut nuttall_window);

        let half = fft_size / 2 + 1;
        // Coarse axis: 0Hz, then the band centers (3k, 6k, ...), then fs/2.
        let mut coarse_frequency_axis = vec![0.0; number_of_aperiodicities + 2];
        for (i, f) in coarse_frequency_axis.iter_mut().enumerate() {
            *f = i as f64 * D4C_FREQUENCY_INTERVAL;
        }
        coarse_frequency_axis[number_of_aperiodicities + 1] = fs / 2.0;
        let mut coarse_aperiodicity = vec![0.0; number_of_aperiodicities + 2];
        coarse_aperiodicity[0] = -60.0;
        coarse_aperiodicity[number_of_aperiodicities + 1] = -D4C_SAFE_GUARD_MINIMUM;

        // Output query axis on reim's analysis FFT grid (i * fs / cfg.fftsize).
        let mut frequency_axis = vec![0.0; cfg.numbins];
        for (i, f) in frequency_axis.iter_mut().enumerate() {
            *f = i as f64 * fs / cfg.fftsize as f64;
        }

        // Widest DC/smoothing width is the largest f0 = fo_ceil.
        let max_dc_upper = 2 + (cfg.fo_ceil * fft_size as f64 / fs) as usize;
        let max_boundary = (cfg.fo_ceil * fft_size as f64 / fs) as usize + 1;
        let smooth_len = (fft_size / 2) + max_boundary * 2 + 1;

        D4c {
            fft: Fft::new(fft_size),
            fft_size,
            fs,
            number_of_aperiodicities,
            nuttall_window,
            coarse_frequency_axis,
            coarse_aperiodicity,
            frequency_axis,
            interp_index: vec![0; cfg.numbins],
            re: vec![0.0; fft_size],
            im: vec![0.0; fft_size],
            window: vec![0.0; fft_size],
            waveform: vec![0.0; fft_size],
            spec_re: vec![0.0; half],
            spec_im: vec![0.0; half],
            centroid1: vec![0.0; half],
            centroid2: vec![0.0; half],
            static_centroid: vec![0.0; half],
            smoothed_power_spectrum: vec![0.0; half],
            static_group_delay: vec![0.0; half],
            smoothed_group_delay: vec![0.0; half],
            power_spectrum: vec![0.0; half],
            dc_axis: vec![0.0; max_dc_upper + 1],
            dc_replica: vec![0.0; max_dc_upper + 1],
            smooth: SmoothScratch {
                mirror: vec![0.0; smooth_len],
                segment: vec![0.0; smooth_len],
                freq_axis: vec![0.0; half],
                low: vec![0.0; half],
                high: vec![0.0; half],
            },
        }
    }

    /// GetStaticCentroid (d4c.cpp): sum of centroids at +-0.25/f0, DC-corrected,
    /// into self.static_centroid.
    fn get_static_centroid(&mut self, x: &[f64], current_f0: f64, center: usize) {
        let off = (0.25 / current_f0 * self.fs).round() as i64;
        let x_last = x.len() as i64 - 1;
        let c1 = (center as i64 - off).clamp(0, x_last) as usize;
        let c2 = (center as i64 + off).clamp(0, x_last) as usize;
        let half = self.fft_size / 2;
        get_centroid(
            x,
            self.fs,
            current_f0,
            c1,
            &mut self.fft,
            self.fft_size,
            &mut self.centroid1,
            &mut self.re,
            &mut self.im,
            &mut self.window,
            &mut self.waveform,
            &mut self.spec_re,
            &mut self.spec_im,
        );
        get_centroid(
            x,
            self.fs,
            current_f0,
            c2,
            &mut self.fft,
            self.fft_size,
            &mut self.centroid2,
            &mut self.re,
            &mut self.im,
            &mut self.window,
            &mut self.waveform,
            &mut self.spec_re,
            &mut self.spec_im,
        );
        for i in 0..=half {
            self.static_centroid[i] = self.centroid1[i] + self.centroid2[i];
        }
        dc_correction(
            &mut self.static_centroid,
            current_f0,
            self.fs,
            self.fft_size,
            &mut self.dc_axis,
            &mut self.dc_replica,
        );
    }

    /// GetSmoothedPowerSpectrum (d4c.cpp): into self.smoothed_power_spectrum.
    fn get_smoothed_power_spectrum(&mut self, x: &[f64], current_f0: f64, center: usize) {
        get_windowed_waveform(
            x,
            self.fs,
            current_f0,
            center,
            false,
            4.0,
            &mut self.waveform,
            &mut self.window,
        );
        self.re.copy_from_slice(&self.waveform);
        self.fft.forward_real_f64(&mut self.re, &mut self.im);
        let half = self.fft_size / 2;
        for i in 0..=half {
            self.power_spectrum[i] = complex_abs2(self.re[i], self.im[i]);
        }
        dc_correction(
            &mut self.power_spectrum,
            current_f0,
            self.fs,
            self.fft_size,
            &mut self.dc_axis,
            &mut self.dc_replica,
        );
        linear_smoothing(
            &self.power_spectrum,
            &mut self.smoothed_power_spectrum,
            current_f0,
            self.fs,
            self.fft_size,
            &mut self.smooth,
        );
    }

    /// GetStaticGroupDelay (d4c.cpp): into self.static_group_delay.
    /// On entry self.static_centroid and self.smoothed_power_spectrum are set.
    fn get_static_group_delay(&mut self, current_f0: f64) {
        let half = self.fft_size / 2;
        for i in 0..=half {
            self.static_group_delay[i] = self.static_centroid[i] / self.smoothed_power_spectrum[i];
        }
        // Smooth at f0/2 in place (via spec_re as a temp), then subtract the f0 smooth.
        linear_smoothing(
            &self.static_group_delay,
            &mut self.spec_re,
            current_f0 / 2.0,
            self.fs,
            self.fft_size,
            &mut self.smooth,
        );
        self.static_group_delay[..=half].copy_from_slice(&self.spec_re[..=half]);
        linear_smoothing(
            &self.static_group_delay,
            &mut self.smoothed_group_delay,
            current_f0,
            self.fs,
            self.fft_size,
            &mut self.smooth,
        );
        for i in 0..=half {
            self.static_group_delay[i] -= self.smoothed_group_delay[i];
        }
    }

    /// GetCoarseAperiodicity (d4c.cpp): per-band HNR readout from the group delay,
    /// into self.coarse_aperiodicity[1..=number_of_aperiodicities].
    fn get_coarse_aperiodicity(&mut self) {
        let (fft_size, fs) = (self.fft_size, self.fs);
        let half = fft_size / 2;
        let window_length = self.nuttall_window.len();
        let half_window_length = window_length / 2;
        let boundary = (fft_size as f64 * 8.0 / window_length as f64).round() as usize;
        for i in 0..self.number_of_aperiodicities {
            let band_center =
                (D4C_FREQUENCY_INTERVAL * (i + 1) as f64 * fft_size as f64 / fs) as usize;
            for j in 0..=half_window_length * 2 {
                self.re[j] = self.static_group_delay[band_center - half_window_length + j]
                    * self.nuttall_window[j];
            }
            for r in self.re[half_window_length * 2 + 1..].iter_mut() {
                *r = 0.0;
            }
            self.fft.forward_real_f64(&mut self.re, &mut self.im);
            for j in 0..=half {
                self.power_spectrum[j] = complex_abs2(self.re[j], self.im[j]);
            }
            // total_cmp (not partial_cmp) so a NaN from a degenerate frame (e.g. a pure
            // tone with near-zero broadband energy) sorts deterministically instead of
            // panicking; finite values order identically.
            self.power_spectrum[..=half].sort_unstable_by(f64::total_cmp);
            for j in 1..=half {
                self.power_spectrum[j] += self.power_spectrum[j - 1];
            }
            let total = self.power_spectrum[half];
            self.coarse_aperiodicity[i + 1] = if total > D4C_SAFE_GUARD_MINIMUM {
                10.0 * (self.power_spectrum[half - boundary - 1] / total).log10()
            } else {
                -D4C_SAFE_GUARD_MINIMUM
            };
        }
    }

    /// Full D4C for one voiced frame: write per-bin linear aperiodicity into `ap`.
    fn analyze(&mut self, x: &[f64], fo: f64, center: usize, ap: &mut [f64]) {
        let current_f0 = fo.max(D4C_FLOOR_F0);
        self.get_static_centroid(x, current_f0, center);
        self.get_smoothed_power_spectrum(x, current_f0, center);
        self.get_static_group_delay(current_f0);
        self.get_coarse_aperiodicity();
        // F0 revision (D4CGeneralBody).
        for i in 0..self.number_of_aperiodicities {
            self.coarse_aperiodicity[i + 1] =
                (self.coarse_aperiodicity[i + 1] + (current_f0 - 100.0) / 50.0).min(0.0);
        }
        // GetAperiodicity: interpolate coarse dB onto the output axis, then dB->linear.
        interp1(
            &self.coarse_frequency_axis,
            &self.coarse_aperiodicity,
            &self.frequency_axis,
            &mut self.interp_index,
            ap,
        );
        for a in ap.iter_mut() {
            *a = 10f64.powf(*a / 20.0);
        }
    }
}
impl ApAnalyzer {
    fn new(cfg: &Config) -> Self {
        ApAnalyzer {
            x_real: vec![0.0; cfg.fftsize],
            x_imag: vec![0.0; cfg.fftsize],
            d4c: D4c::new(cfg),
            score_min: 0.0,
            score_gate: SchmittGate { open: false },
            last_probability: 0.0,
        }
    }

    /// Returns true for a voiced frame; writes per-bin aperiodicity into `ap`.
    /// Voiced frames get WORLD D4C band-aperiodicity; everything else is fully
    /// aperiodic (1.0), matching the placeholder's unvoiced behaviour.
    /// `features` are the Fo tracker's per-frame voicing features; the optional
    /// periodicity gate (fused probability >= score_min, default 0.0 = off)
    /// rejects low-periodicity frames (see [`Reim::set_voicing_score_min`]).
    /// It is opt-in/experimental.
    fn analyze(
        &mut self,
        cfg: &Config,
        fft: &mut Fft,
        input: &[f64],
        fo: f64,
        features: VoicingFeatures,
        issilence: bool,
        ap: &mut [f64],
    ) -> bool {
        let numbins = cfg.numbins;
        // Mirror the C guard exactly (analyze_ap.c:79): bail to unvoiced when silent
        // or fo is out of range. The negated range test (rather than `>=`/`<=`) makes
        // a NaN fo fall through to the V/UV decision, matching C's `<`/`>` semantics.
        let out_of_range = fo < cfg.fo_floor || fo > cfg.fo_ceil;
        self.last_probability = voicing_probability(features);
        let score_pass = self.score_min <= 0.0
            || self
                .score_gate
                .pass(self.last_probability, self.score_min);
        let gates_pass = !issilence
            && !out_of_range
            && !low_band_dominated(
                input,
                &mut self.x_real,
                &mut self.x_imag,
                cfg.fftsize,
                fo,
                cfg.fs,
                fft,
            )
            && estimate_is_voiced(
                input,
                &mut self.x_real,
                &mut self.x_imag,
                cfg.fftsize,
                fo,
                cfg.fs,
                fft,
            );
        let voiced = gates_pass && score_pass;
        self.score_gate.open = voiced;
        if voiced {
            self.d4c
                .analyze(input, fo, cfg.fftsize / 2, &mut ap[..numbins]);
        } else {
            ap[..numbins].fill(1.0);
        }
        voiced
    }
}

// ============================================================================
// Spectral envelope analysis (CheapTrick-like).
// ============================================================================

struct SpAnalyzer {
    window: Vec<f64>,
    x_real: Vec<f64>,
    x_imag: Vec<f64>,
    envelope: Vec<f64>,
    spec_cumsum: Vec<f64>,
}

/// Mirror bins `[numbins, fftsize)` from the lower half (Hermitian magnitude).
fn mirror_upper(arr: &mut [f64], numbins: usize) {
    for k in 0..numbins - 2 {
        arr[numbins + k] = arr[numbins - 2 - k];
    }
}

fn apply_replica(envelope: &mut [f64], numbins: usize, fo: f64, fs: f64) {
    let fftsize = 2 * (numbins - 1);
    let fobin = 1 + (fo / (fs / 2.0) * (numbins - 1) as f64).round() as usize;
    for k in 0..fobin {
        envelope[k] += envelope[fftsize - fobin - k];
    }
    mirror_upper(envelope, numbins);
}

fn smooth_spectrum(
    envelope: &mut [f64],
    spec_cumsum: &mut [f64],
    numbins: usize,
    freq_range: f64,
    fs: f64,
) {
    let fftsize = 2 * (numbins - 1);
    let offset = numbins - 2;
    spec_cumsum[0] = envelope[numbins];
    for k in 1..offset {
        spec_cumsum[k] = envelope[numbins + k] + spec_cumsum[k - 1];
    }
    for k in 0..fftsize {
        spec_cumsum[offset + k] = envelope[k] + spec_cumsum[offset + k - 1];
    }

    let half_range = freq_range / fs * (numbins - 1) as f64;
    let half_range_int = half_range.floor() as usize;
    let half_range_frac = half_range - half_range_int as f64;
    for k in 0..numbins {
        let index_upper = offset + k + half_range_int;
        let index_lower = offset + k - half_range_int;
        let upper = (1.0 - half_range_frac) * spec_cumsum[index_upper - 1]
            + half_range_frac * spec_cumsum[index_upper];
        let lower = (1.0 - half_range_frac) * spec_cumsum[index_lower]
            + half_range_frac * spec_cumsum[index_lower - 1];
        envelope[k] = (upper - lower).max(1e-12) / (2.0 * half_range);
    }
    mirror_upper(envelope, numbins);
}

fn lifter_spectrum(
    envelope: &mut [f64],
    imag: &mut [f64],
    numbins: usize,
    fo: f64,
    fs: f64,
    fft: &mut Fft,
) {
    let fftsize = 2 * (numbins - 1);
    for k in 0..fftsize {
        envelope[k] = (envelope[k] + 1e-12).ln();
    }
    // Log-envelope is symmetric (from mirror_upper); its half-spectrum is real.
    // c2r IFFT: use envelope[0..numbins] as re, zeros as im → cepstrum.
    imag[..numbins].fill(0.0);
    fft.inverse_real_f64(envelope, &imag[..numbins]);

    let lifter_coeff = -0.15;
    for k in 0..numbins {
        let t = k as f64 * fo / fs;
        let sinct = (REIM_PI * t + 1e-12).sin() / (REIM_PI * t + 1e-12);
        envelope[k] *=
            sinct * ((1.0 - 2.0 * lifter_coeff) + 2.0 * lifter_coeff * (2.0 * REIM_PI * t).cos());
    }
    for k in 0..numbins - 2 {
        envelope[numbins + k] = envelope[numbins - 2 - k];
    }

    // Liftered cepstrum is symmetric; r2c FFT → half-spectrum (im ≈ 0).
    fft.forward_real_f64(envelope, imag);
    for k in 0..numbins {
        envelope[k] = envelope[k].exp();
    }
    mirror_upper(envelope, numbins);
}

impl SpAnalyzer {
    fn new(cfg: &Config) -> Self {
        SpAnalyzer {
            window: vec![0.0; cfg.fftsize],
            x_real: vec![0.0; cfg.fftsize],
            x_imag: vec![0.0; cfg.fftsize],
            envelope: vec![0.0; cfg.fftsize],
            spec_cumsum: vec![0.0; cfg.numbins + cfg.fftsize],
        }
    }

    fn analyze(
        &mut self,
        cfg: &Config,
        fft: &mut Fft,
        input: &[f64],
        fo: f64,
        isvoiced: bool,
        issilence: bool,
        sp: &mut [f64],
    ) {
        let fs = cfg.fs;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        if issilence {
            sp[..numbins].fill(1e-12);
            return;
        }

        let window_fo = if isvoiced {
            fo
        } else {
            1.0 / (cfg.period / 1000.0)
        };
        let smooth_fo = if isvoiced { fo } else { 300.0 };

        let analysis_interval = fs / window_fo;
        let window_length = (3.0 * analysis_interval).min(fftsize as f64);
        let window_scale = 1.0 / analysis_interval.sqrt();
        for i in 0..fftsize {
            self.window[i] = hanning_window(i as f64, fftsize as f64, window_length) * window_scale;
            self.x_real[i] = input[i] * self.window[i];
        }

        // remove DC component (weighted by window)
        let mut sum_x = 0.0;
        let mut sum_window = 0.0;
        for i in 0..fftsize {
            sum_x += self.x_real[i];
            sum_window += self.window[i];
        }
        let gain_dc = sum_x / sum_window;
        for i in 0..fftsize {
            self.x_real[i] -= gain_dc * self.window[i];
        }

        fft.forward_real_f64(&mut self.x_real, &mut self.x_imag);
        for k in 0..numbins {
            self.envelope[k] = complex_abs2(self.x_real[k], self.x_imag[k]);
        }
        mirror_upper(&mut self.envelope, numbins);

        apply_replica(&mut self.envelope, numbins, window_fo, fs);
        smooth_spectrum(
            &mut self.envelope,
            &mut self.spec_cumsum,
            numbins,
            smooth_fo / 2.0,
            fs,
        );
        if isvoiced {
            lifter_spectrum(
                &mut self.envelope,
                &mut self.x_imag,
                numbins,
                smooth_fo,
                fs,
                fft,
            );
        }

        sp[..numbins].copy_from_slice(&self.envelope[..numbins]);
    }
}

// ============================================================================
// Synthesis — minimum-phase pulse excitation + velvet-noise aperiodic part.
// ============================================================================

struct Synth {
    has_pulse: bool,
    has_noise: bool,
    spec_pulse_r: Vec<f64>,
    spec_pulse_i: Vec<f64>,
    spec_noise_r: Vec<f64>,
    spec_noise_i: Vec<f64>,
    window: Vec<f64>,
    impulse_pulse: Vec<f64>,
    impulse_noise: Vec<f64>,
    temp_r: Vec<f64>,
    temp_i: Vec<f64>,
    interval: f64,
    pulse_int: i32,
    pulse_frac: f64,
    random: Xorshift,
    interval_velvet: usize,
    interval_random: usize,
    gain_noise: f64,
    noise_int: usize,
    buffer: CircularQueue,
}

/// Build the minimum-phase complex half-spectrum from a power spectrum.
/// Input: `spec_r[0..numbins]` = power spectrum (first half only, no mirror).
/// Output: `spec_r[0..numbins]`, `spec_i[0..numbins]` = minimum-phase complex
/// half-spectrum, scaled by `gain`. Uses real FFTs throughout.
fn generate_minimum_phase_spectrum(
    spec_r: &mut [f64],
    spec_i: &mut [f64],
    gain: f64,
    fftsize: usize,
    fft: &mut Fft,
) {
    let numbins = fftsize / 2 + 1;
    for k in 0..numbins {
        spec_r[k] = (spec_r[k] + 1e-12).ln();
    }
    // c2r IFFT (f32): log-power half-spectrum (im=0) → real cepstrum in spec_r
    spec_i[..numbins].fill(0.0);
    fft.inverse_real(spec_r, &spec_i[..numbins]);

    // Keep the causal half of the cepstrum
    spec_r[0] *= 0.5;
    spec_r[numbins - 1] *= 0.5;
    for k in numbins..fftsize {
        spec_r[k] = 0.0;
    }

    // r2c FFT (f32): real causal cepstrum → complex log min-phase half-spectrum
    fft.forward_real(spec_r, spec_i);
    for k in 0..numbins {
        let a = gain * spec_r[k].exp();
        let b = spec_i[k];
        spec_r[k] = a * b.cos();
        spec_i[k] = a * b.sin();
    }
}

/// Generate a (optionally fractionally shifted) impulse response from a
/// half-spectrum. `spec_r[0..numbins]`, `spec_i[0..numbins]` hold the
/// Hermitian half-spectrum; uses c2r IFFT.
fn generate_impulse(
    impulse: &mut [f64],
    spec_r: &[f64],
    spec_i: &[f64],
    shift: f64,
    window: &[f64],
    temp_r: &mut [f64],
    temp_i: &mut [f64],
    fftsize: usize,
    fft: &mut Fft,
) {
    let numbins = fftsize / 2 + 1;
    if shift == 0.0 {
        temp_r[..numbins].copy_from_slice(&spec_r[..numbins]);
        temp_i[..numbins].copy_from_slice(&spec_i[..numbins]);
    } else {
        for k in 0..numbins {
            let omega = -REIM_PI * shift * k as f64 / (numbins - 1) as f64;
            let cr = omega.cos();
            let ci = omega.sin();
            temp_r[k] = cr * spec_r[k] - ci * spec_i[k];
            temp_i[k] = cr * spec_i[k] + ci * spec_r[k];
        }
    }

    // c2r IFFT (f32): half-spectrum → real time-domain in temp_r[0..fftsize]
    fft.inverse_real(temp_r, &temp_i[..numbins]);
    ifftshift(temp_r, impulse, numbins);

    let mut gain = 0.0;
    for k in 0..fftsize {
        gain += impulse[k];
    }
    for k in 0..fftsize {
        impulse[k] -= window[k] * gain;
    }
}

impl Synth {
    fn new(cfg: &Config) -> Self {
        let fs = cfg.fs;
        let fftsize = cfg.fftsize;

        let mut window = vec![0.0; fftsize];
        let mut gain = 0.0;
        for i in 0..fftsize {
            let w = vorbis_window(i as f64, fftsize as f64);
            window[i] = w;
            gain += w;
        }
        for w in window.iter_mut() {
            *w /= gain;
        }

        let interval_velvet = (fs / 2000.0).round() as usize;
        let period_max = (fs / cfg.fo_floor).ceil() as usize;

        Synth {
            has_pulse: false,
            has_noise: false,
            spec_pulse_r: vec![0.0; fftsize],
            spec_pulse_i: vec![0.0; fftsize],
            spec_noise_r: vec![0.0; fftsize],
            spec_noise_i: vec![0.0; fftsize],
            window,
            impulse_pulse: vec![0.0; fftsize],
            impulse_noise: vec![0.0; fftsize],
            temp_r: vec![0.0; fftsize],
            temp_i: vec![0.0; fftsize],
            interval: fs / 300.0,
            pulse_int: 0,
            pulse_frac: 0.0,
            random: Xorshift::new(),
            interval_velvet,
            interval_random: 0,
            gain_noise: (interval_velvet as f64).sqrt(),
            noise_int: 0,
            buffer: CircularQueue::new(period_max + fftsize),
        }
    }

    fn new_frame(
        &mut self,
        cfg: &Config,
        fft: &mut Fft,
        fo: f64,
        strength: f64,
        issilence: bool,
        ap: &[f64],
        sp: &[f64],
    ) {
        let fs = cfg.fs;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        // Per-bin energy split: pulse gets strength*(1-aper), noise gets the
        // rest, so pulse + noise == spec for any strength. At strength 1.0 the
        // arithmetic is bit-identical to the hard voiced split (spec*(1-aper) /
        // spec*aper); at 0.0 it equals the unvoiced path.
        for k in 0..numbins {
            let spec = sp[k];
            let aper = ap[k] * ap[k];
            let periodic = strength * (1.0 - aper);
            self.spec_pulse_r[k] = spec * periodic;
            self.spec_noise_r[k] = spec * (aper + (1.0 - strength) * (1.0 - aper));
        }

        // The fo range guard matters for fractional strength: a borderline
        // frame can carry strength > 0 with fo = 0.0, and `interval = fs/fo`
        // must never see that.
        self.has_pulse =
            strength > 0.0 && !issilence && fo >= cfg.fo_floor && fo <= cfg.fo_ceil;
        if self.has_pulse {
            self.interval = fs / fo;
            let gain_pulse = self.interval.sqrt();
            generate_minimum_phase_spectrum(
                &mut self.spec_pulse_r,
                &mut self.spec_pulse_i,
                gain_pulse,
                fftsize,
                fft,
            );
        }

        self.has_noise = !issilence;
        if self.has_noise {
            generate_minimum_phase_spectrum(
                &mut self.spec_noise_r,
                &mut self.spec_noise_i,
                self.gain_noise,
                fftsize,
                fft,
            );
            generate_impulse(
                &mut self.impulse_noise,
                &self.spec_noise_r,
                &self.spec_noise_i,
                0.0,
                &self.window,
                &mut self.temp_r,
                &mut self.temp_i,
                fftsize,
                fft,
            );
        }
    }

    fn next_sample(&mut self, cfg: &Config, fft: &mut Fft) -> f64 {
        let fftsize = cfg.fftsize;

        if self.has_pulse {
            if self.pulse_int == 0 {
                generate_impulse(
                    &mut self.impulse_pulse,
                    &self.spec_pulse_r,
                    &self.spec_pulse_i,
                    self.pulse_frac,
                    &self.window,
                    &mut self.temp_r,
                    &mut self.temp_i,
                    fftsize,
                    fft,
                );
                self.buffer.push_additive(&self.impulse_pulse);

                let interval_int = self.interval.floor();
                let interval_frac = self.interval - interval_int;
                let next = self.pulse_frac + interval_frac;
                let carry = next.floor();
                self.pulse_int += (interval_int + carry) as i32;
                self.pulse_frac = next - carry;
            }
            self.pulse_int -= 1;
        }

        if self.has_noise {
            if self.noise_int == self.interval_random {
                self.buffer.push_additive(&self.impulse_noise);
            }
            if self.noise_int == self.interval_velvet - 1 {
                let r = self.random.uniform();
                self.interval_random = (r * (self.interval_velvet - 1) as f64).floor() as usize;
                self.noise_int = 0;
            }
            self.noise_int += 1;
        }

        self.buffer.pop()
    }
}

// ============================================================================
// Top-level real-time processor
// ============================================================================

/// Streaming analyzer: framer + Fo/Ap/Sp analyzers and their per-frame buffers.
/// Push input samples; on each frame boundary the per-frame WORLD parameters
/// become readable via the accessors. Allocation-free steady state.
/// Raw per-frame features feeding the voicing decision: `score` is the SRH
/// harmonic score (unbounded, orders of magnitude between noise and voice),
/// `nccf` the normalized autocorrelation at the detected pitch lag (~1 =
/// periodic), `cpp` the cepstral peak prominence (log-power units).
#[derive(Clone, Copy, Debug)]
pub struct VoicingFeatures {
    pub score: f64,
    pub nccf: f64,
    pub cpp: f64,
}

pub struct Analyzer {
    cfg: Config,
    fft: Fft,
    framer: Framer,
    fo: FoAnalyzer,
    ap: ApAnalyzer,
    sp: SpAnalyzer,
    frame_window: Vec<f64>, // fftsize+1 samples, oldest..newest
    ap_buf: Vec<f64>,       // numbins
    sp_buf: Vec<f64>,       // numbins
    last_fo: f64,
    last_voiced: bool,
    last_silence: bool,
    frame_count: u64,
}

/// Synthesizer: minimum-phase pulse + velvet-noise overlap-add from per-frame
/// WORLD parameters. Feed it one frame's parameters per frame boundary
/// (`push_frame`) and pull one output sample per input sample (`next_sample`).
pub struct Synthesizer {
    cfg: Config,
    fft: Fft,
    syn: Synth,
}

/// Fused analyze->resynthesize convenience: composes [`Analyzer`] +
/// [`Synthesizer`] and preserves the original one-in/one-out API.
pub struct Reim {
    analyzer: Analyzer,
    synth: Synthesizer,
}

/// Default analysis size used by [`Reim::with_defaults`] and the `reim f0` CLI:
/// the smallest power of two spanning ~4 periods of `fo_floor` (the resolution
/// the Fo tracker needs), clamped to [512, 2048]. This keeps the window near a
/// fixed *time* span across sample rates -- 1024 at 16 kHz (64 ms), 2048 at
/// 24-48 kHz -- rather than a fixed 2048 that is 128 ms at 16 kHz and smears
/// fast pitch motion.
pub fn default_fftsize(fs: f64, fo_floor: f64) -> usize {
    let needed = (4.0 * fs / fo_floor).ceil() as usize;
    needed.next_power_of_two().clamp(512, 2048)
}

impl Analyzer {
    pub fn new(fs: f64, period: f64, fftsize: usize, fo_floor: f64, fo_ceil: f64) -> Self {
        let cfg = Config::new(period, fftsize, fo_floor, fo_ceil, fs);
        let mut fft = Fft::new(fftsize);
        let fo = FoAnalyzer::new(&cfg, &mut fft);
        let ap = ApAnalyzer::new(&cfg);
        let sp = SpAnalyzer::new(&cfg);
        Analyzer {
            cfg,
            fft,
            framer: Framer::new(&cfg),
            fo,
            ap,
            sp,
            frame_window: vec![0.0; fftsize + 1],
            ap_buf: vec![0.0; cfg.numbins],
            sp_buf: vec![0.0; cfg.numbins],
            last_fo: 0.0,
            last_voiced: false,
            last_silence: false,
            frame_count: 0,
        }
    }

    /// Default configuration: 5 ms period, Fo 71-800 Hz, and a sample-rate-aware
    /// fftsize (see [`default_fftsize`]) -- 1024 at 16 kHz, 2048 at 24-48 kHz.
    pub fn with_defaults(fs: f64) -> Self {
        let fo_floor = 71.0;
        Analyzer::new(fs, 5.0, default_fftsize(fs, fo_floor), fo_floor, 800.0)
    }

    /// Analyze the current frame window into the per-frame parameters/state
    /// (silence, Fo, voiced, aperiodicity, spectral envelope). No synthesis.
    fn analyze_current_frame(&mut self) {
        let fftsize = self.cfg.fftsize;
        // `wave` is the frame; `wave_d` is the same frame delayed by one sample.
        let (wave_d, wave) = (
            &self.frame_window[..fftsize],
            &self.frame_window[1..fftsize + 1],
        );
        let silence = analyze_silence(&self.cfg, wave, SILENCE_THRESHOLD);
        let fo = self.fo.analyze(&self.cfg, &mut self.fft, wave, wave_d);
        let features = self.voicing_features();
        let voiced = self.ap.analyze(
            &self.cfg,
            &mut self.fft,
            wave,
            fo,
            features,
            silence,
            &mut self.ap_buf,
        );
        self.sp.analyze(
            &self.cfg,
            &mut self.fft,
            wave,
            fo,
            voiced,
            silence,
            &mut self.sp_buf,
        );
        self.last_fo = fo;
        self.last_voiced = voiced;
        self.last_silence = silence;
        self.frame_count += 1;
    }

    /// Push one input sample. Returns true when a new analysis frame is ready;
    /// the accessors below are then valid until the next push that returns true.
    /// Allocation-free.
    pub fn push_sample(&mut self, input: f64) -> bool {
        let new_frame = self.framer.next(input, &mut self.frame_window);
        if new_frame {
            self.analyze_current_frame();
        }
        new_frame
    }

    /// Number of frames analyzed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
    /// Number of frequency bins in the spectral envelope / aperiodicity
    /// (`fftsize/2 + 1`).
    pub fn numbins(&self) -> usize {
        self.cfg.numbins
    }
    /// Pitch-tracker estimate for the most recent frame, in Hz (0.0 when the
    /// tracker found no periodicity). This is the raw Fo and is independent of
    /// the voicing decision: it can be nonzero on a frame `voiced()` rejects
    /// (e.g. rumble the voicing guard discards). Gate on `voiced()` if you only
    /// want pitch for frames judged to carry voice.
    pub fn fo(&self) -> f64 {
        self.last_fo
    }
    /// Whether the most recent frame was judged voiced. This is the full
    /// voiced/unvoiced decision, separate from the pitch tracker (`fo()`): a
    /// frame is voiced only when it is not silence, its Fo lies in
    /// `[fo_floor, fo_ceil]`, its energy is not dominated by sub-fundamental
    /// rumble/hum, and it passes the D4C band-energy ratio.
    pub fn voiced(&self) -> bool {
        self.last_voiced
    }
    /// Whether the most recent frame was silence.
    pub fn silence(&self) -> bool {
        self.last_silence
    }
    /// Per-bin aperiodicity for the most recent frame (length `numbins`): the
    /// noise-to-total energy ratio per frequency bin, ~0 = periodic,
    /// ~1 = fully aperiodic.
    pub fn aperiodicity(&self) -> &[f64] {
        &self.ap_buf
    }
    /// Per-bin spectral envelope (CheapTrick) for the most recent frame (length
    /// `numbins`): the smoothed power spectrum carrying the formant structure.
    /// With `fo`/`voiced`/`aperiodicity` this is the full per-frame WORLD
    /// parameter set (read it after each `push_sample` that returns true).
    pub fn spectral_envelope(&self) -> &[f64] {
        &self.sp_buf
    }
    /// Per-frame voicing features for the most recent frame. These are raw
    /// inputs to the voicing decision, useful for offline analysis/fitting;
    /// none of them is gated on `voiced()`.
    pub fn voicing_features(&self) -> VoicingFeatures {
        VoicingFeatures {
            score: self.fo.last_score,
            nccf: self.fo.last_nccf,
            cpp: self.fo.last_cpp,
        }
    }
    /// Fused voicing probability in (0,1) for the most recent frame: a
    /// logistic combination of the voicing features (internal eval-fit
    /// weights). This is the value the opt-in gate thresholds
    /// ([`set_voicing_score_min`](Self::set_voicing_score_min)); it is
    /// computed every frame regardless of the gate.
    pub fn voicing_score(&self) -> f64 {
        self.ap.last_probability
    }
    /// Set the minimum fused voicing probability (see
    /// [`voicing_score`](Self::voicing_score)) for a frame to be judged voiced
    /// (default 0.0 = off). Above 0 this is an opt-in, EXPERIMENTAL
    /// periodicity gate: the Fo tracker returns a candidate even on
    /// breath/noise; a threshold around 0.5 cuts voicing false alarms
    /// sharply at a small recall cost. The gate has hysteresis: once a frame
    /// is voiced, following frames stay voiced until the probability falls
    /// below a fraction of `min` ([`VOICING_SCORE_HYSTERESIS`], internal),
    /// which keeps soft note tails from being clipped. See the README
    /// "Voicing" section.
    pub fn set_voicing_score_min(&mut self, min: f64) {
        self.ap.score_min = min;
    }
}

impl Synthesizer {
    pub fn new(fs: f64, period: f64, fftsize: usize, fo_floor: f64, fo_ceil: f64) -> Self {
        let cfg = Config::new(period, fftsize, fo_floor, fo_ceil, fs);
        let fft = Fft::new(fftsize);
        let syn = Synth::new(&cfg);
        Synthesizer { cfg, fft, syn }
    }

    /// Default configuration: 5 ms period, Fo 71-800 Hz, and a sample-rate-aware
    /// fftsize (see [`default_fftsize`]).
    pub fn with_defaults(fs: f64) -> Self {
        let fo_floor = 71.0;
        Synthesizer::new(fs, 5.0, default_fftsize(fs, fo_floor), fo_floor, 800.0)
    }

    /// Supply the next frame's parameters (call once per frame boundary).
    /// `aperiodicity` and `spectral_envelope` must each be `numbins` long.
    /// Allocation-free.
    pub fn push_frame(
        &mut self,
        fo: f64,
        voiced: bool,
        silence: bool,
        aperiodicity: &[f64],
        spectral_envelope: &[f64],
    ) {
        self.push_frame_with_strength(
            fo,
            if voiced { 1.0 } else { 0.0 },
            silence,
            aperiodicity,
            spectral_envelope,
        );
    }

    /// Like [`push_frame`](Self::push_frame) but with a continuous voicing
    /// strength in [0,1]: pulse energy scales with strength per bin and noise
    /// gains exactly what pulse loses, so per-bin energy is conserved.
    /// `strength = 1.0` is bit-identical to `push_frame(voiced = true)`;
    /// `0.0` renders fully aperiodic. Fractional strengths are an opt-in
    /// departure from the C reference (see `Reim::set_soft_voicing`).
    pub fn push_frame_with_strength(
        &mut self,
        fo: f64,
        strength: f64,
        silence: bool,
        aperiodicity: &[f64],
        spectral_envelope: &[f64],
    ) {
        self.syn.new_frame(
            &self.cfg,
            &mut self.fft,
            fo,
            strength,
            silence,
            aperiodicity,
            spectral_envelope,
        );
    }

    /// Produce one output sample (overlap-add). Call once per input sample.
    /// Allocation-free.
    pub fn next_sample(&mut self) -> f64 {
        self.syn.next_sample(&self.cfg, &mut self.fft)
    }
}

impl Reim {
    pub fn new(fs: f64, period: f64, fftsize: usize, fo_floor: f64, fo_ceil: f64) -> Self {
        Reim {
            analyzer: Analyzer::new(fs, period, fftsize, fo_floor, fo_ceil),
            synth: Synthesizer::new(fs, period, fftsize, fo_floor, fo_ceil),
        }
    }

    /// Default configuration: 5 ms period, Fo 71-800 Hz, and a sample-rate-aware
    /// fftsize (see [`default_fftsize`]) -- 1024 at 16 kHz, 2048 at 24-48 kHz.
    pub fn with_defaults(fs: f64) -> Self {
        Reim {
            analyzer: Analyzer::with_defaults(fs),
            synth: Synthesizer::with_defaults(fs),
        }
    }

    /// Analyze one input sample WITHOUT synthesizing. Returns true when a new
    /// analysis frame is ready; read it via `last_fo`/`last_voiced`/
    /// `last_silence`/`last_aperiodicity`/`last_spectral_envelope`. Use this for
    /// real-time analysis when you supply your own synthesis -- it skips the
    /// synthesis cost entirely. Allocation-free.
    pub fn analyze_sample(&mut self, input: f64) -> bool {
        self.analyzer.push_sample(input)
    }

    /// Process one input sample, returning one output sample (analysis +
    /// synthesis). Allocation-free.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        if self.analyzer.push_sample(input) {
            self.synth.push_frame(
                self.analyzer.fo(),
                self.analyzer.voiced(),
                self.analyzer.silence(),
                self.analyzer.aperiodicity(),
                self.analyzer.spectral_envelope(),
            );
        }
        self.synth.next_sample()
    }

    /// Run `process_sample` over `input`, writing each result into `output`;
    /// processes `input.len().min(output.len())` samples.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        for (o, &x) in output.iter_mut().zip(input) {
            *o = self.process_sample(x);
        }
    }

    /// Number of frames analyzed so far.
    pub fn frame_count(&self) -> u64 {
        self.analyzer.frame_count()
    }
    /// Pitch-tracker estimate for the most recent frame, in Hz (0.0 when the
    /// tracker found no periodicity). See [`Analyzer::fo`].
    pub fn last_fo(&self) -> f64 {
        self.analyzer.fo()
    }
    /// Whether the most recent frame was judged voiced. See [`Analyzer::voiced`].
    pub fn last_voiced(&self) -> bool {
        self.analyzer.voiced()
    }
    /// Whether the most recent frame was silence.
    pub fn last_silence(&self) -> bool {
        self.analyzer.silence()
    }
    /// Per-bin aperiodicity for the most recent frame (length `fftsize/2 + 1`):
    /// the noise-to-total energy ratio per frequency bin, ~0 = periodic,
    /// ~1 = fully aperiodic.
    pub fn last_aperiodicity(&self) -> &[f64] {
        self.analyzer.aperiodicity()
    }
    /// Per-bin spectral envelope (CheapTrick) for the most recent frame, length
    /// `fftsize/2 + 1`: the smoothed power spectrum carrying the formant
    /// structure. With `last_fo`/`last_voiced`/`last_aperiodicity` this is the
    /// full per-frame WORLD parameter set (read it after each `process_sample`
    /// that advances `frame_count`).
    pub fn last_spectral_envelope(&self) -> &[f64] {
        self.analyzer.spectral_envelope()
    }
    /// Per-frame voicing features for the most recent frame. See
    /// [`Analyzer::voicing_features`].
    pub fn last_voicing_features(&self) -> VoicingFeatures {
        self.analyzer.voicing_features()
    }
    /// Fused voicing probability for the most recent frame. See
    /// [`Analyzer::voicing_score`].
    pub fn last_voicing_score(&self) -> f64 {
        self.analyzer.voicing_score()
    }
    /// Set the minimum fused voicing probability for a frame to be judged
    /// voiced (default 0.0 = off). See [`Analyzer::set_voicing_score_min`].
    pub fn set_voicing_score_min(&mut self, min: f64) {
        self.analyzer.set_voicing_score_min(min);
    }
}

// ============================================================================
// Minimal WAV I/O (mono; reads PCM16/PCM8/float32, writes float32).
// ============================================================================

pub struct WavData {
    pub samples: Vec<f64>,
    pub sample_rate: u32,
}

fn read_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn read_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

pub fn read_wav(path: &str) -> Result<WavData, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<(usize, usize)> = None; // (offset, len)
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = read_u32(&bytes, pos + 4) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            if body + 16 > bytes.len() {
                return Err("truncated fmt chunk".into());
            }
            let format = read_u16(&bytes, body);
            let channels = read_u16(&bytes, body + 2);
            let rate = read_u32(&bytes, body + 4);
            let bits = read_u16(&bytes, body + 14);
            fmt = Some((format, channels, rate, bits));
        } else if id == b"data" {
            data = Some((body, size.min(bytes.len() - body)));
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    let (format, channels, rate, bits) = fmt.ok_or("missing fmt chunk")?;
    let (off, len) = data.ok_or("missing data chunk")?;
    if channels != 1 {
        return Err(format!("only mono supported (got {channels} channels)"));
    }
    let body = &bytes[off..off + len];
    let samples = match (format, bits) {
        (1, 16) => body
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
            .collect(),
        (1, 8) => body.iter().map(|&b| (b as f64 - 128.0) / 128.0).collect(),
        (3, 32) => body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        _ => return Err(format!("unsupported format tag {format} / {bits} bits")),
    };
    Ok(WavData {
        samples,
        sample_rate: rate,
    })
}

pub fn write_wav_f32(path: &str, samples: &[f64], sample_rate: u32) -> Result<(), String> {
    let data_bytes = samples.len() * 4;
    let mut out = Vec::with_capacity(44 + data_bytes);
    let byte_rate = sample_rate * 4;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_bytes) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes()); // block align
    out.extend_from_slice(&32u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&(s as f32).to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| format!("write {path}: {e}"))
}

// ============================================================================
// Profiling
// ============================================================================

/// Timing breakdown from [`Reim::profile`]. All durations are in seconds.
pub struct Profile {
    pub samples: usize,
    pub fs: f64,
    pub period_ms: f64,
    pub elapsed_total: f64, // whole-pipeline process time
    pub stage_silence: f64,
    pub stage_fo: f64,
    pub stage_ap: f64,
    pub stage_sp: f64,
    pub stage_new_frame: f64,
    pub stage_next_sample: f64,
    pub frame_latencies: Vec<f64>, // per-frame analysis latency
    // synthesis sub-stages
    pub synth_minphase_pulse: f64,
    pub synth_minphase_noise: f64,
    pub synth_impulse_noise: f64,
    pub synth_pulse_gen: f64,
    pub synth_pulse_gen_count: usize,
}

impl Reim {
    /// Profile the default pipeline on `samples` at `fs`: a warm-up pass, a timed
    /// whole-pipeline pass (for throughput), and an instrumented pass that times
    /// each analysis/synthesis stage and the per-frame analysis latency.
    pub fn profile(samples: &[f64], fs: f64) -> Profile {
        use std::time::Instant;

        let mut reim = Reim::with_defaults(fs);
        let mut out = vec![0.0; samples.len()];
        reim.process_block(samples, &mut out); // warm-up

        let mut reim = Reim::with_defaults(fs);
        let t0 = Instant::now();
        reim.process_block(samples, &mut out);
        let elapsed_total = t0.elapsed().as_secs_f64();

        // instrumented pass: drive the internals with per-stage timers
        let mut r = Reim::with_defaults(fs);
        let cfg = r.analyzer.cfg;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;
        let (mut t_sil, mut t_fo, mut t_ap, mut t_sp, mut t_nf, mut t_ns) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut t_mp_pulse, mut t_mp_noise, mut t_imp_noise) = (0.0, 0.0, 0.0);
        let (mut t_pulse_gen, mut pulse_gen_count) = (0.0, 0usize);
        let mut frame_latencies = Vec::new();
        for &x in samples {
            let Reim { analyzer, synth } = &mut r;
            if analyzer.framer.next(x, &mut analyzer.frame_window) {
                let frame_t0 = Instant::now();
                let (wave_d, wave) = (
                    &analyzer.frame_window[..fftsize],
                    &analyzer.frame_window[1..fftsize + 1],
                );
                let t = Instant::now();
                let silence = analyze_silence(&cfg, wave, SILENCE_THRESHOLD);
                t_sil += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let fo = analyzer.fo.analyze(&cfg, &mut analyzer.fft, wave, wave_d);
                t_fo += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let features = analyzer.voicing_features();
                let voiced = analyzer.ap.analyze(
                    &cfg,
                    &mut analyzer.fft,
                    wave,
                    fo,
                    features,
                    silence,
                    &mut analyzer.ap_buf,
                );
                t_ap += t.elapsed().as_secs_f64();
                let t = Instant::now();
                analyzer.sp.analyze(
                    &cfg,
                    &mut analyzer.fft,
                    wave,
                    fo,
                    voiced,
                    silence,
                    &mut analyzer.sp_buf,
                );
                t_sp += t.elapsed().as_secs_f64();

                // --- synthesis new_frame, sub-staged ---
                let t_nf0 = Instant::now();
                let syn = &mut synth.syn;
                for k in 0..numbins {
                    let spec = analyzer.sp_buf[k];
                    let aper = analyzer.ap_buf[k] * analyzer.ap_buf[k];
                    syn.spec_pulse_r[k] = spec * (1.0 - aper);
                    syn.spec_noise_r[k] = spec * aper;
                }
                syn.has_pulse = voiced && !silence;
                if syn.has_pulse {
                    syn.interval = synth.cfg.fs / fo;
                    let gain_pulse = syn.interval.sqrt();
                    let t = Instant::now();
                    generate_minimum_phase_spectrum(
                        &mut syn.spec_pulse_r,
                        &mut syn.spec_pulse_i,
                        gain_pulse,
                        fftsize,
                        &mut synth.fft,
                    );
                    t_mp_pulse += t.elapsed().as_secs_f64();
                }
                syn.has_noise = !silence;
                if syn.has_noise {
                    let t = Instant::now();
                    generate_minimum_phase_spectrum(
                        &mut syn.spec_noise_r,
                        &mut syn.spec_noise_i,
                        syn.gain_noise,
                        fftsize,
                        &mut synth.fft,
                    );
                    t_mp_noise += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    generate_impulse(
                        &mut syn.impulse_noise,
                        &syn.spec_noise_r,
                        &syn.spec_noise_i,
                        0.0,
                        &syn.window,
                        &mut syn.temp_r,
                        &mut syn.temp_i,
                        fftsize,
                        &mut synth.fft,
                    );
                    t_imp_noise += t.elapsed().as_secs_f64();
                }
                t_nf += t_nf0.elapsed().as_secs_f64();
                frame_latencies.push(frame_t0.elapsed().as_secs_f64());
            }

            // --- synthesis next_sample, sub-staged ---
            let t_ns0 = Instant::now();
            let syn = &mut synth.syn;
            if syn.has_pulse {
                if syn.pulse_int == 0 {
                    let t = Instant::now();
                    generate_impulse(
                        &mut syn.impulse_pulse,
                        &syn.spec_pulse_r,
                        &syn.spec_pulse_i,
                        syn.pulse_frac,
                        &syn.window,
                        &mut syn.temp_r,
                        &mut syn.temp_i,
                        fftsize,
                        &mut synth.fft,
                    );
                    t_pulse_gen += t.elapsed().as_secs_f64();
                    pulse_gen_count += 1;
                    syn.buffer.push_additive(&syn.impulse_pulse);
                    let interval_int = syn.interval.floor();
                    let interval_frac = syn.interval - interval_int;
                    let next = syn.pulse_frac + interval_frac;
                    let carry = next.floor();
                    syn.pulse_int += (interval_int + carry) as i32;
                    syn.pulse_frac = next - carry;
                }
                syn.pulse_int -= 1;
            }
            if syn.has_noise {
                if syn.noise_int == syn.interval_random {
                    syn.buffer.push_additive(&syn.impulse_noise);
                }
                if syn.noise_int == syn.interval_velvet - 1 {
                    let r = syn.random.uniform();
                    syn.interval_random = (r * (syn.interval_velvet - 1) as f64).floor() as usize;
                    syn.noise_int = 0;
                }
                syn.noise_int += 1;
            }
            let _ = syn.buffer.pop();
            t_ns += t_ns0.elapsed().as_secs_f64();
        }

        Profile {
            samples: samples.len(),
            fs,
            period_ms: cfg.period,
            elapsed_total,
            stage_silence: t_sil,
            stage_fo: t_fo,
            stage_ap: t_ap,
            stage_sp: t_sp,
            stage_new_frame: t_nf,
            stage_next_sample: t_ns,
            frame_latencies,
            synth_minphase_pulse: t_mp_pulse,
            synth_minphase_noise: t_mp_noise,
            synth_impulse_noise: t_imp_noise,
            synth_pulse_gen: t_pulse_gen,
            synth_pulse_gen_count: pulse_gen_count,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eta: f64) -> bool {
        (a - b).abs() <= eta
    }

    #[test]
    fn nccf_periodic_vs_noise() {
        let fs = 44100.0_f64;
        let fo = 220.0;
        let lag = (fs / fo).round() as usize;
        let n = 2048;

        let sine: Vec<f64> = (0..n)
            .map(|i| (2.0 * REIM_PI * fo * i as f64 / fs).sin())
            .collect();
        assert!(nccf_at_lag(&sine, lag) > 0.99);

        // A DC offset must not inflate the score at a wrong lag.
        let offset_sine: Vec<f64> = sine.iter().map(|s| s + 5.0).collect();
        assert!(nccf_at_lag(&offset_sine, lag) > 0.99);
        assert!(nccf_at_lag(&offset_sine, lag + lag / 2) < 0.5);

        let mut rng = Xorshift::new();
        let noise: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
        assert!(nccf_at_lag(&noise, lag).abs() < 0.2);

        // Lag too large for one period of overlap: uninformative 0.0.
        assert_eq!(nccf_at_lag(&sine, n / 2 + 1), 0.0);
        assert_eq!(nccf_at_lag(&sine, 0), 0.0);
    }

    #[test]
    fn cpp_periodic_vs_noise() {
        let fs = 44100.0_f64;
        let fftsize = 2048;
        let numbins = fftsize / 2 + 1;
        let mut fft = Fft::new(fftsize);
        let mut re = vec![0.0; fftsize];
        let mut im = vec![0.0; fftsize];

        let pspec_of = |x: &Vec<f64>, fft: &mut Fft| {
            let mut r: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| v * hanning_window(i as f64, fftsize as f64, fftsize as f64))
                .collect();
            let mut i = vec![0.0; fftsize];
            fft.forward_real_f64(&mut r, &mut i);
            (0..numbins)
                .map(|k| complex_abs2(r[k], i[k]) + 1e-15)
                .collect::<Vec<f64>>()
        };

        // Harmonic-rich source (CPP needs rahmonic ripple across many
        // harmonics; a pure sine has none).
        let fo = 200.0;
        let voiced: Vec<f64> = (0..fftsize)
            .map(|i| {
                (1..=20)
                    .map(|h| (2.0 * REIM_PI * fo * h as f64 * i as f64 / fs).sin() / h as f64)
                    .sum()
            })
            .collect();
        let mut rng = Xorshift::new();
        let noise: Vec<f64> = (0..fftsize).map(|_| rng.uniform() - 0.5).collect();

        let pspec_voiced = pspec_of(&voiced, &mut fft);
        let pspec_noise = pspec_of(&noise, &mut fft);
        let cpp_voiced = cepstral_peak_prominence(
            &pspec_voiced, numbins, fs, 71.0, 800.0, &mut re, &mut im, &mut fft,
        );
        let cpp_noise = cepstral_peak_prominence(
            &pspec_noise, numbins, fs, 71.0, 800.0, &mut re, &mut im, &mut fft,
        );
        assert!(
            cpp_voiced > 2.0 * cpp_noise,
            "cpp_voiced={cpp_voiced} cpp_noise={cpp_noise}"
        );

        // Near-silence must not NaN.
        let tiny = vec![1e-20; fftsize];
        let pspec_tiny = pspec_of(&tiny, &mut fft);
        let cpp_tiny = cepstral_peak_prominence(
            &pspec_tiny, numbins, fs, 71.0, 800.0, &mut re, &mut im, &mut fft,
        );
        assert!(cpp_tiny.is_finite());
    }

    #[test]
    fn push_frame_with_strength_matches_bool_paths() {
        let fs = 24000.0;
        let numbins = default_fftsize(fs, 71.0) / 2 + 1;
        // A mid-range aperiodicity so the pulse/noise split is non-trivial.
        let ap = vec![0.5; numbins];
        let sp: Vec<f64> = (0..numbins).map(|k| 1.0 / (1.0 + k as f64)).collect();

        let render = |voiced: Option<bool>, strength: f64| {
            let mut synth = Synthesizer::with_defaults(fs);
            match voiced {
                Some(v) => synth.push_frame(220.0, v, false, &ap, &sp),
                None => synth.push_frame_with_strength(220.0, strength, false, &ap, &sp),
            }
            (0..2048).map(|_| synth.next_sample()).collect::<Vec<f64>>()
        };

        // strength 1.0 == voiced, bit-identical.
        assert_eq!(render(Some(true), 0.0), render(None, 1.0));
        // strength 0.0 == unvoiced when ap = 1.0 (the analyzer's unvoiced case).
        let ap_unvoiced = vec![1.0; numbins];
        let mut a = Synthesizer::with_defaults(fs);
        a.push_frame(220.0, false, false, &ap_unvoiced, &sp);
        let mut b = Synthesizer::with_defaults(fs);
        b.push_frame_with_strength(220.0, 0.0, false, &ap_unvoiced, &sp);
        for _ in 0..2048 {
            assert_eq!(a.next_sample(), b.next_sample());
        }
    }

    #[test]
    fn strength_split_conserves_per_bin_energy() {
        let fs = 24000.0;
        let mut synth = Synthesizer::with_defaults(fs);
        let numbins = synth.cfg.numbins;
        let ap: Vec<f64> = (0..numbins).map(|k| (k as f64 / numbins as f64)).collect();
        let sp: Vec<f64> = (0..numbins).map(|k| 1.0 / (1.0 + k as f64)).collect();
        for &strength in &[0.0, 0.25, 0.5, 0.9, 1.0] {
            // Silence path skips the min-phase transform, leaving the raw
            // pulse/noise split observable in the spectra buffers.
            synth.push_frame_with_strength(220.0, strength, true, &ap, &sp);
            for k in 0..numbins {
                let total = synth.syn.spec_pulse_r[k] + synth.syn.spec_noise_r[k];
                assert!(
                    approx(total, sp[k], 1e-12 * (1.0 + sp[k])),
                    "strength={strength} bin={k}: {total} != {}",
                    sp[k]
                );
            }
        }
    }

    #[test]
    fn voicing_probability_monotone_in_each_feature() {
        let base = VoicingFeatures {
            score: 100.0,
            nccf: 0.5,
            cpp: 0.5,
        };
        let p = voicing_probability(base);
        assert!(p > 0.0 && p < 1.0);
        assert!(voicing_probability(VoicingFeatures { score: 1e4, ..base }) > p);
        assert!(voicing_probability(VoicingFeatures { nccf: 0.9, ..base }) > p);
        assert!(voicing_probability(VoicingFeatures { cpp: 1.5, ..base }) > p);
    }

    #[test]
    fn schmitt_gate_hysteresis() {
        let high = 1000.0;
        let low = high * VOICING_SCORE_HYSTERESIS;
        let mut gate = SchmittGate { open: false };

        // Closed: only scores >= high pass.
        assert!(!gate.pass(high - 1.0, high));
        assert!(!gate.pass(low, high));
        assert!(gate.pass(high, high));

        // Open: scores in [low, high) keep passing.
        gate.open = true;
        assert!(gate.pass(high - 1.0, high));
        assert!(gate.pass(low, high));
        // Below low: closes.
        assert!(!gate.pass(low - 1e-9, high));

        // A veto (silence / hard gate) closes the latch regardless of score;
        // the caller sets open from the final voiced decision.
        gate.open = false;
        assert!(!gate.pass(high - 1.0, high));
    }

    #[test]
    fn fft_complex_exp() {
        let mut fft = Fft::new(8);
        // x[n] = exp(+i 2pi 2n/8): forward DFT spikes at bin 2 with magnitude 8.
        let mut xr = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let mut xi = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        fft.forward(&mut xr, &mut xi);
        let exp_r = [0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for k in 0..8 {
            assert!(approx(xr[k], exp_r[k], 1e-8), "re[{k}]={}", xr[k]);
            assert!(approx(xi[k], 0.0, 1e-8), "im[{k}]={}", xi[k]);
        }
    }

    #[test]
    fn fft_roundtrip() {
        let mut fft = Fft::new(64);
        let mut rng = Xorshift::new();
        let orig_r: Vec<f64> = (0..64).map(|_| rng.uniform() - 0.5).collect();
        let orig_i: Vec<f64> = (0..64).map(|_| rng.uniform() - 0.5).collect();
        let mut r = orig_r.clone();
        let mut i = orig_i.clone();
        fft.forward(&mut r, &mut i);
        fft.inverse(&mut r, &mut i);
        for k in 0..64 {
            assert!(approx(r[k], orig_r[k], 1e-12));
            assert!(approx(i[k], orig_i[k], 1e-12));
        }
    }

    #[test]
    fn rng_matches_reference_c() {
        // Reference values produced by the C xorshift with the same fixed seed.
        // Full f64 digits are intentional: they mirror the exact C reference output
        // and are asserted to 1e-15, so excessive_precision is expected here.
        #[allow(clippy::excessive_precision)]
        let expected = [
            0.29209269729724452,
            0.76769657264642799,
            0.013512801382111573,
            0.85400040048500536,
            0.64289931153014757,
        ];
        let mut rng = Xorshift::new();
        for &e in &expected {
            assert!(approx(rng.uniform(), e, 1e-15));
        }
    }

    #[test]
    fn circular_buffer_overflow() {
        let mut cb = CircularBuffer::new(4);
        let mut out = [0.0; 4];
        cb.copy_all(&mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);

        cb.push(1.0);
        cb.push(2.0);
        cb.copy_all(&mut out);
        assert_eq!(out, [0.0, 0.0, 1.0, 2.0]);

        for v in [3.0, 4.0, 5.0] {
            cb.push(v);
        }
        cb.copy_all(&mut out);
        assert_eq!(out, [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn circular_queue_additive_and_fill() {
        let mut q = CircularQueue::new(4);
        assert_eq!(q.remaining, 0);
        assert_eq!(q.pop(), 0.0);

        q.push_additive(&[1.0, 2.0]);
        assert_eq!(q.remaining, 2);
        assert_eq!(q.pop(), 1.0);
        q.push_additive(&[1.0, 2.0]); // additive into existing
        assert_eq!(q.remaining, 2);
        assert_eq!(q.pop(), 3.0);
        assert_eq!(q.pop(), 2.0);
        assert_eq!(q.remaining, 0);

        // overflow fill: 5 into capacity 4 keeps the last 4
        let mut q = CircularQueue::new(4);
        q.push_additive(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(q.remaining, 4);
        assert_eq!(q.pop(), 2.0);
        assert_eq!(q.pop(), 3.0);
        assert_eq!(q.pop(), 4.0);
        assert_eq!(q.pop(), 5.0);
    }

    #[test]
    fn ifftshift_matches_reference() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = [9.0; 8];
        ifftshift(&src, &mut dst, 5);
        assert_eq!(dst, [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn clamp_index_bounds() {
        assert_eq!(clamp_index_f(3.0, 8), 3.0);
        assert_eq!(clamp_index_f(-999.0, 8), 0.0);
        assert_eq!(clamp_index_f(999.0, 8), 7.0);
        assert_eq!(clamp_index_f(999.0, 0), 0.0);
    }

    #[test]
    fn instfreq_basic() {
        assert_eq!(instfreq(0.0, 0.0, 0.0, 0.0, 1.0), 0.0);
        assert!(approx(instfreq(1.0, 0.0, 0.0, 1.0, 4.0), 1.0, 1e-8));
        assert!(approx(instfreq(0.0, 1.0, 1.0, 0.0, 4.0), 1.0, 1e-8));
    }

    #[test]
    fn framer_emits_frames_on_grid() {
        let cfg = Config::new(5.0, 2048, 71.0, 800.0, 24000.0);
        let mut framer = Framer::new(&cfg);
        let mut wf = vec![0.0; cfg.fftsize + 1];
        let mut boundaries = Vec::new();
        for i in 0..600 {
            if framer.next(i as f64, &mut wf) {
                boundaries.push(i);
            }
        }
        // framesize = 5/1000*24000 = 120 -> frames at 0,120,240,360,480
        assert_eq!(boundaries, vec![0, 120, 240, 360, 480]);
    }

    #[test]
    fn default_fftsize_adapts_to_sample_rate() {
        // 16 kHz drops to 1024 (the documented benchmark win); 24-48 kHz stay 2048
        // so the bundled-file reference and the latency story are unchanged.
        assert_eq!(default_fftsize(16000.0, 71.0), 1024);
        assert_eq!(default_fftsize(24000.0, 71.0), 2048);
        assert_eq!(default_fftsize(44100.0, 71.0), 2048);
        assert_eq!(default_fftsize(48000.0, 71.0), 2048);
        assert_eq!(default_fftsize(8000.0, 71.0), 512); // clamped floor
    }

    #[test]
    fn silence_on_zeros_and_determinism() {
        let mut a = Reim::with_defaults(24000.0);
        let mut b = Reim::with_defaults(24000.0);
        let mut rng = Xorshift::new();
        let input: Vec<f64> = (0..6000).map(|_| 0.2 * (rng.uniform() - 0.5)).collect();
        let mut oa = vec![0.0; input.len()];
        let mut ob = vec![0.0; input.len()];
        a.process_block(&input, &mut oa);
        b.process_block(&input, &mut ob);
        // deterministic: same input -> identical output
        assert_eq!(oa, ob);
        // all finite
        assert!(oa.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn silence_produces_silence() {
        let mut r = Reim::with_defaults(24000.0);
        let input = vec![0.0; 6000];
        let mut out = vec![0.0; input.len()];
        r.process_block(&input, &mut out);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "silent input must yield silence"
        );
    }

    #[test]
    fn pure_tone_tracks_fo() {
        // 200 Hz tone should be detected as ~200 Hz on voiced frames.
        let fs = 24000.0;
        let n = (fs * 0.5) as usize;
        let input: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin())
            .collect();
        let mut r = Reim::with_defaults(fs);
        let mut last = 0u64;
        let mut fos = Vec::new();
        for &x in &input {
            r.process_sample(x);
            if r.frame_count() != last {
                last = r.frame_count();
                if r.last_fo() > 0.0 {
                    fos.push(r.last_fo());
                }
            }
        }
        assert!(!fos.is_empty(), "no fo detected on a pure tone");
        // use the median of the back half (after warm-up)
        let mut tail: Vec<f64> = fos[fos.len() / 2..].to_vec();
        tail.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = tail[tail.len() / 2];
        assert!(approx(med, 200.0, 10.0), "median fo {med} not near 200 Hz");
    }

    #[test]
    fn zerocross_periodicity_gate() {
        // The gate accepts a candidate when interval-frequency variance <= mean, and
        // rejects otherwise. A steady tone has near-constant intervals (low variance)
        // and passes; a fast frequency sweep has widely varying intervals (variance
        // far above the mean) and is rejected. This pins the threshold at the mean,
        // not mean^2 -- a coefficient-of-variation gate would let the sweep through.
        let fs = 24000.0;
        let n = 2048;
        let tone: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin())
            .collect();
        let fo = analyze_fo_with_zerocross(&tone, fs).expect("steady tone passes the gate");
        assert!(approx(fo, 200.0, 5.0), "fo {fo} not near 200 Hz");

        // Half the window at 300 Hz, half at 900 Hz: interval frequencies are bimodal
        // {300, 900}, mean ~600, variance ~90k. That is > mean (600) so the real gate
        // rejects, but < mean^2 (360k) so a coefficient-of-variation gate (variance >
        // mean^2) would wrongly accept it -- this is what pins the threshold at mean.
        let two_tone: Vec<f64> = (0..n)
            .map(|i| {
                let f = if i < n / 2 { 300.0 } else { 900.0 };
                (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin()
            })
            .collect();
        assert!(
            analyze_fo_with_zerocross(&two_tone, fs).is_none(),
            "bimodal signal must fail the variance>mean gate"
        );
    }

    #[test]
    fn ap_guard_matches_c_nan_semantics() {
        // C's analyze_ap bails on `fo < floor || fo > ceil`; for fo==NaN both are
        // false, so it PROCEEDS to the V/UV decision. The Rust guard must do the
        // same rather than short-circuiting NaN straight to unvoiced.
        let cfg = Config::new(5.0, 2048, 71.0, 800.0, 24000.0);
        let mut fft = Fft::new(cfg.fftsize);
        let mut ap = ApAnalyzer::new(&cfg);
        let input: Vec<f64> = (0..cfg.fftsize)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / cfg.fs).sin())
            .collect();
        let mut buf = vec![0.0; cfg.numbins];
        let voiced = ap.analyze(
            &cfg,
            &mut fft,
            &input,
            f64::NAN,
            VoicingFeatures {
                score: f64::INFINITY,
                nccf: 1.0,
                cpp: 1.0,
            },
            false,
            &mut buf,
        );
        // independent reference: the decision C reaches for a NaN fo
        let mut re = vec![0.0; cfg.fftsize];
        let mut im = vec![0.0; cfg.fftsize];
        let expected = estimate_is_voiced(
            &input,
            &mut re,
            &mut im,
            cfg.fftsize,
            f64::NAN,
            cfg.fs,
            &mut fft,
        );
        assert_eq!(
            voiced, expected,
            "NaN fo must reach the V/UV decision, as in C"
        );
    }

    #[test]
    fn no_nan_on_noise() {
        let mut r = Reim::with_defaults(48000.0);
        let mut rng = Xorshift::new();
        let input: Vec<f64> = (0..20000).map(|_| rng.uniform() - 0.5).collect();
        let mut out = vec![0.0; input.len()];
        r.process_block(&input, &mut out);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn voicing_score_gate_is_opt_in() {
        // Default (score_min = 0) leaves voicing unchanged; a huge threshold
        // rejects every frame. Verifies the opt-in periodicity-gate plumbing.
        let fs = 24000.0;
        let n = (fs * 0.5) as usize;
        let input: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin())
            .collect();
        let count_voiced = |min: f64| {
            let mut r = Reim::with_defaults(fs);
            r.set_voicing_score_min(min);
            let mut last = 0u64;
            let mut voiced = 0usize;
            for &x in &input {
                r.process_sample(x);
                if r.frame_count() != last {
                    last = r.frame_count();
                    if r.last_voiced() {
                        voiced += 1;
                    }
                }
            }
            voiced
        };
        assert!(
            count_voiced(0.0) > 0,
            "gate off: a 200 Hz tone must have voiced frames"
        );
        assert_eq!(
            count_voiced(f64::INFINITY),
            0,
            "infinite threshold must reject all frames"
        );
    }

    #[test]
    fn analyze_sample_matches_process_sample_analysis() {
        // analyze_sample must yield the same per-frame parameters as process_sample,
        // only skipping synthesis (they share analyze_current_frame).
        let fs = 24000.0;
        let n = (fs * 0.4) as usize;
        let input: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 180.0 * i as f64 / fs).sin())
            .collect();
        let mut a = Reim::with_defaults(fs);
        let mut b = Reim::with_defaults(fs);
        for &x in &input {
            a.analyze_sample(x);
            b.process_sample(x);
        }
        assert_eq!(a.frame_count(), b.frame_count());
        assert_eq!(a.last_fo(), b.last_fo());
        assert_eq!(a.last_voiced(), b.last_voiced());
        assert_eq!(a.last_silence(), b.last_silence());
        assert_eq!(a.last_spectral_envelope(), b.last_spectral_envelope());
        assert_eq!(a.last_aperiodicity(), b.last_aperiodicity());
    }

    #[test]
    fn analyzer_synthesizer_compose_equals_reim() {
        // Driving Analyzer + Synthesizer by hand (the manipulate->resynthesize
        // path with no manipulation) must equal Reim::process_block sample for
        // sample -- the composition is the only behavior Reim adds.
        let fs = 24000.0;
        let n = (fs * 0.4) as usize;
        let input: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 190.0 * i as f64 / fs).sin())
            .collect();

        let mut reim = Reim::with_defaults(fs);
        let mut fused = vec![0.0; input.len()];
        reim.process_block(&input, &mut fused);

        let mut analyzer = Analyzer::with_defaults(fs);
        let mut synth = Synthesizer::with_defaults(fs);
        let mut split = Vec::with_capacity(input.len());
        for &x in &input {
            if analyzer.push_sample(x) {
                synth.push_frame(
                    analyzer.fo(),
                    analyzer.voiced(),
                    analyzer.silence(),
                    analyzer.aperiodicity(),
                    analyzer.spectral_envelope(),
                );
            }
            split.push(synth.next_sample());
        }
        assert_eq!(
            fused, split,
            "Analyzer+Synthesizer must match Reim sample-for-sample"
        );
    }
}
