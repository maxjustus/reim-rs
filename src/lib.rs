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
    clippy::approx_constant,
    clippy::manual_range_contains
)]

// The reference C code uses a *truncated* pi literal. Reproduce it exactly so the
// windowing/phase math matches the C bit-for-bit where the FFT allows.
const REIM_PI: f64 = 3.14159265358979;
const U32_MAX_F: f64 = u32::MAX as f64;

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
    let hi = if len == 0 { 0.0 } else { (len - 1) as f64 };
    x.max(0.0).min(hi)
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
// FFT — iterative radix-2 Cooley-Tukey (decimation-in-time), planar in-place.
// Forward uses the exp(-i) kernel; inverse uses exp(+i) and scales by 1/N,
// matching the reference's `execute_fft` / `execute_ifft` conventions.
// ============================================================================

struct Fft {
    n: usize,
    rev: Vec<usize>,  // bit-reversal permutation
    cos: Vec<f64>,    // cos(2*pi*t/n), t in 0..n/2
    sin: Vec<f64>,    // sin(2*pi*t/n), t in 0..n/2
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2, "fft size must be a power of two");
        let bits = n.trailing_zeros();
        let rev = (0..n).map(|i| (i as u32).reverse_bits() >> (32 - bits)).map(|x| x as usize).collect();
        let half = n / 2;
        let mut cos = vec![0.0; half];
        let mut sin = vec![0.0; half];
        for t in 0..half {
            let ang = 2.0 * std::f64::consts::PI * t as f64 / n as f64;
            cos[t] = ang.cos();
            sin[t] = ang.sin();
        }
        Fft { n, rev, cos, sin }
    }

    #[inline]
    fn transform(&self, re: &mut [f64], im: &mut [f64], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(re.len(), n);
        debug_assert_eq!(im.len(), n);
        // bit-reversal reordering
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        // butterflies
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut start = 0;
            while start < n {
                let mut tdx = 0;
                for k in 0..half {
                    let wr = self.cos[tdx];
                    let wi = if inverse { self.sin[tdx] } else { -self.sin[tdx] };
                    let a = start + k;
                    let b = a + half;
                    let xr = re[b] * wr - im[b] * wi;
                    let xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                    tdx += step;
                }
                start += len;
            }
            len <<= 1;
        }
    }

    #[inline]
    fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        self.transform(re, im, false);
    }

    #[inline]
    fn inverse(&self, re: &mut [f64], im: &mut [f64]) {
        self.transform(re, im, true);
        let scale = 1.0 / self.n as f64;
        for k in 0..self.n {
            re[k] *= scale;
            im[k] *= scale;
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
        CircularBuffer { head: 0, buf: vec![0.0; capacity] }
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
        CircularQueue { head: 0, remaining: 0, buf: vec![0.0; capacity] }
    }
    /// Add `samples` into the queue starting at the current head (overlap-add).
    fn push_additive(&mut self, samples: &[f64]) {
        let cap = self.buf.len();
        let size = samples.len();
        let mut index = self.head;
        for (i, &value) in samples.iter().enumerate() {
            if i >= cap {
                self.buf[index] = value; // overwrite on overflow
                self.head = wrap_next(self.head, cap);
            } else {
                self.buf[index] += value;
            }
            index = wrap_next(index, cap);
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
        Config { period, fs, fo_floor, fo_ceil, fftsize, numbins: fftsize / 2 + 1 }
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
    let mut frame_sum_sqr = 0.0;
    for i in 1..fftsize {
        let x = input[i];
        frame_sum_sqr += x * x;
    }
    let frame_rms2 = frame_sum_sqr / fftsize as f64;
    frame_rms2 < threshold * threshold
}

// ============================================================================
// Fo analysis — DIO (zero-crossing) candidates refined by instantaneous
// frequency, scored by summation of residual harmonics (SRH).
// ============================================================================

struct FoAnalyzer {
    num_candidates: usize,
    channel_filters: Vec<f64>,  // [ch*fftsize + k]
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
    fo_previous: f64,
}

fn get_interpolated_spectrum(freq: f64, fs: f64, spec: &[f64], numbins: usize) -> f64 {
    let position = clamp_index_f(freq / (fs / 2.0) * (numbins - 1) as f64, numbins - 1);
    let index = position.floor() as usize;
    let delta = position - index as f64;
    (1.0 - delta) * spec[index] + delta * spec[index + 1]
}

fn refine_fo(fo: f64, fo_floor: f64, fo_ceil: f64, fs: f64, ifreqf: &[f64], pspec: &[f64], numbins: usize) -> f64 {
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

/// Estimate Fo from zero-crossings / peaks / dips of a band-limited waveform.
/// Returns (fo, relative-std-deviation) or None when no intervals were found.
fn analyze_fo_with_zerocross(x: &[f64], fs: f64) -> Option<(f64, f64)> {
    let length = x.len();
    let mut last_positive: i64 = -1;
    let mut last_negative: i64 = -1;
    let mut last_peak: i64 = -1;
    let mut last_dip: i64 = -1;

    let mut denominator = 0.0;
    let mut sum_freq = 0.0;
    let mut sum_square_freq = 0.0;

    let accumulate = |i: i64, last: i64, num: &mut f64, sf: &mut f64, ssf: &mut f64| {
        let interval = (i - last) as f64;
        let freq = fs / interval;
        *num += interval;
        *sf += freq * interval;
        *ssf += freq * freq * interval;
    };

    let mut xprev = x[0];
    let mut xdiffprev = x[1] - x[0];
    for i in 1..length - 1 {
        let xcurr = x[i];
        let xdiffcurr = x[i + 1] - x[i];
        let ii = i as i64;
        if xprev < 0.0 && xcurr >= 0.0 {
            if last_positive >= 0 {
                accumulate(ii, last_positive, &mut denominator, &mut sum_freq, &mut sum_square_freq);
            }
            last_positive = ii;
        } else if xprev > 0.0 && xcurr <= 0.0 {
            if last_negative >= 0 {
                accumulate(ii, last_negative, &mut denominator, &mut sum_freq, &mut sum_square_freq);
            }
            last_negative = ii;
        }
        if xdiffprev < 0.0 && xdiffcurr >= 0.0 {
            if last_peak >= 0 {
                accumulate(ii, last_peak, &mut denominator, &mut sum_freq, &mut sum_square_freq);
            }
            last_peak = ii;
        } else if xdiffprev > 0.0 && xdiffcurr <= 0.0 {
            if last_dip > 1 {
                // NOTE: asymmetric guard (>1, not >=0) preserved from the C source.
                accumulate(ii, last_dip, &mut denominator, &mut sum_freq, &mut sum_square_freq);
            }
            last_dip = ii;
        }
        xprev = xcurr;
        xdiffprev = xdiffcurr;
    }

    if denominator <= 0.0 {
        return None;
    }
    let mean_freq = sum_freq / denominator;
    if mean_freq <= 0.0 || mean_freq > fs / 2.0 {
        return None;
    }
    // NOTE: faithful to a variance-formula bug in the C reference -- it subtracts
    // E[f] (mean_freq), not E[f]^2. Do NOT "fix" it: it is load-bearing. The buggy
    // form makes the `rsd > 1.0` reject gate fire when interval-frequency variance
    // exceeds the mean frequency (an effective periodicity filter); the correct form
    // would only reject variance > mean^2 (~never -> gate inert, noise/rumble passes).
    // Correcting it breaks the C-oracle match (151 -> 7 dB seg-SNR) and worsens rumble
    // rejection, with no measured benefit, so it is reproduced verbatim.
    let std_freq = (sum_square_freq / denominator - mean_freq).sqrt();
    Some((mean_freq, std_freq / mean_freq))
}

impl FoAnalyzer {
    fn new(cfg: &Config, fft: &Fft) -> Self {
        let fs = cfg.fs;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        let channels_per_octave = 2.0;
        let num_candidates = ((cfg.fo_ceil / cfg.fo_floor).log2() * channels_per_octave).ceil() as usize;

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
        let window = (0..fftsize).map(|k| nuttall_window(k as f64, fftsize as f64, window_length)).collect();

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
            fo_previous: 0.0,
        }
    }

    fn analyze(&mut self, cfg: &Config, fft: &Fft, input: &[f64], input_delayed: &[f64]) -> f64 {
        let fs = cfg.fs;
        let fo_floor = cfg.fo_floor;
        let fo_ceil = cfg.fo_ceil;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        // windowed spectra of the frame and its one-sample-delayed copy
        for k in 0..fftsize {
            self.spec_r[k] = input[k] * self.window[k];
            self.spec_i[k] = 0.0;
            self.specd_r[k] = input_delayed[k] * self.window[k];
            self.specd_i[k] = 0.0;
        }
        fft.forward(&mut self.spec_r, &mut self.spec_i);
        fft.forward(&mut self.specd_r, &mut self.specd_i);

        for k in 0..numbins {
            self.pspec[k] = complex_abs2(self.spec_r[k], self.spec_i[k]) + 1e-15;
            self.ifreqf[k] = instfreq(self.spec_r[k], self.spec_i[k], self.specd_r[k], self.specd_i[k], fs);
        }

        // spectrum of the DC-removed frame (for band-pass filtering)
        let mut mean_input = 0.0;
        for k in 0..fftsize {
            mean_input += input[k];
        }
        mean_input /= fftsize as f64;
        for k in 0..fftsize {
            self.spec_filt_r[k] = input[k] - mean_input;
            self.spec_filt_i[k] = 0.0;
        }
        fft.forward(&mut self.spec_filt_r, &mut self.spec_filt_i);

        // initial estimate: previous fo
        let mut best_fo = -1.0;
        let mut best_score = -1.0;
        if self.fo_previous > fo_floor {
            best_fo = refine_fo(self.fo_previous, fo_floor, fo_ceil, fs, &self.ifreqf, &self.pspec, numbins);
            best_score = get_harmonic_score(best_fo, fs, &self.pspec, numbins);
        }

        // DIO over each channel filter
        for ch in 0..self.num_candidates {
            let base = ch * fftsize;
            for k in 0..fftsize {
                let filter = self.channel_filters[base + k];
                self.filtered_r[k] = self.spec_filt_r[k] * filter;
                self.filtered_i[k] = self.spec_filt_i[k] * filter;
            }
            fft.inverse(&mut self.filtered_r, &mut self.filtered_i);

            let offset = self.channel_offsets[ch];
            let (fo, rsd) = match analyze_fo_with_zerocross(&self.filtered_r[offset..], fs) {
                Some(v) => v,
                None => continue,
            };
            if fo.is_nan() || fo < fo_floor || fo > fo_ceil || rsd > 1.0 {
                continue;
            }

            let fo_refined = refine_fo(fo, fo_floor, fo_ceil, fs, &self.ifreqf, &self.pspec, numbins);
            let score = get_harmonic_score(fo_refined, fs, &self.pspec, numbins);
            if best_score < score {
                best_fo = fo_refined;
                best_score = score;
            }
        }

        if best_fo < fo_floor || best_fo > fo_ceil || best_score < 0.0 {
            return 0.0;
        }
        self.fo_previous = best_fo;
        best_fo
    }
}

// ============================================================================
// Aperiodicity analysis (placeholder, as in the reference: V/UV decision only).
// ============================================================================

struct ApAnalyzer {
    x_real: Vec<f64>,
    x_imag: Vec<f64>,
}

// Reject a frame as unvoiced when its energy is concentrated BELOW the detected
// fundamental (rumble / mains hum) -- the gap the HF LoveTrain leaves open. Uses a
// full-frame Hann window for low-frequency resolution. Returns true = sub-fo energy
// dominates the fundamental+harmonic band, so it is not a real voice. Tunable.
const LOWBAND_REJECT_RATIO: f64 = 0.4;

fn low_band_dominated(input: &[f64], re: &mut [f64], im: &mut [f64], fftsize: usize, fo: f64, fs: f64, fft: &Fft) -> bool {
    for i in 0..fftsize {
        re[i] = input[i] * hanning_window(i as f64, fftsize as f64, fftsize as f64);
        im[i] = 0.0;
    }
    fft.forward(re, im);
    let bin = |hz: f64| ((hz / fs * fftsize as f64) as usize).min(fftsize / 2);
    let (lo, mid) = (bin(20.0), bin(0.8 * fo));
    let hi = bin((6.0 * fo).min(fs / 2.0 - 1.0));
    let sub: f64 = (lo..mid).map(|k| complex_abs2(re[k], im[k])).sum();
    let voice: f64 = (mid..=hi).map(|k| complex_abs2(re[k], im[k])).sum();
    sub > LOWBAND_REJECT_RATIO * (sub + voice + 1e-12)
}

/// D4C "LoveTrain"-style voiced/unvoiced decision based on low/high band energy.
fn estimate_is_voiced(input: &[f64], re: &mut [f64], im: &mut [f64], fftsize: usize, fo: f64, fs: f64, fft: &Fft) -> bool {
    if fs < 16000.0 {
        return true;
    }
    let window_length = (1.5 * fs / fo).min(fftsize as f64);
    for i in 0..fftsize {
        re[i] = input[i] * blackman_window(i as f64, fftsize as f64, window_length);
        im[i] = 0.0;
    }
    fft.forward(re, im);
    for k in 0..=fftsize / 2 {
        re[k] = complex_abs2(re[k], im[k]);
    }
    let index_lower = (100.0 / fs * fftsize as f64).floor() as usize;
    let index_upper1 = (4000.0 / fs * fftsize as f64).floor() as usize;
    let index_upper2 = (7900.0 / fs * fftsize as f64).floor() as usize;

    let mut weight1 = 1e-6;
    for k in index_lower + 1..=index_upper1 {
        weight1 += re[k];
    }
    let mut weight2 = weight1;
    for k in index_upper1 + 1..=index_upper2 {
        weight2 += re[k];
    }
    weight1 / weight2 > 0.7
}

impl ApAnalyzer {
    fn new(cfg: &Config) -> Self {
        ApAnalyzer { x_real: vec![0.0; cfg.fftsize], x_imag: vec![0.0; cfg.fftsize] }
    }

    /// Returns true for a voiced frame; writes per-bin aperiodicity into `ap`.
    fn analyze(&mut self, cfg: &Config, fft: &Fft, input: &[f64], fo: f64, issilence: bool, ap: &mut [f64]) -> bool {
        let numbins = cfg.numbins;
        // Mirror the C guard exactly (analyze_ap.c:79): bail to unvoiced when silent
        // or fo is out of range. The negated range test (rather than `>=`/`<=`) makes
        // a NaN fo fall through to the V/UV decision, matching C's `<`/`>` semantics.
        let out_of_range = fo < cfg.fo_floor || fo > cfg.fo_ceil;
        let voiced = !issilence
            && !out_of_range
            && !low_band_dominated(input, &mut self.x_real, &mut self.x_imag, cfg.fftsize, fo, cfg.fs, fft)
            && estimate_is_voiced(input, &mut self.x_real, &mut self.x_imag, cfg.fftsize, fo, cfg.fs, fft);
        let value = if voiced { 1e-3 } else { 1.0 };
        for a in ap.iter_mut().take(numbins) {
            *a = value;
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
    pspec: Vec<f64>,
    spec_cumsum: Vec<f64>,
}

/// Mirror bins `[numbins, fftsize)` from the lower half (Hermitian magnitude).
fn mirror_upper(arr: &mut [f64], numbins: usize) {
    for k in 0..numbins - 2 {
        arr[numbins + k] = arr[numbins - 2 - k];
    }
}

fn apply_replica(pspec: &mut [f64], numbins: usize, fo: f64, fs: f64) {
    let fftsize = 2 * (numbins - 1);
    let fobin = 1 + (fo / (fs / 2.0) * (numbins - 1) as f64).round() as usize;
    for k in 0..fobin {
        pspec[k] += pspec[fftsize - fobin - k];
    }
    mirror_upper(pspec, numbins);
}

fn smooth_spectrum(pspec: &mut [f64], spec_cumsum: &mut [f64], numbins: usize, freq_range: f64, fs: f64) {
    let fftsize = 2 * (numbins - 1);
    let offset = numbins - 2;
    spec_cumsum[0] = pspec[numbins];
    for k in 1..offset {
        spec_cumsum[k] = pspec[numbins + k] + spec_cumsum[k - 1];
    }
    for k in 0..fftsize {
        spec_cumsum[offset + k] = pspec[k] + spec_cumsum[offset + k - 1];
    }

    let half_range = freq_range / fs * (numbins - 1) as f64;
    let half_range_int = half_range.floor() as usize;
    let half_range_frc = half_range - half_range_int as f64;
    for k in 0..numbins {
        let index_upper = offset + k + half_range_int;
        let index_lower = offset + k - half_range_int;
        let upper = (1.0 - half_range_frc) * spec_cumsum[index_upper - 1] + half_range_frc * spec_cumsum[index_upper];
        let lower = (1.0 - half_range_frc) * spec_cumsum[index_lower] + half_range_frc * spec_cumsum[index_lower - 1];
        pspec[k] = (upper - lower).max(1e-12) / (2.0 * half_range);
    }
    mirror_upper(pspec, numbins);
}

fn lifter_spectrum(pspec: &mut [f64], imag: &mut [f64], numbins: usize, fo: f64, fs: f64, fft: &Fft) {
    let fftsize = 2 * (numbins - 1);
    for k in 0..fftsize {
        pspec[k] = (pspec[k] + 1e-12).ln();
        imag[k] = 0.0;
    }
    fft.inverse(pspec, imag);

    let q = -0.15;
    for k in 0..numbins {
        let t = k as f64 * fo / fs;
        let sinct = (REIM_PI * t + 1e-12).sin() / (REIM_PI * t + 1e-12);
        pspec[k] *= sinct * ((1.0 - 2.0 * q) + 2.0 * q * (2.0 * REIM_PI * t).cos());
        imag[k] = 0.0;
    }
    for k in 0..numbins - 2 {
        pspec[numbins + k] = pspec[numbins - 2 - k];
        imag[numbins + k] = 0.0;
    }

    fft.forward(pspec, imag);
    for k in 0..numbins {
        pspec[k] = pspec[k].exp();
    }
    mirror_upper(pspec, numbins);
}

impl SpAnalyzer {
    fn new(cfg: &Config) -> Self {
        SpAnalyzer {
            window: vec![0.0; cfg.fftsize],
            x_real: vec![0.0; cfg.fftsize],
            x_imag: vec![0.0; cfg.fftsize],
            pspec: vec![0.0; cfg.fftsize],
            spec_cumsum: vec![0.0; cfg.numbins + cfg.fftsize],
        }
    }

    fn analyze(&mut self, cfg: &Config, fft: &Fft, input: &[f64], fo: f64, isvoiced: bool, issilence: bool, sp: &mut [f64]) {
        let fs = cfg.fs;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        if issilence {
            for s in sp.iter_mut().take(numbins) {
                *s = 1e-12;
            }
            return;
        }

        let window_fo = if isvoiced { fo } else { 1.0 / (cfg.period / 1000.0) };
        let smooth_fo = if isvoiced { fo } else { 300.0 };

        let analysis_interval = fs / window_fo;
        let window_length = (3.0 * analysis_interval).min(fftsize as f64);
        let window_scale = 1.0 / analysis_interval.sqrt();
        for i in 0..fftsize {
            self.window[i] = hanning_window(i as f64, fftsize as f64, window_length) * window_scale;
            self.x_real[i] = input[i] * self.window[i];
            self.x_imag[i] = 0.0;
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

        fft.forward(&mut self.x_real, &mut self.x_imag);
        for k in 0..numbins {
            self.pspec[k] = complex_abs2(self.x_real[k], self.x_imag[k]);
        }
        mirror_upper(&mut self.pspec, numbins);

        apply_replica(&mut self.pspec, numbins, window_fo, fs);
        smooth_spectrum(&mut self.pspec, &mut self.spec_cumsum, numbins, smooth_fo / 2.0, fs);
        if isvoiced {
            lifter_spectrum(&mut self.pspec, &mut self.x_imag, numbins, smooth_fo, fs, fft);
        }

        sp[..numbins].copy_from_slice(&self.pspec[..numbins]);
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
    pulse_frc: f64,
    random: Xorshift,
    interval_velvet: usize,
    interval_random: usize,
    gain_noise: f64,
    noise_int: usize,
    buffer: CircularQueue,
}

/// Build the minimum-phase complex spectrum from a power spectrum (in `spec_r`),
/// scaled by `gain`. `spec_i` is used as scratch. Result occupies both arrays.
fn generate_minimum_phase_spectrum(spec_r: &mut [f64], spec_i: &mut [f64], gain: f64, fftsize: usize, fft: &Fft) {
    let numbins = fftsize / 2 + 1;
    for k in 0..fftsize {
        spec_r[k] = (spec_r[k] + 1e-12).ln();
        spec_i[k] = 0.0;
    }
    fft.inverse(spec_r, spec_i);

    spec_r[0] *= 0.5;
    spec_i[0] *= 0.5;
    spec_r[numbins - 1] *= 0.5;
    spec_i[numbins - 1] *= 0.5;
    for k in numbins..fftsize {
        spec_r[k] = 0.0;
        spec_i[k] = 0.0;
    }

    fft.forward(spec_r, spec_i);
    for k in 0..numbins {
        let a = gain * spec_r[k].exp();
        let b = spec_i[k];
        spec_r[k] = a * b.cos();
        spec_i[k] = a * b.sin();
    }
    for k in 0..numbins - 2 {
        spec_r[numbins + k] = spec_r[numbins - 2 - k];
        spec_i[numbins + k] = -spec_i[numbins - 2 - k];
    }
}

/// Generate a (optionally fractionally shifted) impulse response from a spectrum.
fn generate_impulse(impulse: &mut [f64], spec_r: &[f64], spec_i: &[f64], shift: f64, window: &[f64], temp_r: &mut [f64], temp_i: &mut [f64], fftsize: usize, fft: &Fft) {
    let numbins = fftsize / 2 + 1;
    if shift == 0.0 {
        temp_r[..fftsize].copy_from_slice(&spec_r[..fftsize]);
        temp_i[..fftsize].copy_from_slice(&spec_i[..fftsize]);
    } else {
        for k in 0..numbins {
            let omega = -REIM_PI * shift * k as f64 / (numbins - 1) as f64;
            temp_r[k] = omega.cos();
            temp_i[k] = omega.sin();
        }
        for k in 0..numbins - 2 {
            temp_r[numbins + k] = temp_r[numbins - 2 - k];
            temp_i[numbins + k] = -temp_i[numbins - 2 - k];
        }
        for k in 0..fftsize {
            let xr1 = temp_r[k];
            let xi1 = temp_i[k];
            let xr2 = spec_r[k];
            let xi2 = spec_i[k];
            temp_r[k] = xr1 * xr2 - xi1 * xi2;
            temp_i[k] = xr1 * xi2 + xi1 * xr2;
        }
    }

    fft.inverse(temp_r, temp_i);
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
            pulse_frc: 0.0,
            random: Xorshift::new(),
            interval_velvet,
            interval_random: 0,
            gain_noise: (interval_velvet as f64).sqrt(),
            noise_int: 0,
            buffer: CircularQueue::new(period_max + fftsize),
        }
    }

    fn new_frame(&mut self, cfg: &Config, fft: &Fft, fo: f64, isvoiced: bool, issilence: bool, ap: &[f64], sp: &[f64]) {
        let fs = cfg.fs;
        let fftsize = cfg.fftsize;
        let numbins = cfg.numbins;

        for k in 0..numbins {
            let spec = sp[k];
            let aper = ap[k] * ap[k];
            self.spec_pulse_r[k] = spec * (1.0 - aper);
            self.spec_noise_r[k] = spec * aper;
        }
        mirror_upper(&mut self.spec_pulse_r, numbins);
        mirror_upper(&mut self.spec_noise_r, numbins);

        self.has_pulse = isvoiced && !issilence;
        if self.has_pulse {
            self.interval = fs / fo;
            let gain_pulse = self.interval.sqrt();
            generate_minimum_phase_spectrum(&mut self.spec_pulse_r, &mut self.spec_pulse_i, gain_pulse, fftsize, fft);
        }

        self.has_noise = !issilence;
        if self.has_noise {
            let gain_noise = self.gain_noise;
            generate_minimum_phase_spectrum(&mut self.spec_noise_r, &mut self.spec_noise_i, gain_noise, fftsize, fft);
            generate_impulse(&mut self.impulse_noise, &self.spec_noise_r, &self.spec_noise_i, 0.0, &self.window, &mut self.temp_r, &mut self.temp_i, fftsize, fft);
        }
    }

    fn next_sample(&mut self, cfg: &Config, fft: &Fft) -> f64 {
        let fftsize = cfg.fftsize;

        if self.has_pulse {
            if self.pulse_int == 0 {
                generate_impulse(&mut self.impulse_pulse, &self.spec_pulse_r, &self.spec_pulse_i, self.pulse_frc, &self.window, &mut self.temp_r, &mut self.temp_i, fftsize, fft);
                self.buffer.push_additive(&self.impulse_pulse);

                let interval_int = self.interval.floor();
                let interval_frc = self.interval - interval_int;
                let next = self.pulse_frc + interval_frc;
                let carry = next.floor();
                self.pulse_int += (interval_int + carry) as i32;
                self.pulse_frc = next - carry;
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

pub struct Reim {
    cfg: Config,
    fft: Fft,
    framer: Framer,
    fo: FoAnalyzer,
    ap: ApAnalyzer,
    sp: SpAnalyzer,
    syn: Synth,
    wf: Vec<f64>,     // fftsize+1 frame window (oldest..newest)
    ap_buf: Vec<f64>, // numbins
    sp_buf: Vec<f64>, // numbins
    last_fo: f64,
    last_voiced: bool,
    last_silence: bool,
    frame_count: u64,
}

impl Reim {
    pub fn new(fs: f64, period: f64, fftsize: usize, fo_floor: f64, fo_ceil: f64) -> Self {
        let cfg = Config::new(period, fftsize, fo_floor, fo_ceil, fs);
        let fft = Fft::new(fftsize);
        let fo = FoAnalyzer::new(&cfg, &fft);
        let ap = ApAnalyzer::new(&cfg);
        let sp = SpAnalyzer::new(&cfg);
        let syn = Synth::new(&cfg);
        Reim {
            cfg,
            fft,
            framer: Framer::new(&cfg),
            fo,
            ap,
            sp,
            syn,
            wf: vec![0.0; fftsize + 1],
            ap_buf: vec![0.0; cfg.numbins],
            sp_buf: vec![0.0; cfg.numbins],
            last_fo: 0.0,
            last_voiced: false,
            last_silence: false,
            frame_count: 0,
        }
    }

    /// Default configuration matching the reference example.
    pub fn with_defaults(fs: f64) -> Self {
        Reim::new(fs, 5.0, 2048, 71.0, 800.0)
    }

    /// Process one input sample, returning one output sample. Allocation-free.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        if self.framer.next(input, &mut self.wf) {
            let fftsize = self.cfg.fftsize;
            // `wave` is the frame; `wave_d` is the same frame delayed by one sample.
            let (wave_d, wave) = (&self.wf[..fftsize], &self.wf[1..fftsize + 1]);
            let silence = analyze_silence(&self.cfg, wave, SILENCE_THRESHOLD);
            let fo = self.fo.analyze(&self.cfg, &self.fft, wave, wave_d);
            let voiced = self.ap.analyze(&self.cfg, &self.fft, wave, fo, silence, &mut self.ap_buf);
            self.sp.analyze(&self.cfg, &self.fft, wave, fo, voiced, silence, &mut self.sp_buf);
            self.syn.new_frame(&self.cfg, &self.fft, fo, voiced, silence, &self.ap_buf, &self.sp_buf);
            self.last_fo = fo;
            self.last_voiced = voiced;
            self.last_silence = silence;
            self.frame_count += 1;
        }
        self.syn.next_sample(&self.cfg, &self.fft)
    }

    /// Process a block in place-ish: returns a new output vector (convenience).
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        for (o, &x) in output.iter_mut().zip(input) {
            *o = self.process_sample(x);
        }
    }

    /// Number of frames analyzed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
    /// Most recent frame's estimated Fo in Hz (0.0 when no pitch).
    pub fn last_fo(&self) -> f64 {
        self.last_fo
    }
    /// Whether the most recent frame was voiced.
    pub fn last_voiced(&self) -> bool {
        self.last_voiced
    }
    /// Whether the most recent frame was silence.
    pub fn last_silence(&self) -> bool {
        self.last_silence
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
        (1, 16) => body.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0).collect(),
        (1, 8) => body.iter().map(|&b| (b as f64 - 128.0) / 128.0).collect(),
        (3, 32) => body.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
        _ => return Err(format!("unsupported format tag {format} / {bits} bits")),
    };
    Ok(WavData { samples, sample_rate: rate })
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
        let cfg = r.cfg;
        let fftsize = cfg.fftsize;
        let (mut t_sil, mut t_fo, mut t_ap, mut t_sp, mut t_nf, mut t_ns) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut frame_latencies = Vec::new();
        for &x in samples {
            if r.framer.next(x, &mut r.wf) {
                let frame_t0 = Instant::now();
                let (wave_d, wave) = (&r.wf[..fftsize], &r.wf[1..fftsize + 1]);
                let t = Instant::now();
                let silence = analyze_silence(&cfg, wave, SILENCE_THRESHOLD);
                t_sil += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let fo = r.fo.analyze(&cfg, &r.fft, wave, wave_d);
                t_fo += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let voiced = r.ap.analyze(&cfg, &r.fft, wave, fo, silence, &mut r.ap_buf);
                t_ap += t.elapsed().as_secs_f64();
                let t = Instant::now();
                r.sp.analyze(&cfg, &r.fft, wave, fo, voiced, silence, &mut r.sp_buf);
                t_sp += t.elapsed().as_secs_f64();
                let t = Instant::now();
                r.syn.new_frame(&cfg, &r.fft, fo, voiced, silence, &r.ap_buf, &r.sp_buf);
                t_nf += t.elapsed().as_secs_f64();
                frame_latencies.push(frame_t0.elapsed().as_secs_f64());
            }
            let t = Instant::now();
            let _ = r.syn.next_sample(&cfg, &r.fft);
            t_ns += t.elapsed().as_secs_f64();
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
    fn fft_complex_exp() {
        let fft = Fft::new(8);
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
        let fft = Fft::new(64);
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
        assert!(out.iter().all(|&x| x == 0.0), "silent input must yield silence");
    }

    #[test]
    fn pure_tone_tracks_fo() {
        // 200 Hz tone should be detected as ~200 Hz on voiced frames.
        let fs = 24000.0;
        let n = (fs * 0.5) as usize;
        let input: Vec<f64> = (0..n).map(|i| 0.5 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin()).collect();
        let mut r = Reim::with_defaults(fs);
        let mut last = 0u64;
        let mut fos = Vec::new();
        for &x in &input {
            r.process_sample(x);
            if r.frame_count != last {
                last = r.frame_count;
                if r.last_fo > 0.0 {
                    fos.push(r.last_fo);
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
    fn ap_guard_matches_c_nan_semantics() {
        // C's analyze_ap bails on `fo < floor || fo > ceil`; for fo==NaN both are
        // false, so it PROCEEDS to the V/UV decision. The Rust guard must do the
        // same rather than short-circuiting NaN straight to unvoiced.
        let cfg = Config::new(5.0, 2048, 71.0, 800.0, 24000.0);
        let fft = Fft::new(cfg.fftsize);
        let mut ap = ApAnalyzer::new(&cfg);
        let input: Vec<f64> = (0..cfg.fftsize)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / cfg.fs).sin())
            .collect();
        let mut buf = vec![0.0; cfg.numbins];
        let voiced = ap.analyze(&cfg, &fft, &input, f64::NAN, false, &mut buf);
        // independent reference: the decision C reaches for a NaN fo
        let mut re = vec![0.0; cfg.fftsize];
        let mut im = vec![0.0; cfg.fftsize];
        let expected = estimate_is_voiced(&input, &mut re, &mut im, cfg.fftsize, f64::NAN, cfg.fs, &fft);
        assert_eq!(voiced, expected, "NaN fo must reach the V/UV decision, as in C");
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
}
