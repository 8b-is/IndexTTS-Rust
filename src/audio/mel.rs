//! Mel-spectrogram computation
//!
//! Implements Short-Time Fourier Transform (STFT) and mel filterbank

use crate::{Error, Result};
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex;
use realfft::RealFftPlanner;
use std::f32::consts::PI;

use super::AudioConfig;

/// Mel filterbank for converting linear spectrogram to mel scale
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    /// Filterbank matrix (n_mels x n_fft/2+1)
    pub filters: Array2<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of mel bands
    pub n_mels: usize,
    /// FFT size
    pub n_fft: usize,
}

impl MelFilterbank {
    /// Create mel filterbank
    pub fn new(sample_rate: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Self {
        let filters = create_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax);
        Self {
            filters,
            sample_rate,
            n_mels,
            n_fft,
        }
    }

    /// Apply filterbank to power spectrogram
    pub fn apply(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        // spectrogram: (n_fft/2+1, time_frames)
        // filters: (n_mels, n_fft/2+1)
        // output: (n_mels, time_frames)
        self.filters.dot(spectrogram)
    }
}

/// Convert frequency to mel scale
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to frequency
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank matrix
fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;

    // Convert to mel scale
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Create mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin numbers
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft as f32 + 1.0) * hz / sample_rate as f32).floor() as usize)
        .collect();

    // Create filterbank
    let mut filters = Array2::zeros((n_mels, n_freqs));

    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        // Left slope
        for k in f_left..f_center {
            if k < n_freqs {
                filters[[m, k]] = (k - f_left) as f32 / (f_center - f_left).max(1) as f32;
            }
        }

        // Right slope
        for k in f_center..f_right {
            if k < n_freqs {
                filters[[m, k]] = (f_right - k) as f32 / (f_right - f_center).max(1) as f32;
            }
        }
    }

    filters
}

/// Compute Hann window
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / size as f32).cos()))
        .collect()
}

/// Compute Short-Time Fourier Transform (STFT)
///
/// # Arguments
/// * `signal` - Input audio signal
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length between frames
/// * `win_length` - Window length (padded to n_fft)
///
/// # Returns
/// Complex STFT matrix (n_fft/2+1, time_frames)
pub fn stft(
    signal: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<Complex<f32>>> {
    if signal.is_empty() {
        return Err(Error::Audio("Empty signal".into()));
    }

    // Create window
    let window = hann_window(win_length);

    // Pad signal
    let pad_length = n_fft / 2;
    let mut padded = vec![0.0f32; pad_length];
    padded.extend_from_slice(signal);
    padded.extend(vec![0.0f32; pad_length]);

    // Calculate number of frames
    let num_frames = (padded.len() - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;

    // Create FFT planner
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Output matrix
    let mut stft_matrix = Array2::zeros((n_freqs, num_frames));

    // Process each frame
    let mut input_buffer = vec![0.0f32; n_fft];
    let mut output_buffer = vec![Complex::new(0.0f32, 0.0f32); n_freqs];

    for (frame_idx, start) in (0..padded.len() - n_fft + 1)
        .step_by(hop_length)
        .enumerate()
    {
        if frame_idx >= num_frames {
            break;
        }

        // Extract and window the frame
        for i in 0..win_length {
            input_buffer[i] = padded[start + i] * window[i];
        }
        // Zero pad if win_length < n_fft
        for i in win_length..n_fft {
            input_buffer[i] = 0.0;
        }

        // Perform FFT
        fft.process(&mut input_buffer, &mut output_buffer)
            .map_err(|e| Error::Audio(format!("FFT failed: {}", e)))?;

        // Store result
        for (freq_idx, &val) in output_buffer.iter().enumerate() {
            stft_matrix[[freq_idx, frame_idx]] = val;
        }
    }

    Ok(stft_matrix)
}

/// Compute magnitude spectrogram from STFT
pub fn magnitude_spectrogram(stft_matrix: &Array2<Complex<f32>>) -> Array2<f32> {
    stft_matrix.mapv(|c| c.norm())
}

/// Compute power spectrogram from STFT
pub fn power_spectrogram(stft_matrix: &Array2<Complex<f32>>) -> Array2<f32> {
    stft_matrix.mapv(|c| c.norm_sqr())
}

/// Compute mel spectrogram from audio signal
///
/// # Arguments
/// * `signal` - Audio samples
/// * `config` - Audio configuration
///
/// # Returns
/// Log mel spectrogram (n_mels, time_frames)
pub fn mel_spectrogram(signal: &[f32], config: &AudioConfig) -> Result<Array2<f32>> {
    // Compute STFT
    let stft_matrix = stft(signal, config.n_fft, config.hop_length, config.win_length)?;

    // Compute power spectrogram
    let power_spec = power_spectrogram(&stft_matrix);

    // Create mel filterbank
    let mel_fb = MelFilterbank::new(
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.fmin,
        config.fmax,
    );

    // Apply mel filterbank
    let mel_spec = mel_fb.apply(&power_spec);

    // Apply log compression
    let log_mel_spec = mel_spec.mapv(|x| (x.max(1e-10)).ln());

    Ok(log_mel_spec)
}

/// Compute mel spectrogram with normalization
pub fn mel_spectrogram_normalized(
    signal: &[f32],
    config: &AudioConfig,
    mean: Option<f32>,
    std: Option<f32>,
) -> Result<Array2<f32>> {
    let mut mel_spec = mel_spectrogram(signal, config)?;

    // Normalize
    if let (Some(m), Some(s)) = (mean, std) {
        mel_spec.mapv_inplace(|x| (x - m) / s);
    } else {
        // Compute statistics from spectrogram
        let m = mel_spec.mean().unwrap_or(0.0);
        let s = mel_spec.std(0.0);
        if s > 1e-8 {
            mel_spec.mapv_inplace(|x| (x - m) / s);
        }
    }

    Ok(mel_spec)
}

/// Convert mel spectrogram back to linear spectrogram (approximate)
pub fn mel_to_linear(mel_spec: &Array2<f32>, mel_fb: &MelFilterbank) -> Array2<f32> {
    // Pseudo-inverse of mel filterbank
    let filters_t = mel_fb.filters.t();
    let gram = mel_fb.filters.dot(&filters_t);

    // Simple approximation using transpose
    filters_t.dot(mel_spec)
}

/// Compute spectrogram energy per frame
pub fn frame_energy(mel_spec: &Array2<f32>) -> Array1<f32> {
    mel_spec.sum_axis(Axis(0))
}

/// Detect voice activity based on energy threshold
pub fn voice_activity_detection(mel_spec: &Array2<f32>, threshold_db: f32) -> Vec<bool> {
    let energy = frame_energy(mel_spec);
    let max_energy = energy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_energy + threshold_db; // threshold_db is negative

    energy.iter().map(|&e| e > threshold).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_mel() {
        // Test known conversions
        assert!((hz_to_mel(0.0) - 0.0).abs() < 1e-6);
        assert!((hz_to_mel(1000.0) - 1000.0).abs() < 50.0); // Roughly linear at low freqs
    }

    #[test]
    fn test_mel_to_hz() {
        // Round trip
        let hz = 440.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 1e-4);
    }

    #[test]
    fn test_mel_filterbank_creation() {
        let fb = MelFilterbank::new(22050, 1024, 80, 0.0, 8000.0);
        assert_eq!(fb.filters.shape(), &[80, 513]);

        // Check that filters are non-empty (some filter banks have coverage)
        let total_sum: f32 = fb.filters.iter().sum();
        assert!(total_sum > 0.0, "Filterbank should have some non-zero values");
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(1024);
        assert_eq!(window.len(), 1024);
        // Check endpoints are near zero
        assert!(window[0].abs() < 1e-6);
        // Check middle is near 1
        assert!((window[512] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_stft_basic() {
        // Create a simple sine wave
        let sr = 22050;
        let freq = 440.0;
        let duration = 0.1;
        let num_samples = (sr as f32 * duration) as usize;

        let signal: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let result = stft(&signal, 1024, 256, 1024);
        assert!(result.is_ok());

        let stft_matrix = result.unwrap();
        assert_eq!(stft_matrix.shape()[0], 513); // n_fft/2 + 1
        assert!(stft_matrix.shape()[1] > 0); // Some frames
    }

    #[test]
    fn test_mel_spectrogram() {
        let config = AudioConfig::default();
        let num_samples = (config.sample_rate as f32 * 0.1) as usize;
        let signal: Vec<f32> = (0..num_samples).map(|i| (i as f32 * 0.01).sin()).collect();

        let result = mel_spectrogram(&signal, &config);
        assert!(result.is_ok());

        let mel_spec = result.unwrap();
        assert_eq!(mel_spec.shape()[0], config.n_mels);
        assert!(mel_spec.shape()[1] > 0);
    }
}
