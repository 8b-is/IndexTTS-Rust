//! Audio processing module for IndexTTS
//!
//! Provides mel-spectrogram computation, audio I/O, and DSP operations.

mod dsp;
mod io;
pub mod mel;
mod resample;

pub use dsp::{
    apply_fade, apply_preemphasis, dynamic_range_compression, dynamic_range_decompression,
    normalize_audio, normalize_audio_peak,
};
pub use io::{load_audio, save_audio, AudioData};
pub use mel::{mel_spectrogram, mel_to_linear, MelFilterbank};
pub use resample::resample;

use crate::Result;

/// Audio processing configuration
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// FFT size
    pub n_fft: usize,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Window length
    pub win_length: usize,
    /// Number of mel bands
    pub n_mels: usize,
    /// Minimum frequency
    pub fmin: f32,
    /// Maximum frequency
    pub fmax: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 768_000,
            n_fft: 32768,
            hop_length: 8192,
            win_length: 32768,
            n_mels: 80,
            fmin: 0.0,
            fmax: 384_000.0,
        }
    }
}

/// Compute mel spectrogram from audio file
pub fn compute_mel_from_file(path: &str, config: &AudioConfig) -> Result<ndarray::Array2<f32>> {
    let audio = load_audio(path, Some(config.sample_rate))?;
    mel_spectrogram(&audio.samples, config)
}
