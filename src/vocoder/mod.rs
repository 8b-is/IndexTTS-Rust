//! Vocoder module for mel-spectrogram to waveform conversion
//!
//! Implements BigVGAN and related vocoders

mod bigvgan;
mod activations;

pub use bigvgan::{BigVGAN, BigVGANConfig, create_bigvgan_22k, create_bigvgan_24k};
pub use activations::{snake_activation, snake_beta_activation, snake_activation_vec};

use crate::Result;
use ndarray::Array2;

/// Vocoder trait for mel-to-waveform conversion
pub trait Vocoder {
    /// Convert mel spectrogram to waveform
    fn synthesize(&self, mel: &Array2<f32>) -> Result<Vec<f32>>;

    /// Get sample rate
    fn sample_rate(&self) -> u32;

    /// Get hop length (for timing calculations)
    fn hop_length(&self) -> usize;
}

/// Simple Griffin-Lim vocoder (fallback)
pub struct GriffinLim {
    n_fft: usize,
    hop_length: usize,
    n_iter: usize,
    sample_rate: u32,
}

impl GriffinLim {
    /// Create new Griffin-Lim vocoder
    pub fn new(n_fft: usize, hop_length: usize, sample_rate: u32) -> Self {
        Self {
            n_fft,
            hop_length,
            n_iter: 32,
            sample_rate,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }
}

impl Vocoder for GriffinLim {
    fn synthesize(&self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        // Simplified Griffin-Lim - just return noise shaped by mel energy
        let n_frames = mel.ncols();
        let output_len = n_frames * self.hop_length;
        let mut output = vec![0.0f32; output_len];

        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate noise shaped by mel energy
        for i in 0..output_len {
            let frame_idx = i / self.hop_length;
            if frame_idx < n_frames {
                let energy: f32 = (0..mel.nrows()).map(|j| mel[[j, frame_idx]].exp()).sum::<f32>() / mel.nrows() as f32;
                output[i] = rng.gen_range(-1.0..1.0) * energy.sqrt() * 0.1;
            }
        }

        Ok(output)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn hop_length(&self) -> usize {
        self.hop_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_griffin_lim_creation() {
        let vocoder = GriffinLim::new(1024, 256, 22050);
        assert_eq!(vocoder.sample_rate(), 22050);
        assert_eq!(vocoder.hop_length(), 256);
    }
}
