//! BigVGAN vocoder implementation
//!
//! High-quality neural vocoder for mel-spectrogram to waveform conversion

use crate::{Error, Result};
use ndarray::{Array2, IxDyn};
use std::collections::HashMap;
use std::path::Path;

use crate::model::OnnxSession;
use super::{Vocoder, snake_activation_vec};

/// BigVGAN configuration
#[derive(Debug, Clone)]
pub struct BigVGANConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// Number of mel channels
    pub num_mels: usize,
    /// Upsampling rates
    pub upsample_rates: Vec<usize>,
    /// Upsampling kernel sizes
    pub upsample_kernel_sizes: Vec<usize>,
    /// ResBlock kernel sizes
    pub resblock_kernel_sizes: Vec<usize>,
    /// ResBlock dilation sizes
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// Initial channel size
    pub upsample_initial_channel: usize,
    /// Use anti-aliasing
    pub use_anti_alias: bool,
}

impl Default for BigVGANConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            num_mels: 80,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_initial_channel: 512,
            use_anti_alias: true,
        }
    }
}

impl BigVGANConfig {
    /// Calculate total upsampling factor
    pub fn total_upsample_factor(&self) -> usize {
        self.upsample_rates.iter().product()
    }

    /// Get hop length (same as upsample factor)
    pub fn hop_length(&self) -> usize {
        self.total_upsample_factor()
    }
}

/// BigVGAN vocoder
pub struct BigVGAN {
    session: Option<OnnxSession>,
    config: BigVGANConfig,
}

impl BigVGAN {
    /// Load BigVGAN from ONNX model
    pub fn load<P: AsRef<Path>>(path: P, config: BigVGANConfig) -> Result<Self> {
        let session = OnnxSession::load(path)?;
        Ok(Self {
            session: Some(session),
            config,
        })
    }

    /// Create BigVGAN with fallback synthesizer
    pub fn new_fallback(config: BigVGANConfig) -> Self {
        Self {
            session: None,
            config,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BigVGANConfig {
        &self.config
    }

    /// Synthesize audio using fallback algorithm
    fn synthesize_fallback(&self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        // Simple overlap-add synthesis as fallback
        let num_frames = mel.ncols();
        let hop_length = self.config.hop_length();
        let frame_size = hop_length * 4; // Use 4x overlap

        let output_length = (num_frames - 1) * hop_length + frame_size;
        let mut output = vec![0.0f32; output_length];
        let mut window_sum = vec![0.0f32; output_length];

        // Hann window
        let window: Vec<f32> = (0..frame_size)
            .map(|n| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / frame_size as f32).cos())
            })
            .collect();

        // Generate frames from mel
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;

            // Generate frame from mel (simplified: use mel features to modulate noise)
            let mel_frame: Vec<f32> = (0..self.config.num_mels)
                .map(|i| mel[[i, frame_idx]])
                .collect();

            // Generate frame using mel features
            let frame = self.generate_frame(&mel_frame, frame_size);

            // Overlap-add
            for i in 0..frame_size {
                if start + i < output_length {
                    output[start + i] += frame[i] * window[i];
                    window_sum[start + i] += window[i] * window[i];
                }
            }
        }

        // Normalize by window sum
        for i in 0..output_length {
            if window_sum[i] > 1e-8 {
                output[i] /= window_sum[i];
            }
        }

        // Apply post-processing
        let output = snake_activation_vec(&output, 0.3);

        Ok(output)
    }

    /// Generate a single frame from mel features
    fn generate_frame(&self, mel: &[f32], frame_size: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Compute overall energy from mel
        let energy: f32 = mel.iter().map(|x| x.exp()).sum::<f32>() / mel.len() as f32;
        let energy = energy.sqrt().min(2.0);

        // Generate frame with harmonic content
        let mut frame = vec![0.0f32; frame_size];

        // Use mel bands to create frequency content
        for (freq_idx, &mel_val) in mel.iter().enumerate() {
            let freq = (freq_idx as f32 / mel.len() as f32) * (self.config.sample_rate as f32 / 2.0);
            let amplitude = mel_val.exp().min(1.0) * 0.1;

            // Add harmonic
            for i in 0..frame_size {
                let t = i as f32 / self.config.sample_rate as f32;
                frame[i] += amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
        }

        // Add filtered noise
        for i in 0..frame_size {
            frame[i] += rng.gen_range(-0.1..0.1) * energy * 0.1;
        }

        // Normalize
        let max_abs = frame.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_abs > 1.0 {
            for v in frame.iter_mut() {
                *v /= max_abs;
            }
        }

        frame
    }

    /// Apply post-processing to output
    pub fn post_process(&self, audio: &[f32]) -> Vec<f32> {
        use crate::audio::{normalize_audio, apply_fade};

        let normalized = normalize_audio(audio);

        // Apply fade to avoid clicks
        let fade_samples = (self.config.sample_rate as f32 * 0.01) as usize; // 10ms fade
        apply_fade(&normalized, fade_samples, fade_samples)
    }
}

impl Vocoder for BigVGAN {
    fn synthesize(&self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        if let Some(ref session) = self.session {
            // Use ONNX model
            let input = mel.clone().into_shape(IxDyn(&[1, mel.nrows(), mel.ncols()]))?;

            let mut inputs = HashMap::new();
            inputs.insert("mel".to_string(), input);

            let outputs = session.run(inputs)?;

            let audio = outputs
                .get("audio")
                .ok_or_else(|| Error::Vocoder("Missing audio output".into()))?;

            // Extract audio samples
            let samples: Vec<f32> = audio.iter().cloned().collect();

            Ok(self.post_process(&samples))
        } else {
            // Use fallback synthesis
            let audio = self.synthesize_fallback(mel)?;
            Ok(self.post_process(&audio))
        }
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn hop_length(&self) -> usize {
        self.config.hop_length()
    }
}

/// Helper function to create BigVGAN for 22kHz audio
pub fn create_bigvgan_22k() -> BigVGAN {
    let config = BigVGANConfig {
        sample_rate: 22050,
        ..Default::default()
    };
    BigVGAN::new_fallback(config)
}

/// Helper function to create BigVGAN for 24kHz audio
pub fn create_bigvgan_24k() -> BigVGAN {
    let config = BigVGANConfig {
        sample_rate: 24000,
        upsample_rates: vec![12, 10, 2, 2],
        ..Default::default()
    };
    BigVGAN::new_fallback(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigvgan_config() {
        let config = BigVGANConfig::default();
        assert_eq!(config.total_upsample_factor(), 256);
        assert_eq!(config.hop_length(), 256);
    }

    #[test]
    fn test_bigvgan_fallback() {
        let vocoder = create_bigvgan_22k();
        assert_eq!(vocoder.sample_rate(), 22050);

        // Create small test mel
        let mel = Array2::zeros((80, 10));
        let result = vocoder.synthesize(&mel);
        assert!(result.is_ok());

        let audio = result.unwrap();
        assert!(audio.len() > 0);
    }

    #[test]
    fn test_generate_frame() {
        let vocoder = create_bigvgan_22k();
        let mel = vec![0.0f32; 80];
        let frame = vocoder.generate_frame(&mel, 256);
        assert_eq!(frame.len(), 256);
    }

    #[test]
    fn test_post_process() {
        let vocoder = create_bigvgan_22k();
        let audio = vec![0.5f32; 1000];
        let processed = vocoder.post_process(&audio);
        assert_eq!(processed.len(), audio.len());
        // Check fade was applied (first samples should be smaller)
        assert!(processed[0].abs() < 0.1);
    }
}
