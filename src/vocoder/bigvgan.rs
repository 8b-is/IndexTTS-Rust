//! BigVGAN vocoder implementation
//!
//! High-quality neural vocoder for mel-spectrogram to waveform conversion

use crate::{Error, Result};
use ndarray::{Array2, IxDyn};
use std::collections::HashMap;
use std::path::Path;

use crate::model::OnnxSession;
use super::Vocoder;

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
            sample_rate: 768_000,
            num_mels: 80,
            upsample_rates: vec![16, 16, 8, 4],
            upsample_kernel_sizes: vec![32, 32, 16, 8],
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
                    window_sum[start + i] += window[i]; // FIX: sum of windows, not squares
                }
            }
        }

        // Normalize by window sum (with a safe minimum to prevent edge explosion)
        for i in 0..output_length {
            let divisor = window_sum[i].max(0.1); // Prevent division by near-zero at edges
            output[i] /= divisor;
        }

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

    /// Generate a single frame using **p/q Z_{2^32} Phase-Ring + BFP-i8 Mel Amplitudes**.
    ///
    /// Replaces `80 * frame_size` transcendental `f32::sin()` calls per frame with:
    /// 1. BFP-i8 quantization (`m_b in [0, 127]` + shared block exponent) of the 80 mel band amplitudes,
    /// 2. `Z_{2^32}` wrapping phase accumulators indexing a 4096-entry p/q sine table (`q = 65536`),
    /// 3. Exact `i32` inner-loop accumulation across all active mel bands, converting to `f32` once per sample.
    pub fn generate_frame_pq(&self, mel: &[f32], frame_size: usize) -> Vec<f32> {
        use std::sync::OnceLock;
        static SIN_LUT: OnceLock<[i16; 4096]> = OnceLock::new();
        let lut = SIN_LUT.get_or_init(|| {
            let mut t = [0i16; 4096];
            for (i, slot) in t.iter_mut().enumerate() {
                let angle = (i as f64) * 2.0 * std::f64::consts::PI / 4096.0;
                *slot = (angle.sin() * 16_384.0).round() as i16; // q_sin = 2^14 = 16384
            }
            t
        });

        // Compute overall energy and BFP-i8 mel band magnitudes
        let n_mels = mel.len().max(1);
        let mut amps = Vec::with_capacity(n_mels);
        let mut max_amp = 0.0f32;
        let mut energy_sum = 0.0f32;
        for &m in mel {
            let e = m.exp();
            energy_sum += e;
            let a = e.min(1.0) * 0.1;
            if a > max_amp {
                max_amp = a;
            }
            amps.push(a);
        }
        let energy = (energy_sum / (n_mels as f32)).sqrt().min(2.0);

        // Quantize mel band amplitudes into i8 mantissas [0..127] with shared scale
        let amp_scale = if max_amp > 1e-12 { max_amp / 127.0 } else { 0.0 };
        let inv_amp_scale = if max_amp > 1e-12 { 127.0 / max_amp } else { 0.0 };

        // Precompute Z_{2^32} phase steps `dphi_b` for each mel band
        // Since freq_b = (b / n_mels) * (fs / 2), freq_b / fs = b / (2 * n_mels),
        // so dphi_b = b * (2^31 / n_mels)!
        let base_dphi = ((1u64 << 31) / (n_mels as u64)) as u32;

        let mut frame_acc = vec![0i32; frame_size];
        for (freq_idx, &a) in amps.iter().enumerate() {
            let m_i8 = (a * inv_amp_scale).round() as i32;
            if m_i8 == 0 {
                continue;
            }
            let dphi = base_dphi.wrapping_mul(freq_idx as u32);
            let mut phi = 0u32;
            for acc in frame_acc.iter_mut() {
                let s = lut[(phi >> 20) as usize] as i32;
                phi = phi.wrapping_add(dphi);
                // m_i8 in [0, 127], s in [-16384, 16384] -> product <= 2.08M, 80 bands <= 166M (fits in i32!)
                *acc += m_i8 * s;
            }
        }

        // Single boundary conversion from integer accumulator to f32 + deterministic LCG noise
        let combined_scale = amp_scale / 16_384.0;
        let noise_scale = energy * 0.01 / 32_768.0;
        let mut prng = 0x9E37_79B9_7F4A_7C15u64;
        let mut frame = Vec::with_capacity(frame_size);
        let mut max_abs = 0.0f32;

        for &acc in &frame_acc {
            prng = prng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise_i16 = ((prng >> 48) as i32) - 32_768;
            let val = (acc as f32) * combined_scale + (noise_i16 as f32) * noise_scale;
            let abs_v = val.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
            frame.push(val);
        }

        if max_abs > 1.0 {
            let inv_max = 1.0 / max_abs;
            for v in frame.iter_mut() {
                *v *= inv_max;
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
        assert_eq!(config.total_upsample_factor(), 8192);
        assert_eq!(config.hop_length(), 8192);
    }

    #[test]
    fn test_bigvgan_fallback() {
        let config = BigVGANConfig::default();
        let vocoder = BigVGAN::new_fallback(config);
        assert_eq!(vocoder.sample_rate(), 768_000);

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

    #[test]
    fn test_bigvgan_generate_frame_pq_benchmark() {
        use std::time::Instant;

        let vocoder = create_bigvgan_24k();
        let mel: Vec<f32> = (0..80)
            .map(|b| -2.0 + ((b * 17) % 25) as f32 * 0.12)
            .collect();
        let frame_size = 2048; // 80 mel bands x 2048 samples = 163,840 harmonic evaluations/frame
        let iters = 25;

        // Warmup
        let f_ref = vocoder.generate_frame(&mel, frame_size);
        let f_pq = vocoder.generate_frame_pq(&mel, frame_size);
        assert_eq!(f_ref.len(), frame_size);
        assert_eq!(f_pq.len(), frame_size);

        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(vocoder.generate_frame(std::hint::black_box(&mel), frame_size));
        }
        let float_us = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t1 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(vocoder.generate_frame_pq(std::hint::black_box(&mel), frame_size));
        }
        let pq_us = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        println!("\n=== BIGVGAN FRAME SYNTHESIS BENCHMARK (80 Mels x 2048 Samples) ===");
        println!("  Float32 Transcendental (163,840 sin/frame): {:>9.2} us", float_us);
        println!(
            "  p/q Z_{{2^32}} Phase-Ring + BFP-i8 Mels:    {:>9.2} us ({:.2}x speedup)",
            pq_us,
            float_us / pq_us
        );
        assert!(pq_us < float_us);
    }
}
