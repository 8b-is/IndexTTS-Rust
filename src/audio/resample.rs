//! Audio resampling using rubato

use crate::{Error, Result};
use rubato::{
    FastFixedIn, PolynomialDegree, Resampler,
};

use super::AudioData;

/// Resample audio to target sample rate
///
/// Uses high-quality sinc interpolation
pub fn resample(audio: &AudioData, target_sr: u32) -> Result<AudioData> {
    if audio.sample_rate == target_sr {
        return Ok(audio.clone());
    }

    let resample_ratio = target_sr as f64 / audio.sample_rate as f64;

    // Create resampler
    let mut resampler = FastFixedIn::<f32>::new(
        resample_ratio,
        1.0, // max relative ratio (no variance)
        PolynomialDegree::Cubic,
        1024, // chunk size
        1,    // channels
    ).map_err(|e| Error::Audio(format!("Failed to create resampler: {}", e)))?;

    // Process in chunks
    let input_frames_needed = resampler.input_frames_next();
    let mut input_buffer = vec![vec![0.0f32; input_frames_needed]];
    let mut output_samples = Vec::new();

    let mut pos = 0;    
    while pos < audio.samples.len() {
        // Fill input buffer
        let end = (pos + input_frames_needed).min(audio.samples.len());
        let chunk_size = end - pos;

        input_buffer[0][..chunk_size].copy_from_slice(&audio.samples[pos..end]);

        // Pad with zeros if needed
        if chunk_size < input_frames_needed {
            input_buffer[0][chunk_size..].fill(0.0);
        }

        // Resample
        let output = resampler
            .process(&input_buffer, None)
            .map_err(|e| Error::Audio(format!("Resampling failed: {}", e)))?;

        output_samples.extend_from_slice(&output[0]);
        pos += chunk_size;

        if chunk_size < input_frames_needed {
            break;
        }
    }

    // Trim to expected length
    let expected_len = (audio.samples.len() as f64 * resample_ratio).ceil() as usize;
    output_samples.truncate(expected_len);

    Ok(AudioData::new(output_samples, target_sr))
}

/// Resample to 22050 Hz (common TTS sample rate)
pub fn resample_to_22k(audio: &AudioData) -> Result<AudioData> {
    resample(audio, 22050)
}

/// Resample to 16000 Hz (common for ASR)
pub fn resample_to_16k(audio: &AudioData) -> Result<AudioData> {
    resample(audio, 16000)
}
