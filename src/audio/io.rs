//! Audio I/O operations

use crate::{Error, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

/// Audio data container
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Audio samples (mono, normalized to [-1, 1])
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl AudioData {
    /// Create new audio data
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Load audio from WAV file
///
/// # Arguments
/// * `path` - Path to WAV file
/// * `target_sr` - Optional target sample rate (will resample if different)
///
/// # Returns
/// Audio data with samples normalized to [-1, 1]
pub fn load_audio<P: AsRef<Path>>(path: P, target_sr: Option<u32>) -> Result<AudioData> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(Error::FileNotFound(path.display().to_string()));
    }

    let reader = WavReader::open(path).map_err(|e| Error::Audio(format!("Failed to open WAV: {}", e)))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    // Read samples based on format
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::Audio(format!("Failed to read samples: {}", e)))?;
            samples
        }
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let samples: Vec<i32> = reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::Audio(format!("Failed to read samples: {}", e)))?;

            // Normalize to [-1, 1]
            let max_val = (1 << (bits - 1)) as f32;
            samples.iter().map(|&s| s as f32 / max_val).collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    let mut audio = AudioData::new(mono_samples, sample_rate);

    // Resample if needed
    if let Some(target) = target_sr {
        if target != sample_rate {
            audio = super::resample::resample(&audio, target)?;
        }
    }

    Ok(audio)
}

/// Save audio to WAV file
///
/// # Arguments
/// * `path` - Output path
/// * `audio` - Audio data to save
pub fn save_audio<P: AsRef<Path>>(path: P, audio: &AudioData) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| Error::Audio(format!("Failed to create WAV writer: {}", e)))?;

    for &sample in &audio.samples {
        writer
            .write_sample(sample)
            .map_err(|e| Error::Audio(format!("Failed to write sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| Error::Audio(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Save audio samples with specified sample rate
pub fn save_samples<P: AsRef<Path>>(path: P, samples: &[f32], sample_rate: u32) -> Result<()> {
    let audio = AudioData::new(samples.to_vec(), sample_rate);
    save_audio(path, &audio)
}

/// Load multiple audio files in parallel
pub fn load_audio_batch<P: AsRef<Path> + Sync>(
    paths: &[P],
    target_sr: Option<u32>,
) -> Result<Vec<AudioData>> {
    use rayon::prelude::*;

    paths
        .par_iter()
        .map(|p| load_audio(p, target_sr))
        .collect()
}
