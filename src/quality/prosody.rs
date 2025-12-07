//! Marine Prosody Conditioner - Extract 8D interpretable emotion vectors
//!
//! Uses Marine salience to extract prosodic features from reference audio.
//! These features are interpretable and can be directly edited for control.
//!
//! The 8D vector captures:
//! 1. Period jitter (mean & std) - pitch stability
//! 2. Amplitude jitter (mean & std) - roughness/strain
//! 3. Harmonic alignment - voiced vs noisy
//! 4. Overall salience - authenticity score
//! 5. Peak density - speech rate/intensity
//! 6. Energy - loudness

use crate::error::{Error, Result};

/// 8-dimensional prosody vector extracted from audio
///
/// These features capture the "emotional signature" of speech:
/// - Low jitter + high energy = confident/happy
/// - High jitter + low energy = nervous/uneasy
/// - Stable patterns = calm, unstable = agitated
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarineProsodyVector {
    /// Mean period jitter (pitch stability)
    /// Lower = more stable pitch, Higher = more variation
    pub jp_mean: f32,

    /// Standard deviation of period jitter
    /// Captures consistency of pitch patterns
    pub jp_std: f32,

    /// Mean amplitude jitter (volume stability)
    /// Lower = consistent volume, Higher = erratic
    pub ja_mean: f32,

    /// Standard deviation of amplitude jitter
    /// Captures volume pattern consistency
    pub ja_std: f32,

    /// Mean harmonic alignment score
    /// 1.0 = perfectly voiced, 0.0 = noise
    pub h_mean: f32,

    /// Mean overall salience score
    /// Overall authenticity/quality rating
    pub s_mean: f32,

    /// Peak density (peaks per second)
    /// Related to speech rate and intensity
    pub peak_density: f32,

    /// Mean energy level
    /// Average loudness of detected peaks
    pub energy_mean: f32,
}

impl MarineProsodyVector {
    /// Create zero vector (baseline)
    pub fn zeros() -> Self {
        Self {
            jp_mean: 0.0,
            jp_std: 0.0,
            ja_mean: 0.0,
            ja_std: 0.0,
            h_mean: 1.0,
            s_mean: 1.0,
            peak_density: 0.0,
            energy_mean: 0.0,
        }
    }

    /// Convert to f32 array for neural network input
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.jp_mean,
            self.jp_std,
            self.ja_mean,
            self.ja_std,
            self.h_mean,
            self.s_mean,
            self.peak_density,
            self.energy_mean,
        ]
    }

    /// Create from f32 array
    pub fn from_array(arr: [f32; 8]) -> Self {
        Self {
            jp_mean: arr[0],
            jp_std: arr[1],
            ja_mean: arr[2],
            ja_std: arr[3],
            h_mean: arr[4],
            s_mean: arr[5],
            peak_density: arr[6],
            energy_mean: arr[7],
        }
    }

    /// Get combined jitter (average of period and amplitude)
    pub fn combined_jitter(&self) -> f32 {
        (self.jp_mean + self.ja_mean) / 2.0
    }

    /// Estimate emotional valence from prosody
    /// Returns value from -1.0 (negative) to 1.0 (positive)
    pub fn estimate_valence(&self) -> f32 {
        // High energy + low jitter = positive
        // Low energy + high jitter = negative
        let jitter_factor = 1.0 / (1.0 + self.combined_jitter());
        let energy_factor = self.energy_mean.sqrt();

        // Combine factors, normalize to -1..1 range
        (jitter_factor * energy_factor * 2.0 - 1.0).clamp(-1.0, 1.0)
    }

    /// Estimate arousal/intensity level
    /// Returns value from 0.0 (calm) to 1.0 (excited)
    pub fn estimate_arousal(&self) -> f32 {
        // High peak density + high energy + some jitter variance = high arousal
        let density_factor = (self.peak_density / 100.0).clamp(0.0, 1.0);
        let energy_factor = self.energy_mean.sqrt();
        let variance_factor = (self.jp_std + self.ja_std).clamp(0.0, 1.0);

        ((density_factor + energy_factor + variance_factor) / 3.0).clamp(0.0, 1.0)
    }
}

impl Default for MarineProsodyVector {
    fn default() -> Self {
        Self::zeros()
    }
}

/// Marine-based prosody conditioner for TTS
///
/// Replaces heavy Conformer-style extractors with lightweight, interpretable
/// Marine salience features. This gives you:
/// - 8D interpretable emotion vector
/// - Direct editability for control
/// - Biologically plausible processing
/// - O(n) linear time extraction
pub struct MarineProsodyConditioner {
    sample_rate: u32,
    jitter_low: f32,
    jitter_high: f32,
    min_period: u32,
    max_period: u32,
    ema_alpha: f32,
}

impl MarineProsodyConditioner {
    /// Create new prosody conditioner for given sample rate
    pub fn new(sample_rate: u32) -> Self {
        // F0 range: ~60Hz (low male) to ~4kHz (includes harmonics)
        let min_period = sample_rate / 4000;
        let max_period = sample_rate / 60;

        Self {
            sample_rate,
            jitter_low: 0.02,
            jitter_high: 0.60,
            min_period,
            max_period,
            ema_alpha: 0.01,
        }
    }

    /// Extract prosody vector from audio samples
    ///
    /// Analyzes the audio to produce an 8D prosody vector capturing
    /// the emotional/stylistic characteristics of the speech.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (typically -1.0 to 1.0 range)
    ///
    /// # Returns
    /// * `Ok(MarineProsodyVector)` - Extracted prosody features
    /// * `Err` - If insufficient peaks detected
    pub fn from_samples(&self, samples: &[f32]) -> Result<MarineProsodyVector> {
        if samples.is_empty() {
            return Err(Error::Audio("Empty audio buffer".into()));
        }

        // Detect peaks and collect jitter measurements
        let mut peaks: Vec<PeakInfo> = Vec::new();
        let clip_threshold = 1e-3;

        // Simple peak detection
        for i in 1..samples.len().saturating_sub(1) {
            let prev = samples[i - 1].abs();
            let curr = samples[i].abs();
            let next = samples[i + 1].abs();

            if curr > prev && curr > next && curr > clip_threshold {
                peaks.push(PeakInfo {
                    index: i,
                    amplitude: curr,
                });
            }
        }

        if peaks.len() < 3 {
            // Not enough peaks for meaningful analysis
            return Ok(MarineProsodyVector::zeros());
        }

        // Calculate inter-peak periods and jitter
        let mut periods: Vec<f32> = Vec::new();
        let mut amplitudes: Vec<f32> = Vec::new();
        let mut jp_values: Vec<f32> = Vec::new();
        let mut ja_values: Vec<f32> = Vec::new();

        // Use EMA for tracking
        let mut ema_period = 0.0f32;
        let mut ema_amp = 0.0f32;
        let mut ema_initialized = false;

        for i in 1..peaks.len() {
            let period = (peaks[i].index - peaks[i - 1].index) as f32;
            let amp = peaks[i].amplitude;

            // Check if period is in valid range
            if period > self.min_period as f32 && period < self.max_period as f32 {
                periods.push(period);
                amplitudes.push(amp);

                if !ema_initialized {
                    ema_period = period;
                    ema_amp = amp;
                    ema_initialized = true;
                } else {
                    // Calculate jitter
                    let jp = (period - ema_period).abs() / ema_period;
                    let ja = (amp - ema_amp).abs() / ema_amp;
                    jp_values.push(jp);
                    ja_values.push(ja);

                    // Update EMA
                    ema_period = self.ema_alpha * period + (1.0 - self.ema_alpha) * ema_period;
                    ema_amp = self.ema_alpha * amp + (1.0 - self.ema_alpha) * ema_amp;
                }
            }
        }

        if jp_values.is_empty() {
            return Ok(MarineProsodyVector::zeros());
        }

        // Compute statistics
        let n = jp_values.len() as f32;
        let duration_sec = samples.len() as f32 / self.sample_rate as f32;

        // Mean calculations
        let jp_mean = jp_values.iter().sum::<f32>() / n;
        let ja_mean = ja_values.iter().sum::<f32>() / n;
        let energy_mean = amplitudes.iter().map(|a| a * a).sum::<f32>() / amplitudes.len() as f32;

        // Std calculations
        let jp_var = jp_values.iter().map(|x| (x - jp_mean).powi(2)).sum::<f32>() / n;
        let ja_var = ja_values.iter().map(|x| (x - ja_mean).powi(2)).sum::<f32>() / n;
        let jp_std = jp_var.sqrt();
        let ja_std = ja_var.sqrt();

        // Harmonic score (simplified - assume voiced content)
        let h_mean = 1.0;

        // Overall salience score
        let s_mean = 1.0 / (1.0 + jp_mean + ja_mean);

        // Peak density
        let peak_density = peaks.len() as f32 / duration_sec;

        Ok(MarineProsodyVector {
            jp_mean,
            jp_std,
            ja_mean,
            ja_std,
            h_mean,
            s_mean,
            peak_density,
            energy_mean,
        })
    }

    /// Validate TTS output quality using Marine salience
    ///
    /// Returns quality score and potential issues detected
    pub fn validate_tts_output(&self, samples: &[f32]) -> Result<TTSQualityReport> {
        let prosody = self.from_samples(samples)?;

        let mut issues = Vec::new();

        // Check for common TTS problems
        if prosody.jp_mean < 0.005 {
            issues.push("Too perfect - sounds robotic (add natural variation)");
        }

        if prosody.jp_mean > 0.3 {
            issues.push("High period jitter - possible artifacts");
        }

        if prosody.ja_mean > 0.4 {
            issues.push("High amplitude jitter - volume inconsistency");
        }

        if prosody.s_mean < 0.4 {
            issues.push("Low salience - audio quality issues");
        }

        if prosody.peak_density < 10.0 {
            issues.push("Low peak density - missing speech energy");
        }

        let quality_score = prosody.s_mean * 100.0;

        Ok(TTSQualityReport {
            prosody,
            quality_score,
            issues,
        })
    }

    /// Get the configured sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Internal peak information
struct PeakInfo {
    index: usize,
    amplitude: f32,
}

/// TTS quality validation report
#[derive(Debug, Clone)]
pub struct TTSQualityReport {
    /// Extracted prosody vector
    pub prosody: MarineProsodyVector,
    /// Overall quality score (0-100)
    pub quality_score: f32,
    /// List of detected issues
    pub issues: Vec<&'static str>,
}

impl TTSQualityReport {
    /// Check if quality passes threshold
    pub fn passes(&self, threshold: f32) -> bool {
        self.quality_score >= threshold && self.issues.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prosody_vector_array_conversion() {
        let vec = MarineProsodyVector {
            jp_mean: 0.1,
            jp_std: 0.05,
            ja_mean: 0.2,
            ja_std: 0.1,
            h_mean: 0.9,
            s_mean: 0.8,
            peak_density: 50.0,
            energy_mean: 0.3,
        };

        let arr = vec.to_array();
        let reconstructed = MarineProsodyVector::from_array(arr);

        assert_eq!(vec.jp_mean, reconstructed.jp_mean);
        assert_eq!(vec.s_mean, reconstructed.s_mean);
    }

    #[test]
    fn test_conditioner_empty_buffer() {
        let conditioner = MarineProsodyConditioner::new(22050);
        let result = conditioner.from_samples(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_conditioner_silence() {
        let conditioner = MarineProsodyConditioner::new(22050);
        let silence = vec![0.0; 1000];
        let prosody = conditioner.from_samples(&silence).unwrap();
        // Should return zeros for silence
        assert_eq!(prosody.peak_density, 0.0);
    }

    #[test]
    fn test_estimate_valence() {
        let positive = MarineProsodyVector {
            jp_mean: 0.01,
            jp_std: 0.01,
            ja_mean: 0.01,
            ja_std: 0.01,
            h_mean: 1.0,
            s_mean: 0.95,
            peak_density: 100.0,
            energy_mean: 0.8,
        };

        let negative = MarineProsodyVector {
            jp_mean: 0.5,
            jp_std: 0.3,
            ja_mean: 0.4,
            ja_std: 0.2,
            h_mean: 0.7,
            s_mean: 0.4,
            peak_density: 30.0,
            energy_mean: 0.1,
        };

        // Higher energy + lower jitter should give more positive valence
        assert!(positive.estimate_valence() > negative.estimate_valence());
    }
}
