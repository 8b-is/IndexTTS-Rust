//! Marine algorithm configuration
//!
//! Tunable parameters for jitter detection. These have been calibrated
//! for speech/audio processing but can be adjusted for specific use cases.

#![cfg_attr(not(feature = "std"), no_std)]

/// Configuration for Marine salience detection
///
/// These parameters control sensitivity and behavior of the jitter detector.
/// The defaults are tuned for speech processing at common sample rates.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct MarineConfig {
    /// Minimum amplitude to consider a sample (gating threshold)
    /// Samples below this are ignored as noise
    /// Default: 1e-3 (~-60dB)
    pub clip_threshold: f32,

    /// EMA smoothing factor for period tracking (0..1)
    /// Lower = smoother, slower adaptation
    /// Default: 0.01
    pub ema_period_alpha: f32,

    /// EMA smoothing factor for amplitude tracking (0..1)
    /// Default: 0.01
    pub ema_amp_alpha: f32,

    /// Minimum inter-peak period in samples
    /// Rejects peaks closer than this (filters high-frequency noise)
    /// Default: sample_rate / 4000 (~4kHz upper F0)
    pub min_period: u32,

    /// Maximum inter-peak period in samples
    /// Rejects peaks farther than this (filters very low frequencies)
    /// Default: sample_rate / 60 (~60Hz lower F0)
    pub max_period: u32,

    /// Threshold below which jitter is considered "low" (stable)
    /// Default: 0.02
    pub jitter_low: f32,

    /// Threshold above which jitter is considered "high" (unstable)
    /// Default: 0.60
    pub jitter_high: f32,
}

impl MarineConfig {
    /// Create config optimized for speech at given sample rate
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 22050, 44100)
    ///
    /// # Example
    /// ```
    /// use marine_salience::MarineConfig;
    /// let config = MarineConfig::speech_default(22050);
    /// assert!(config.min_period < config.max_period);
    /// ```
    pub const fn speech_default(sample_rate: u32) -> Self {
        // F0 range: ~60Hz (low male) to ~4kHz (includes harmonics)
        let min_period = sample_rate / 4000; // Upper bound
        let max_period = sample_rate / 60;   // Lower bound

        Self {
            clip_threshold: 1e-3,
            ema_period_alpha: 0.01,
            ema_amp_alpha: 0.01,
            min_period,
            max_period,
            jitter_low: 0.02,
            jitter_high: 0.60,
        }
    }

    /// Create config for high-sensitivity detection
    /// More peaks detected, faster adaptation
    pub const fn high_sensitivity(sample_rate: u32) -> Self {
        let min_period = sample_rate / 8000;
        let max_period = sample_rate / 40;

        Self {
            clip_threshold: 5e-4,
            ema_period_alpha: 0.05,
            ema_amp_alpha: 0.05,
            min_period,
            max_period,
            jitter_low: 0.01,
            jitter_high: 0.50,
        }
    }

    /// Create config for TTS output validation
    /// Tuned to detect synthetic artifacts
    pub const fn tts_validation(sample_rate: u32) -> Self {
        let min_period = sample_rate / 4000;
        let max_period = sample_rate / 80;

        Self {
            clip_threshold: 1e-3,
            ema_period_alpha: 0.02,
            ema_amp_alpha: 0.02,
            min_period,
            max_period,
            jitter_low: 0.015, // Stricter for synthetic speech
            jitter_high: 0.40, // More sensitive to artifacts
        }
    }
}

impl Default for MarineConfig {
    fn default() -> Self {
        // Default to 22050 Hz (common TTS sample rate)
        Self::speech_default(22050)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_default_periods() {
        let config = MarineConfig::speech_default(22050);
        assert!(config.min_period < config.max_period);
        assert_eq!(config.min_period, 22050 / 4000); // 5 samples
        assert_eq!(config.max_period, 22050 / 60);   // 367 samples
    }

    #[test]
    fn test_different_sample_rates() {
        let config_22k = MarineConfig::speech_default(22050);
        let config_44k = MarineConfig::speech_default(44100);
        let config_48k = MarineConfig::speech_default(48000);

        // Higher sample rates = more samples per period
        assert!(config_44k.max_period > config_22k.max_period);
        assert!(config_48k.max_period > config_44k.max_period);
    }
}
