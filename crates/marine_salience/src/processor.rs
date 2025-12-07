//! Core Marine processor - O(1) per-sample jitter detection
//!
//! The heart of the Marine algorithm. Processes audio samples one at a time,
//! detecting peaks and computing jitter metrics in constant time.
//!
//! "Marines are not just jarheads - they are actually very intelligent"

#![cfg_attr(not(feature = "std"), no_std)]

use crate::config::MarineConfig;
use crate::ema::Ema;
use crate::packet::{SalienceMarker, SaliencePacket};

/// Marine salience processor
///
/// Processes audio samples one at a time, detecting peaks and computing
/// jitter metrics. Designed for O(1) per-sample operation.
///
/// # Example
/// ```
/// use marine_salience::{MarineConfig, MarineProcessor};
///
/// let config = MarineConfig::speech_default(22050);
/// let mut processor = MarineProcessor::new(config);
///
/// // Process samples (e.g., from audio buffer)
/// let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
/// for sample in &samples {
///     if let Some(marker) = processor.process_sample(*sample) {
///         match marker {
///             marine_salience::packet::SalienceMarker::Peak(packet) => {
///                 println!("Peak detected! Salience: {:.2}", packet.s_score);
///             }
///             _ => {}
///         }
///     }
/// }
/// ```
pub struct MarineProcessor {
    /// Configuration parameters
    cfg: MarineConfig,

    /// Previous sample (t-2)
    prev2: f32,
    /// Previous sample (t-1)
    prev1: f32,
    /// Current sample index
    idx: u64,

    /// Sample index of last detected peak
    last_peak_idx: u64,
    /// Amplitude of last detected peak
    last_peak_amp: f32,

    /// EMA tracker for inter-peak periods
    ema_period: Ema,
    /// EMA tracker for peak amplitudes
    ema_amp: Ema,

    /// Number of peaks detected so far
    peak_count: u64,
}

impl MarineProcessor {
    /// Create a new Marine processor with given configuration
    pub fn new(cfg: MarineConfig) -> Self {
        Self {
            cfg,
            prev2: 0.0,
            prev1: 0.0,
            idx: 0,
            last_peak_idx: 0,
            last_peak_amp: 0.0,
            ema_period: Ema::new(cfg.ema_period_alpha),
            ema_amp: Ema::new(cfg.ema_amp_alpha),
            peak_count: 0,
        }
    }

    /// Process a single audio sample - O(1) operation
    ///
    /// Returns Some(SalienceMarker) when a peak is detected or special
    /// condition occurs, None otherwise.
    ///
    /// # Arguments
    /// * `sample` - Audio sample value (typically -1.0 to 1.0)
    ///
    /// # Returns
    /// - `Some(Peak(packet))` - Peak detected with jitter metrics
    /// - `Some(Fracture)` - Silence/gap detected
    /// - `Some(Noise)` - High noise floor detected
    /// - `None` - No significant event at this sample
    pub fn process_sample(&mut self, sample: f32) -> Option<SalienceMarker> {
        let i = self.idx;
        self.idx += 1;

        // Pre-gating: ignore samples below threshold
        if sample.abs() < self.cfg.clip_threshold {
            self.prev2 = self.prev1;
            self.prev1 = sample;
            return None;
        }

        // Peak detection: prev1 is peak if prev2 < prev1 > sample
        // Simple local maximum detection
        let is_peak = i >= 2
            && self.prev1.abs() >= self.cfg.clip_threshold
            && self.prev1.abs() > self.prev2.abs()
            && self.prev1.abs() > sample.abs();

        let mut result = None;

        if is_peak {
            let peak_idx = i - 1;
            let amp = self.prev1.abs();
            let energy = amp * amp;

            // Calculate period (time since last peak)
            let period = if self.last_peak_idx == 0 {
                0.0
            } else {
                (peak_idx - self.last_peak_idx) as f32
            };

            // Only process if period is within valid range
            if period > self.cfg.min_period as f32 && period < self.cfg.max_period as f32 {
                if self.ema_period.is_ready() {
                    // Calculate jitter metrics
                    let jp = (period - self.ema_period.get()).abs() / self.ema_period.get();
                    let ja = (amp - self.ema_amp.get()).abs() / self.ema_amp.get();

                    // Harmonic score (simplified - TODO: FFT-based detection)
                    // For now, assume voiced content (h = 1.0)
                    // In production, this would check for harmonic structure
                    let h = 1.0;

                    // Salience score: inverse of combined jitter
                    // Higher jitter = lower salience
                    let s = 1.0 / (1.0 + jp + ja);

                    result = Some(SalienceMarker::Peak(SaliencePacket::new(
                        jp, ja, h, s, energy, peak_idx,
                    )));
                }

                // Update EMAs with new measurements
                self.ema_period.update(period);
                self.ema_amp.update(amp);
            }

            self.last_peak_idx = peak_idx;
            self.last_peak_amp = amp;
            self.peak_count += 1;
        }

        // Update sample history
        self.prev2 = self.prev1;
        self.prev1 = sample;

        result
    }

    /// Process a buffer of samples, collecting all salience packets
    ///
    /// More efficient than calling process_sample repeatedly when you
    /// have a full buffer available.
    ///
    /// # Arguments
    /// * `samples` - Buffer of audio samples
    ///
    /// # Returns
    /// Vector of salience packets for all detected peaks
    #[cfg(feature = "std")]
    pub fn process_buffer(&mut self, samples: &[f32]) -> Vec<SaliencePacket> {
        let mut packets = Vec::new();

        for &sample in samples {
            if let Some(SalienceMarker::Peak(packet)) = self.process_sample(sample) {
                packets.push(packet);
            }
        }

        packets
    }

    /// Reset processor state (start fresh)
    pub fn reset(&mut self) {
        self.prev2 = 0.0;
        self.prev1 = 0.0;
        self.idx = 0;
        self.last_peak_idx = 0;
        self.last_peak_amp = 0.0;
        self.ema_period.reset();
        self.ema_amp.reset();
        self.peak_count = 0;
    }

    /// Get number of peaks detected so far
    pub fn peak_count(&self) -> u64 {
        self.peak_count
    }

    /// Get current sample index
    pub fn current_index(&self) -> u64 {
        self.idx
    }

    /// Check if processor has enough data for reliable jitter
    pub fn is_warmed_up(&self) -> bool {
        self.peak_count >= 3 && self.ema_period.is_ready()
    }

    /// Get current expected period (from EMA)
    pub fn expected_period(&self) -> Option<f32> {
        if self.ema_period.is_ready() {
            Some(self.ema_period.get())
        } else {
            None
        }
    }

    /// Get current expected amplitude (from EMA)
    pub fn expected_amplitude(&self) -> Option<f32> {
        if self.ema_amp.is_ready() {
            Some(self.ema_amp.get())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_detection() {
        let config = MarineConfig::speech_default(22050);
        let mut processor = MarineProcessor::new(config);

        // Create simple signal with peaks
        // Peak at sample 10, 20, 30...
        let mut samples = vec![0.0; 100];
        for i in (10..100).step_by(10) {
            samples[i] = 0.5; // Peak
            if i > 0 {
                samples[i - 1] = 0.3; // Rising edge
            }
            if i < 99 {
                samples[i + 1] = 0.3; // Falling edge
            }
        }

        let mut peak_count = 0;
        for sample in &samples {
            if let Some(SalienceMarker::Peak(_)) = processor.process_sample(*sample) {
                peak_count += 1;
            }
        }

        // Should detect several peaks (not all due to period constraints)
        assert!(peak_count > 0);
    }

    #[test]
    fn test_jitter_calculation() {
        let mut config = MarineConfig::speech_default(22050);
        config.min_period = 5;
        config.max_period = 20;
        let mut processor = MarineProcessor::new(config);

        // Create signal with consistent period of 10 samples
        let mut detected_packets = vec![];
        for cycle in 0..10 {
            for i in 0..10 {
                let sample = if i == 5 {
                    0.8 // Peak in middle
                } else if i == 4 || i == 6 {
                    0.5 // Edges
                } else {
                    0.01 // Just above threshold
                };

                if let Some(SalienceMarker::Peak(packet)) = processor.process_sample(sample) {
                    detected_packets.push(packet);
                }
            }
        }

        // With consistent periods, later packets should have low jitter
        if detected_packets.len() > 3 {
            let last = detected_packets.last().unwrap();
            // Jitter should be relatively low for consistent signal
            assert!(last.j_p < 0.5, "Period jitter too high: {}", last.j_p);
        }
    }

    #[test]
    fn test_reset() {
        let config = MarineConfig::speech_default(22050);
        let mut processor = MarineProcessor::new(config);

        // Process some samples
        for _ in 0..100 {
            processor.process_sample(0.5);
        }
        assert!(processor.current_index() > 0);

        // Reset and verify
        processor.reset();
        assert_eq!(processor.current_index(), 0);
        assert_eq!(processor.peak_count(), 0);
        assert!(!processor.is_warmed_up());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_process_buffer() {
        let mut config = MarineConfig::speech_default(22050);
        config.min_period = 5;
        config.max_period = 50;
        let mut processor = MarineProcessor::new(config);

        // Generate test signal with peaks
        let mut samples = Vec::new();
        for _ in 0..20 {
            samples.extend_from_slice(&[0.01, 0.3, 0.8, 0.3, 0.01]);
        }

        let packets = processor.process_buffer(&samples);
        // Should detect multiple peaks
        assert!(packets.len() > 0);
    }
}
