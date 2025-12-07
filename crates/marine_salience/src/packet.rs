//! Salience packet - the output of Marine analysis
//!
//! Contains jitter measurements and quality scores for a detected peak.

#![cfg_attr(not(feature = "std"), no_std)]

/// Salience packet emitted on peak detection
///
/// Contains all the jitter and quality metrics for a single audio event.
/// These packets can be aggregated to form prosody vectors or quality scores.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct SaliencePacket {
    /// Period jitter - timing instability between peaks
    /// Lower = more stable/musical, Higher = more chaotic
    /// Range: 0.0+ (normalized difference from expected period)
    pub j_p: f32,

    /// Amplitude jitter - loudness instability
    /// Lower = consistent volume, Higher = erratic dynamics
    /// Range: 0.0+ (normalized difference from expected amplitude)
    pub j_a: f32,

    /// Harmonic alignment score
    /// 1.0 = perfectly voiced/harmonic, 0.0 = noise
    /// For now this is simplified; can be enhanced with FFT
    pub h_score: f32,

    /// Overall salience score (authenticity)
    /// 1.0 = perfect quality, 0.0 = heavily damaged
    /// Computed from inverse of combined jitter
    pub s_score: f32,

    /// Local peak energy (amplitude squared)
    /// Represents loudness at this event
    pub energy: f32,

    /// Sample index where this peak occurred
    /// Useful for temporal analysis
    pub sample_index: u64,
}

impl SaliencePacket {
    /// Create a new salience packet
    pub fn new(
        j_p: f32,
        j_a: f32,
        h_score: f32,
        s_score: f32,
        energy: f32,
        sample_index: u64,
    ) -> Self {
        Self {
            j_p,
            j_a,
            h_score,
            s_score,
            energy,
            sample_index,
        }
    }

    /// Get combined jitter metric
    /// Average of period and amplitude jitter
    pub fn combined_jitter(&self) -> f32 {
        (self.j_p + self.j_a) / 2.0
    }

    /// Check if this represents high-quality audio
    /// (low jitter, high salience)
    pub fn is_high_quality(&self, threshold: f32) -> bool {
        self.s_score >= threshold
    }

    /// Check if this indicates damaged/synthetic audio
    pub fn is_damaged(&self, jitter_threshold: f32) -> bool {
        self.combined_jitter() > jitter_threshold
    }
}

/// Special salience markers for non-peak events
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub enum SalienceMarker {
    /// Normal peak detected
    Peak(SaliencePacket),
    /// Fracture/gap detected (silence)
    Fracture,
    /// High noise floor detected
    Noise,
    /// Insufficient data for analysis
    Insufficient,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_jitter() {
        let packet = SaliencePacket::new(0.1, 0.3, 1.0, 0.8, 0.5, 0);
        assert!((packet.combined_jitter() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_is_high_quality() {
        let good = SaliencePacket::new(0.01, 0.02, 1.0, 0.95, 0.5, 0);
        let bad = SaliencePacket::new(0.5, 0.6, 0.5, 0.3, 0.5, 0);

        assert!(good.is_high_quality(0.8));
        assert!(!bad.is_high_quality(0.8));
    }

    #[test]
    fn test_is_damaged() {
        let good = SaliencePacket::new(0.01, 0.02, 1.0, 0.95, 0.5, 0);
        let bad = SaliencePacket::new(0.5, 0.6, 0.5, 0.3, 0.5, 0);

        assert!(!good.is_damaged(0.3));
        assert!(bad.is_damaged(0.3));
    }
}
