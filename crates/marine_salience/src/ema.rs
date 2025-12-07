//! Exponential Moving Average (EMA) for smooth tracking
//!
//! EMA smooths noisy measurements while maintaining responsiveness.
//! Used to track period and amplitude patterns in Marine algorithm.

#![cfg_attr(not(feature = "std"), no_std)]

/// Exponential Moving Average tracker
///
/// EMA formula: value = alpha * new + (1 - alpha) * old
/// - Higher alpha = faster response, more noise
/// - Lower alpha = slower response, smoother
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct Ema {
    /// Smoothing factor (0..1)
    alpha: f32,
    /// Current smoothed value
    value: f32,
    /// Whether we've received at least one sample
    initialized: bool,
}

impl Ema {
    /// Create new EMA with given smoothing factor
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (0..1). Higher = faster adaptation.
    ///
    /// # Example
    /// ```
    /// use marine_salience::ema::Ema;
    /// let mut ema = Ema::new(0.1); // 10% new, 90% old
    /// ema.update(100.0);
    /// assert_eq!(ema.get(), 100.0); // First value becomes baseline
    /// ema.update(200.0);
    /// assert!((ema.get() - 110.0).abs() < 0.01); // 0.1*200 + 0.9*100
    /// ```
    pub const fn new(alpha: f32) -> Self {
        Self {
            alpha,
            value: 0.0,
            initialized: false,
        }
    }

    /// Update EMA with new measurement
    pub fn update(&mut self, x: f32) {
        if !self.initialized {
            // First value becomes the baseline
            self.value = x;
            self.initialized = true;
        } else {
            // EMA update: new = alpha * x + (1 - alpha) * old
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
    }

    /// Get current smoothed value
    pub fn get(&self) -> f32 {
        self.value
    }

    /// Check if EMA has been initialized (received at least one sample)
    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// Reset EMA to uninitialized state
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }

    /// Get the smoothing factor
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set a new smoothing factor
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_value_becomes_baseline() {
        let mut ema = Ema::new(0.1);
        assert!(!ema.is_ready());
        ema.update(42.0);
        assert!(ema.is_ready());
        assert_eq!(ema.get(), 42.0);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut ema = Ema::new(0.1);
        ema.update(100.0);
        ema.update(200.0);
        // 0.1 * 200 + 0.9 * 100 = 20 + 90 = 110
        assert!((ema.get() - 110.0).abs() < 0.001);
    }

    #[test]
    fn test_high_alpha_fast_response() {
        let mut ema = Ema::new(0.9);
        ema.update(100.0);
        ema.update(200.0);
        // 0.9 * 200 + 0.1 * 100 = 180 + 10 = 190
        assert!((ema.get() - 190.0).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut ema = Ema::new(0.1);
        ema.update(100.0);
        assert!(ema.is_ready());
        ema.reset();
        assert!(!ema.is_ready());
        assert_eq!(ema.get(), 0.0);
    }
}
