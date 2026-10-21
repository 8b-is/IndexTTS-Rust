//! Marine Prosody Extraction and Conditioning (`p/q` Integer Space)
//!
//! Extracts an 8-dimensional interpretable prosody vector (`MarineProsodyVector`)
//! from audio wave packets using Marine Salience zero-crossing period and amplitude
//! jitter statistics in `p/q` (`q = 65536`) integer arithmetic.

/// 8-dimensional interpretable prosody vector derived from Marine Salience packets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarineProsodyVector {
    /// Mean period jitter (`jp`)
    pub jp_mean: f32,
    /// Standard deviation of period jitter
    pub jp_std: f32,
    /// Mean amplitude jitter (`ja`)
    pub ja_mean: f32,
    /// Standard deviation of amplitude jitter
    pub ja_std: f32,
    /// Mean harmonicity (`h`)
    pub h_mean: f32,
    /// Mean salience/authenticity score (`s`)
    pub s_mean: f32,
    /// Peak packet density (packets per second)
    pub peak_density: f32,
    /// Mean normalized packet energy
    pub energy_mean: f32,
}

impl MarineProsodyVector {
    /// Zero-initialized prosody vector
    pub const fn zeros() -> Self {
        Self {
            jp_mean: 0.0,
            jp_std: 0.0,
            ja_mean: 0.0,
            ja_std: 0.0,
            h_mean: 0.0,
            s_mean: 0.0,
            peak_density: 0.0,
            energy_mean: 0.0,
        }
    }

    /// Combined timing and amplitude jitter (`0.5 * (jp_mean + ja_mean)`)
    #[inline]
    pub fn combined_jitter(&self) -> f32 {
        0.5 * (self.jp_mean + self.ja_mean)
    }

    /// Convert to an 8-element array `[f32; 8]`
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
}

impl Default for MarineProsodyVector {
    fn default() -> Self {
        Self::zeros()
    }
}

/// Conditioner that extracts `MarineProsodyVector` from raw audio samples in `p/q` integer space.
pub struct MarineProsodyConditioner {
    sample_rate: u32,
}

impl MarineProsodyConditioner {
    /// Create a new prosody conditioner for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Extract prosody vector in `p/q` (`q = 65536`) integer space.
    pub fn extract(&self, audio: &[f32]) -> MarineProsodyVector {
        const Q: i64 = 65_536;
        if audio.len() < 32 {
            return MarineProsodyVector::zeros();
        }

        let mut periods = Vec::new();
        let mut peaks = Vec::new();
        let mut last_zc: Option<usize> = None;
        let mut cur_peak: i64 = 0;
        let mut prev_pq = (audio[0] * (Q as f32)) as i64;

        for (i, &s) in audio.iter().enumerate().skip(1) {
            let cur_pq = (s * (Q as f32)) as i64;
            let abs_pq = cur_pq.abs();
            if abs_pq > cur_peak {
                cur_peak = abs_pq;
            }
            if prev_pq <= 0 && cur_pq > 0 && cur_peak > 64 {
                if let Some(prev_idx) = last_zc {
                    let dt = i - prev_idx;
                    if dt >= 8 {
                        periods.push(dt as i64);
                        peaks.push(cur_peak);
                    }
                }
                last_zc = Some(i);
                cur_peak = 0;
            }
            prev_pq = cur_pq;
        }

        let n = periods.len();
        if n < 2 {
            return MarineProsodyVector::zeros();
        }

        let mean_p = (periods.iter().sum::<i64>() as f32) / (n as f32);
        let mean_a = (peaks.iter().sum::<i64>() as f32) / (n as f32);

        let mut jp_vals = Vec::with_capacity(n - 1);
        let mut ja_vals = Vec::with_capacity(n - 1);
        for w in 0..(n - 1) {
            jp_vals.push(((periods[w + 1] - periods[w]).abs() as f32) / mean_p.max(1.0));
            ja_vals.push(((peaks[w + 1] - peaks[w]).abs() as f32) / mean_a.max(1.0));
        }

        let m = jp_vals.len() as f32;
        let jp_mean = jp_vals.iter().sum::<f32>() / m;
        let ja_mean = ja_vals.iter().sum::<f32>() / m;
        let jp_std = (jp_vals.iter().map(|v| (v - jp_mean).powi(2)).sum::<f32>() / m).sqrt();
        let ja_std = (ja_vals.iter().map(|v| (v - ja_mean).powi(2)).sum::<f32>() / m).sqrt();
        let s_mean = (1.0 / (1.0 + jp_mean + ja_mean)).clamp(0.0, 1.0);
        let h_mean = (1.0 / (1.0 + jp_mean)).clamp(0.0, 1.0);
        let duration_sec = (audio.len() as f32) / (self.sample_rate.max(1) as f32);

        MarineProsodyVector {
            jp_mean,
            jp_std,
            ja_mean,
            ja_std,
            h_mean,
            s_mean,
            peak_density: if duration_sec > 0.0 {
                (n as f32) / duration_sec
            } else {
                0.0
            },
            energy_mean: (mean_a / (Q as f32)).clamp(0.0, 1.0),
        }
    }
}
