//! Digital Signal Processing utilities


/// Apply pre-emphasis filter to audio signal
///
/// y[n] = x[n] - coef * x[n-1]
///
/// # Arguments
/// * `signal` - Input audio signal
/// * `coef` - Pre-emphasis coefficient (typically 0.97)
pub fn apply_preemphasis(signal: &[f32], coef: f32) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let mut output = Vec::with_capacity(signal.len());
    output.push(signal[0]);

    for i in 1..signal.len() {
        output.push(signal[i] - coef * signal[i - 1]);
    }

    output
}

/// Apply de-emphasis filter (inverse of pre-emphasis)
///
/// y[n] = x[n] + coef * y[n-1]
pub fn apply_deemphasis(signal: &[f32], coef: f32) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let mut output = Vec::with_capacity(signal.len());
    output.push(signal[0]);

    for i in 1..signal.len() {
        output.push(signal[i] + coef * output[i - 1]);
    }

    output
}

/// Normalize audio to [-1, 1] range
pub fn normalize_audio(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let max_abs = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    if max_abs < 1e-8 {
        return signal.to_vec();
    }

    signal.iter().map(|x| x / max_abs).collect()
}

/// Normalize audio to specific peak value
pub fn normalize_audio_peak(signal: &[f32], peak: f32) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let max_abs = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    if max_abs < 1e-8 {
        return signal.to_vec();
    }

    let scale = peak / max_abs;
    signal.iter().map(|x| x * scale).collect()
}

/// Dynamic range compression (log compression)
///
/// Used for mel spectrogram normalization
pub fn dynamic_range_compression(x: f32) -> f32 {
    let clip_val = 1e-5;
    (x.max(clip_val)).ln()
}

/// Dynamic range compression for array
pub fn dynamic_range_compression_array(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| dynamic_range_compression(v)).collect()
}

/// Dynamic range decompression (exp)
pub fn dynamic_range_decompression(x: f32) -> f32 {
    x.exp()
}

/// Dynamic range decompression for array
pub fn dynamic_range_decompression_array(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| dynamic_range_decompression(v)).collect()
}

/// Apply RMS normalization
pub fn normalize_rms(signal: &[f32], target_rms: f32) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let rms = (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt();

    if rms < 1e-8 {
        return signal.to_vec();
    }

    let scale = target_rms / rms;
    signal.iter().map(|x| x * scale).collect()
}

/// Apply soft clipping to prevent harsh distortion
pub fn soft_clip(signal: &[f32], threshold: f32) -> Vec<f32> {
    signal
        .iter()
        .map(|&x| {
            if x.abs() <= threshold {
                x
            } else {
                let sign = x.signum();
                let excess = x.abs() - threshold;
                sign * (threshold + (1.0 - (-excess).exp()))
            }
        })
        .collect()
}

/// Pad audio signal with zeros
pub fn pad_audio(signal: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let mut output = vec![0.0; pad_left];
    output.extend_from_slice(signal);
    output.extend(vec![0.0; pad_right]);
    output
}

/// Trim silence from beginning and end
pub fn trim_silence(signal: &[f32], threshold_db: f32) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let threshold = 10f32.powf(threshold_db / 20.0);

    // Find first non-silent sample
    let start = signal
        .iter()
        .position(|&x| x.abs() > threshold)
        .unwrap_or(0);

    // Find last non-silent sample
    let end = signal
        .iter()
        .rposition(|&x| x.abs() > threshold)
        .unwrap_or(signal.len() - 1);

    if start >= end {
        return vec![];
    }

    signal[start..=end].to_vec()
}

/// Apply fade in/out to avoid clicks
pub fn apply_fade(signal: &[f32], fade_in_samples: usize, fade_out_samples: usize) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let mut output = signal.to_vec();
    let len = output.len();

    // Fade in
    for i in 0..fade_in_samples.min(len) {
        let factor = i as f32 / fade_in_samples as f32;
        output[i] *= factor;
    }

    // Fade out
    for i in 0..fade_out_samples.min(len) {
        let idx = len - 1 - i;
        let factor = i as f32 / fade_out_samples as f32;
        output[idx] *= factor;
    }

    output
}

/// Compute RMS energy
pub fn compute_rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Compute peak amplitude
pub fn compute_peak(signal: &[f32]) -> f32 {
    signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

/// Compute crest factor (peak/RMS ratio)
pub fn compute_crest_factor(signal: &[f32]) -> f32 {
    let rms = compute_rms(signal);
    if rms < 1e-8 {
        return 0.0;
    }
    compute_peak(signal) / rms
}
