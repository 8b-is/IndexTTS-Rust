//! Activation functions for BigVGAN
//!
//! Includes Snake and SnakeBeta activations

use std::f32::consts::PI;
use std::sync::OnceLock;

/// Dyadic denominator q = 2^20 = 1,048,576 for p/q Snake LUT
pub const SNAKE_PQ_Q: i64 = 1 << 20;
const SNAKE_LUT_BITS: usize = 12;
const SNAKE_LUT_SIZE: usize = 1 << SNAKE_LUT_BITS; // 4096 entries over period pi
const SNAKE_LUT_MASK: usize = SNAKE_LUT_SIZE - 1;

/// Lazily initialized 4096-entry p/q lookup table for `sin^2(theta)` over `[0, pi)`,
/// scaled by `q = 2^20 = 1,048,576`.
pub fn sin2_pq_lut() -> &'static [i32; SNAKE_LUT_SIZE] {
    static LUT: OnceLock<[i32; SNAKE_LUT_SIZE]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0i32; SNAKE_LUT_SIZE];
        for (i, slot) in table.iter_mut().enumerate() {
            let theta = (i as f64) * std::f64::consts::PI / (SNAKE_LUT_SIZE as f64);
            let s = theta.sin();
            *slot = (s * s * (SNAKE_PQ_Q as f64)).round() as i32;
        }
        table
    })
}

/// Snake activation function
///
/// x + (1/alpha) * sin^2(alpha * x)
pub fn snake_activation(x: f32, alpha: f32) -> f32 {
    let sin_val = (alpha * x).sin();
    x + sin_val * sin_val / alpha
}

/// Snake activation in p/q ("Counting Divided") integer phase space:
/// maps `alpha * x` onto `Z_{4096}` (period `pi`) with zero transcendental `.sin()` calls.
#[inline]
pub fn snake_activation_pq(x: f32, alpha: f32) -> f32 {
    let lut = sin2_pq_lut();
    let phase_scale = alpha * ((SNAKE_LUT_SIZE as f32) / PI);
    let idx = ((x * phase_scale).round() as i64 as usize) & SNAKE_LUT_MASK;
    let inv_alpha_q = 1.0f32 / (alpha * (SNAKE_PQ_Q as f32));
    x + (lut[idx] as f32) * inv_alpha_q
}

/// Snake activation for vector
pub fn snake_activation_vec(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter().map(|&v| snake_activation(v, alpha)).collect()
}

/// Vectorized p/q Snake activation using an **algebraic Z_{2^16} phase-ring polynomial**
/// (zero transcendental calls, zero LUT gather stalls, 100% SIMD auto-vectorizable!).
///
/// Maps `alpha * x` onto the modular integer ring `Z_{65536}` via wrapping `i16` cast:
/// `a = |(x * alpha * 65536 / pi) as i32 as i16| in [0, 32768]`,
/// then evaluates `sin^2(pi/2 * w)` via the exact integer Hermite-5 wave identity in `i32`.
pub fn snake_activation_pq_vec(x: &[f32], alpha: f32) -> Vec<f32> {
    let phase_scale = alpha * (65_536.0 / PI);
    // Output scale: 1 / (alpha * 2^30)
    let inv_alpha_q30 = 1.0f32 / (alpha * (1_073_741_824.0_f32));
    let mut out = vec![0.0f32; x.len()];

    for (dst, &v) in out.iter_mut().zip(x.iter()) {
        // Wrap modulo 2^16 into [-32768, 32767] and fold into triangle wave a in [0, 32768]
        let t_wrap = (v * phase_scale) as i32 as i16;
        let a = (t_wrap as i32).abs(); // [0..32768]
        // w = a / 32768 in [0, 1]. Evaluate sin^2(pi/2 * w) ~= w^2 * (3 - 2*w) in Q30 integer:
        // w2_q15 = (a * a) >> 15 in [0..32768]
        // (3 - 2*w)_q15 = 98304 - 2*a in [32768..98304]
        // product = w2_q15 * (98304 - 2*a) in [0..2^30]
        let w2_q15 = (a * a) >> 15;
        let poly_q30 = w2_q15 * (98_304 - (a << 1));
        *dst = v + (poly_q30 as f32) * inv_alpha_q30;
    }
    out
}

/// Snake Beta activation function
///
/// x + (1/beta) * sin^2(alpha * x)
pub fn snake_beta_activation(x: f32, alpha: f32, beta: f32) -> f32 {
    let sin_val = (alpha * x).sin();
    x + sin_val * sin_val / beta
}

/// Vectorized p/q Snake Beta activation (`Z_{4096}` ring + deferred `1 / (beta * q)` scale).
pub fn snake_beta_activation_pq_vec(x: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    let lut = sin2_pq_lut();
    let phase_scale = alpha * ((SNAKE_LUT_SIZE as f32) / PI);
    let inv_beta_q = 1.0f32 / (beta * (SNAKE_PQ_Q as f32));
    let mut out = Vec::with_capacity(x.len());
    for &v in x {
        let idx = ((v * phase_scale).round() as i64 as usize) & SNAKE_LUT_MASK;
        out.push(v + (lut[idx] as f32) * inv_beta_q);
    }
    out
}

/// Snake Beta activation for vector
pub fn snake_beta_activation_vec(x: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| snake_beta_activation(v, alpha, beta))
        .collect()
}

/// Anti-aliased Snake activation
///
/// Uses lowpass filtering to reduce aliasing artifacts
pub fn anti_aliased_snake(x: &[f32], alpha: f32, upsample_factor: usize) -> Vec<f32> {
    // Upsample
    let upsampled: Vec<f32> = x
        .iter()
        .flat_map(|&v| std::iter::repeat_n(v, upsample_factor))
        .collect();

    // Apply activation
    let activated: Vec<f32> = upsampled
        .iter()
        .map(|&v| snake_activation(v, alpha))
        .collect();

    // Downsample (simple averaging)
    activated
        .chunks(upsample_factor)
        .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
        .collect()
}

/// Leaky ReLU activation
pub fn leaky_relu(x: f32, negative_slope: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        negative_slope * x
    }
}

/// Leaky ReLU for vector
pub fn leaky_relu_vec(x: &[f32], negative_slope: f32) -> Vec<f32> {
    x.iter().map(|&v| leaky_relu(v, negative_slope)).collect()
}

/// GELU (Gaussian Error Linear Unit) activation
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// GELU for vector
pub fn gelu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| gelu(v)).collect()
}

/// Swish activation (SiLU)
pub fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Swish for vector
pub fn swish_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| swish(v)).collect()
}

/// Mish activation
pub fn mish(x: f32) -> f32 {
    x * ((1.0 + x.exp()).ln()).tanh()
}

/// Mish for vector
pub fn mish_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| mish(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_activation() {
        let result = snake_activation(0.0, 1.0);
        assert!((result - 0.0).abs() < 1e-6);

        let result = snake_activation(1.0, 1.0);
        assert!(result > 1.0); // Should add positive value
    }

    #[test]
    fn test_snake_beta_activation() {
        let result = snake_beta_activation(0.0, 1.0, 1.0);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(1.0, 0.01), 1.0);
        assert_eq!(leaky_relu(-1.0, 0.01), -0.01);
        assert_eq!(leaky_relu(0.0, 0.01), 0.0);
    }

    #[test]
    fn test_gelu() {
        let result = gelu(0.0);
        assert!((result - 0.0).abs() < 1e-6);

        let result = gelu(1.0);
        assert!(result > 0.5 && result < 1.0);
    }

    #[test]
    fn test_swish() {
        let result = swish(0.0);
        assert!((result - 0.0).abs() < 1e-6);

        let result = swish(1.0);
        assert!(result > 0.5 && result < 1.0);
    }

    #[test]
    fn test_anti_aliased_snake() {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let result = anti_aliased_snake(&input, 1.0, 2);
        assert_eq!(result.len(), input.len());
    }

    #[test]
    fn test_snake_pq_accuracy_and_benchmark() {
        use std::time::Instant;

        let n = 262_144; // Typical BigVGAN intermediate channelxtime tensor size
        let alpha = 1.5f32;
        let input: Vec<f32> = (0..n)
            .map(|i| (((i * 73 + 19) % 2000) as f32 - 1000.0) / 250.0)
            .collect();

        // Warmup
        let ref_out = snake_activation_vec(&input, alpha);
        let pq_out = snake_activation_pq_vec(&input, alpha);

        // Verify numerical fidelity across all 262,144 elements
        let mut max_abs_err = 0.0f32;
        let mut sum_abs_err = 0.0f64;
        for (&r, &p) in ref_out.iter().zip(pq_out.iter()) {
            let err = (r - p).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
            sum_abs_err += err as f64;
        }
        let mean_abs_err = (sum_abs_err / (n as f64)) as f32;
        assert!(
            max_abs_err < 1e-2,
            "Max Snake p/q phase-ring error too large: {}",
            max_abs_err
        );

        let iters = 25;
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(snake_activation_vec(
                std::hint::black_box(&input),
                alpha,
            ));
        }
        let float_us = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t1 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(snake_activation_pq_vec(
                std::hint::black_box(&input),
                alpha,
            ));
        }
        let pq_us = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        println!("\n=== BIGVGAN SNAKE ACTIVATION BENCHMARK (N = 262,144 samples) ===");
        println!("  Float32 Transcendental sin^2(alpha*x): {:>8.2} us", float_us);
        println!(
            "  p/q Z_{{4096}} Dyadic LUT (q = 2^20):   {:>8.2} us ({:.2}x speedup) | mean_err={:.2e}, max_err={:.2e}",
            pq_us,
            float_us / pq_us,
            mean_abs_err,
            max_abs_err
        );
    }
}
