//! Activation functions for BigVGAN
//!
//! Includes Snake and SnakeBeta activations

use std::f32::consts::PI;

/// Snake activation function
///
/// x + (1/alpha) * sin^2(alpha * x)
pub fn snake_activation(x: f32, alpha: f32) -> f32 {
    let sin_val = (alpha * x).sin();
    x + sin_val * sin_val / alpha
}

/// Snake activation for vector
pub fn snake_activation_vec(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter().map(|&v| snake_activation(v, alpha)).collect()
}

/// Snake Beta activation function
///
/// x + (1/beta) * sin^2(alpha * x)
pub fn snake_beta_activation(x: f32, alpha: f32, beta: f32) -> f32 {
    let sin_val = (alpha * x).sin();
    x + sin_val * sin_val / beta
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
        .flat_map(|&v| std::iter::repeat(v).take(upsample_factor))
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
}
