//! Model inference module for IndexTTS
//!
//! Provides ONNX Runtime-based model inference for TTS components

mod gpt;
mod embedding;
mod session;

pub use gpt::{GptModel, GptConfig};
pub use embedding::{SpeakerEncoder, EmotionEncoder, SemanticEncoder};
pub use session::{OnnxSession, ModelCache, OrtStatus, check_ort_availability};


/// Sampling strategy for generation
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Greedy decoding (always pick most likely token)
    Greedy,
    /// Top-k sampling
    TopK { k: usize },
    /// Top-p (nucleus) sampling
    TopP { p: f32 },
    /// Combined top-k and top-p
    TopKP { k: usize, p: f32 },
    /// Temperature-scaled sampling
    Temperature { temp: f32 },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::TopKP { k: 50, p: 0.95 }
    }
}

/// Sample from logits using specified strategy
pub fn sample_from_logits(logits: &[f32], strategy: &SamplingStrategy) -> usize {
    match strategy {
        SamplingStrategy::Greedy => {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
        SamplingStrategy::TopK { k } => {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            indexed.truncate(*k);

            // Apply softmax to top-k
            let max_logit = indexed[0].1;
            let exp_sum: f32 = indexed.iter().map(|(_, l)| (l - max_logit).exp()).sum();
            let probs: Vec<f32> = indexed
                .iter()
                .map(|(_, l)| (l - max_logit).exp() / exp_sum)
                .collect();

            sample_categorical(&indexed.iter().map(|(i, _)| *i).collect::<Vec<_>>(), &probs)
        }
        SamplingStrategy::TopP { p } => {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

            // Apply softmax
            let max_logit = indexed[0].1;
            let exp_sum: f32 = indexed.iter().map(|(_, l)| (l - max_logit).exp()).sum();
            let probs: Vec<f32> = indexed
                .iter()
                .map(|(_, l)| (l - max_logit).exp() / exp_sum)
                .collect();

            // Find nucleus
            let mut cumsum = 0.0;
            let mut nucleus_size = probs.len();
            for (i, prob) in probs.iter().enumerate() {
                cumsum += prob;
                if cumsum >= *p {
                    nucleus_size = i + 1;
                    break;
                }
            }

            // Renormalize nucleus
            let nucleus_sum: f32 = probs[..nucleus_size].iter().sum();
            let nucleus_probs: Vec<f32> = probs[..nucleus_size]
                .iter()
                .map(|p| p / nucleus_sum)
                .collect();

            sample_categorical(
                &indexed[..nucleus_size]
                    .iter()
                    .map(|(i, _)| *i)
                    .collect::<Vec<_>>(),
                &nucleus_probs,
            )
        }
        SamplingStrategy::TopKP { k, p } => {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            indexed.truncate(*k);

            // Apply softmax
            let max_logit = indexed[0].1;
            let exp_sum: f32 = indexed.iter().map(|(_, l)| (l - max_logit).exp()).sum();
            let probs: Vec<f32> = indexed
                .iter()
                .map(|(_, l)| (l - max_logit).exp() / exp_sum)
                .collect();

            // Find nucleus within top-k
            let mut cumsum = 0.0;
            let mut nucleus_size = probs.len();
            for (i, prob) in probs.iter().enumerate() {
                cumsum += prob;
                if cumsum >= *p {
                    nucleus_size = i + 1;
                    break;
                }
            }

            let nucleus_sum: f32 = probs[..nucleus_size].iter().sum();
            let nucleus_probs: Vec<f32> = probs[..nucleus_size]
                .iter()
                .map(|p| p / nucleus_sum)
                .collect();

            sample_categorical(
                &indexed[..nucleus_size]
                    .iter()
                    .map(|(i, _)| *i)
                    .collect::<Vec<_>>(),
                &nucleus_probs,
            )
        }
        SamplingStrategy::Temperature { temp } => {
            let scaled: Vec<f32> = logits.iter().map(|l| l / temp).collect();
            let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scaled.iter().map(|l| (l - max_logit).exp()).sum();
            let probs: Vec<f32> = scaled
                .iter()
                .map(|l| (l - max_logit).exp() / exp_sum)
                .collect();

            sample_categorical(&(0..probs.len()).collect::<Vec<_>>(), &probs)
        }
    }
}

/// Sample from categorical distribution
fn sample_categorical(indices: &[usize], probs: &[f32]) -> usize {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return indices[i];
        }
    }

    indices[indices.len() - 1]
}

/// Apply repetition penalty to logits
pub fn apply_repetition_penalty(logits: &mut [f32], previous_tokens: &[usize], penalty: f32) {
    for &token in previous_tokens {
        if token < logits.len() {
            if logits[token] > 0.0 {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}

/// Softmax function
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    logits
        .iter()
        .map(|l| (l - max_logit).exp() / exp_sum)
        .collect()
}

/// Log softmax function
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let log_sum = exp_sum.ln();
    logits.iter().map(|l| l - max_logit - log_sum).collect()
}
