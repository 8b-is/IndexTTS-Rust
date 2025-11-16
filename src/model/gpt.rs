//! GPT-based sequence generation model

use crate::{Error, Result};
use ndarray::{Array, Array1, Array2, Array3, IxDyn};
use std::collections::HashMap;
use std::path::Path;

use super::{OnnxSession, SamplingStrategy, sample_from_logits, apply_repetition_penalty};

/// GPT model configuration
#[derive(Debug, Clone)]
pub struct GptConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Model dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Stop token ID
    pub stop_token: usize,
    /// Start token ID
    pub start_token: usize,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            num_layers: 8,
            hidden_size: 512,
            num_heads: 8,
            max_seq_len: 250,
            vocab_size: 8194,
            stop_token: 8193,
            start_token: 8192,
        }
    }
}

/// GPT model for autoregressive generation
pub struct GptModel {
    session: OnnxSession,
    config: GptConfig,
}

impl GptModel {
    /// Load GPT model from ONNX file
    pub fn load<P: AsRef<Path>>(path: P, config: GptConfig) -> Result<Self> {
        let session = OnnxSession::load(path)?;
        Ok(Self { session, config })
    }

    /// Generate mel tokens from semantic tokens
    pub fn generate(
        &self,
        semantic_tokens: &[i64],
        speaker_embedding: &Array1<f32>,
        max_length: usize,
        strategy: &SamplingStrategy,
        repetition_penalty: f32,
    ) -> Result<Vec<i64>> {
        let mut generated_tokens = vec![self.config.start_token as i64];
        let mut past_tokens = Vec::new();

        for _ in 0..max_length {
            // Prepare input
            let input_tokens = Array::from_shape_vec(
                IxDyn(&[1, generated_tokens.len()]),
                generated_tokens.clone(),
            )?;

            let speaker_emb = speaker_embedding
                .clone()
                .into_shape(IxDyn(&[1, speaker_embedding.len()]))?;

            let semantic_input = Array::from_shape_vec(
                IxDyn(&[1, semantic_tokens.len()]),
                semantic_tokens.to_vec(),
            )?;

            // Create input map
            let mut inputs = HashMap::new();
            inputs.insert("input_ids".to_string(), input_tokens.mapv(|x| x as f32));
            inputs.insert("speaker_embedding".to_string(), speaker_emb);
            inputs.insert("semantic_tokens".to_string(), semantic_input.mapv(|x| x as f32));

            // Run inference
            let outputs = self.session.run(inputs)?;

            // Get logits for next token
            let logits = outputs
                .get("logits")
                .ok_or_else(|| Error::Model("Missing logits output".into()))?;

            // Get last token logits
            let seq_len = logits.shape()[1];
            let vocab_size = logits.shape()[2];
            let last_logits: Vec<f32> = (0..vocab_size)
                .map(|i| logits[[0, seq_len - 1, i]])
                .collect();

            // Apply repetition penalty
            let mut logits_vec = last_logits;
            let past_usize: Vec<usize> = past_tokens.iter().map(|&x| x as usize).collect();
            apply_repetition_penalty(&mut logits_vec, &past_usize, repetition_penalty);

            // Sample next token
            let next_token = sample_from_logits(&logits_vec, strategy) as i64;

            // Check for stop token
            if next_token == self.config.stop_token as i64 {
                break;
            }

            generated_tokens.push(next_token);
            past_tokens.push(next_token);
        }

        Ok(generated_tokens)
    }

    /// Generate with KV cache for efficiency
    pub fn generate_with_cache(
        &self,
        semantic_tokens: &[i64],
        speaker_embedding: &Array1<f32>,
        max_length: usize,
        strategy: &SamplingStrategy,
        repetition_penalty: f32,
    ) -> Result<Vec<i64>> {
        // For models with KV cache support
        // This is a simplified version - full implementation would maintain cache state
        self.generate(
            semantic_tokens,
            speaker_embedding,
            max_length,
            strategy,
            repetition_penalty,
        )
    }

    /// Get model config
    pub fn config(&self) -> &GptConfig {
        &self.config
    }

    /// Estimate memory usage
    pub fn estimate_memory_mb(&self) -> f32 {
        let params = self.config.num_layers
            * self.config.hidden_size
            * self.config.hidden_size
            * 4; // Approximate
        (params * 4) as f32 / 1_000_000.0 // 4 bytes per param
    }
}

/// Simplified GPT model using pure Rust (fallback when ONNX not available)
pub struct SimpleGptModel {
    config: GptConfig,
    /// Token embeddings
    token_embeddings: Array2<f32>,
    /// Position embeddings
    position_embeddings: Array2<f32>,
    /// Output projection
    output_projection: Array2<f32>,
}

impl SimpleGptModel {
    /// Create random initialized model (for testing)
    pub fn new_random(config: GptConfig) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let token_embeddings = Array2::from_shape_fn(
            (config.vocab_size, config.hidden_size),
            |_| rng.gen_range(-0.1..0.1),
        );

        let position_embeddings = Array2::from_shape_fn(
            (config.max_seq_len, config.hidden_size),
            |_| rng.gen_range(-0.1..0.1),
        );

        let output_projection = Array2::from_shape_fn(
            (config.hidden_size, config.vocab_size),
            |_| rng.gen_range(-0.1..0.1),
        );

        Self {
            config,
            token_embeddings,
            position_embeddings,
            output_projection,
        }
    }

    /// Simple forward pass (for demonstration)
    pub fn forward(&self, tokens: &[i64]) -> Vec<f32> {
        // Get embeddings
        let mut hidden = vec![0.0f32; self.config.hidden_size];

        for (pos, &token) in tokens.iter().enumerate().take(self.config.max_seq_len) {
            let token_idx = (token as usize).min(self.config.vocab_size - 1);

            for i in 0..self.config.hidden_size {
                hidden[i] += self.token_embeddings[[token_idx, i]]
                    + self.position_embeddings[[pos, i]];
            }
        }

        // Normalize
        let norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for h in hidden.iter_mut() {
                *h /= norm;
            }
        }

        // Project to vocab
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for (i, logit) in logits.iter_mut().enumerate() {
            for j in 0..self.config.hidden_size {
                *logit += hidden[j] * self.output_projection[[j, i]];
            }
        }

        logits
    }

    /// Generate tokens
    pub fn generate(
        &self,
        prompt: &[i64],
        max_length: usize,
        strategy: &SamplingStrategy,
    ) -> Vec<i64> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_length {
            let logits = self.forward(&tokens);
            let next_token = sample_from_logits(&logits, strategy) as i64;

            if next_token == self.config.stop_token as i64 {
                break;
            }

            tokens.push(next_token);

            if tokens.len() >= self.config.max_seq_len {
                break;
            }
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_config_default() {
        let config = GptConfig::default();
        assert_eq!(config.num_layers, 8);
        assert_eq!(config.hidden_size, 512);
    }

    #[test]
    fn test_simple_gpt_forward() {
        let config = GptConfig {
            vocab_size: 100,
            hidden_size: 32,
            max_seq_len: 10,
            ..Default::default()
        };

        let model = SimpleGptModel::new_random(config);
        let tokens = vec![1i64, 2, 3];
        let logits = model.forward(&tokens);

        assert_eq!(logits.len(), 100);
    }

    #[test]
    fn test_simple_gpt_generate() {
        let config = GptConfig {
            vocab_size: 100,
            hidden_size: 32,
            max_seq_len: 20,
            stop_token: 99,
            ..Default::default()
        };

        let model = SimpleGptModel::new_random(config);
        let prompt = vec![1i64, 2, 3];
        let generated = model.generate(&prompt, 10, &SamplingStrategy::Greedy);

        assert!(generated.len() >= 3);
        assert!(generated.len() <= 20);
    }
}
