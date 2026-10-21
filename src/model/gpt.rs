//! GPT-based sequence generation model

use crate::{Error, Result};
use ndarray::{Array, Array1, Array2, IxDyn};
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

/// Packed Ternary + Group-64 Scales (`BFP-64` / `p/q`) vocabulary projection for `SimpleGptModel`.
///
/// Compresses the `[hidden_size, vocab_size]` (`512 x 8194 = 4,195,328` weights) output projection
/// from `16.78 MB` (FP32) down to `1.05 MB` (2-bit ternary weights + group-64 scales) and provides
/// three integer inner-loop forward modes:
/// 1. `forward_pq`: `q = 2^20` numerators + `i64` accumulator ("Counting Divided" reference)
/// 2. `forward_bfp_i8`: Omni's `BFP-64` `i8` mantissa `[-127, 127]` + `i16` block accumulator (`|S_64| <= 8128`)
/// 3. `forward_mem8_adaptive`: MEM|8 adaptive `i8 -> i16` block promotion when quantization error exceeds `tau`
pub struct QuantizedGptProjection {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_groups: usize,
    /// Ternary weights stored row-major by `[vocab_idx][hidden_idx]` in `{-1, 0, +1}` as `i8`
    /// plus packed 2-bit bytes (`4` weights/byte) for compact storage (`1.05 MB` for `512 x 8194`)
    pub packed_2bit: Vec<u8>,
    /// Unpacked `i8` ternary sign cache `{-1, 0, +1}` for fast SIMD vectorization `[vocab_size * hidden_size]`
    pub ternary_signs: Vec<i8>,
    /// Per-group (64 weights) scale in `p/q` (`q_s = 2^16 = 65536`) `[vocab_size * num_groups]`
    pub group_scales_pq: Vec<i32>,
}

impl SimpleGptModel {
    /// Quantize `output_projection` (`[hidden_size, vocab_size]`) into Group-64 ternary `{-1, 0, +1}`
    /// with per-group scales.
    pub fn quantize_output_projection(&self) -> QuantizedGptProjection {
        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        let num_groups = h.div_ceil(64);
        let mut packed_2bit = vec![0u8; (v * h).div_ceil(4)];
        let mut ternary_signs = vec![0i8; v * h];
        let mut group_scales_pq = vec![0i32; v * num_groups];

        for vi in 0..v {
            for g in 0..num_groups {
                let start = g * 64;
                let end = (start + 64).min(h);
                let mut sum_abs = 0.0f32;
                for hj in start..end {
                    sum_abs += self.output_projection[[hj, vi]].abs();
                }
                let mean_abs = sum_abs / ((end - start) as f32);
                let threshold = 0.5 * mean_abs;

                let mut active_sum = 0.0f32;
                let mut active_cnt = 0usize;
                for hj in start..end {
                    let w = self.output_projection[[hj, vi]];
                    let flat_idx = vi * h + hj;
                    let (sign, code): (i8, u8) = if w > threshold {
                        active_sum += w.abs();
                        active_cnt += 1;
                        (1, 0b01)
                    } else if w < -threshold {
                        active_sum += w.abs();
                        active_cnt += 1;
                        (-1, 0b10)
                    } else {
                        (0, 0b00)
                    };
                    ternary_signs[flat_idx] = sign;
                    packed_2bit[flat_idx >> 2] |= code << ((flat_idx & 3) * 2);
                }
                let scale = if active_cnt > 0 {
                    active_sum / (active_cnt as f32)
                } else {
                    mean_abs
                };
                group_scales_pq[vi * num_groups + g] = (scale * 65_536.0).round() as i32;
            }
        }

        QuantizedGptProjection {
            vocab_size: v,
            hidden_size: h,
            num_groups,
            packed_2bit,
            ternary_signs,
            group_scales_pq,
        }
    }

    /// Compute normalized hidden state `[hidden_size]` for `tokens`.
    pub fn compute_hidden(&self, tokens: &[i64]) -> Vec<f32> {
        let mut hidden = vec![0.0f32; self.config.hidden_size];
        for (pos, &token) in tokens.iter().enumerate().take(self.config.max_seq_len) {
            let token_idx = (token as usize).min(self.config.vocab_size - 1);
            for (i, h_val) in hidden.iter_mut().enumerate() {
                *h_val += self.token_embeddings[[token_idx, i]]
                    + self.position_embeddings[[pos, i]];
            }
        }
        let norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for h_val in hidden.iter_mut() {
                *h_val /= norm;
            }
        }
        hidden
    }

    /// Forward pass using **p/q ("Counting Divided", `q = 2^20`)** + `i64` group accumulation.
    pub fn forward_pq(&self, tokens: &[i64], qproj: &QuantizedGptProjection) -> Vec<f32> {
        let hidden = self.compute_hidden(tokens);
        let h = qproj.hidden_size;
        let num_groups = qproj.num_groups;

        // Quantize hidden state once to p/q (`q_x = 2^20`)
        let px: Vec<i32> = hidden
            .iter()
            .map(|&x| (x * 1_048_576.0).round() as i32)
            .collect();
        let inv_qx_qs = 1.0f32 / ((1_048_576.0_f64 * 65_536.0_f64) as f32);

        let mut logits = Vec::with_capacity(qproj.vocab_size);
        for vi in 0..qproj.vocab_size {
            let row_signs = &qproj.ternary_signs[vi * h..(vi + 1) * h];
            let row_scales = &qproj.group_scales_pq[vi * num_groups..(vi + 1) * num_groups];
            let mut total_i64: i64 = 0;

            for (g, &ps_g) in row_scales.iter().enumerate() {
                let start = g * 64;
                let end = (start + 64).min(h);
                let mut group_acc: i32 = 0;
                for (&x_k, &w_k) in px[start..end].iter().zip(row_signs[start..end].iter()) {
                    group_acc += x_k * (w_k as i32);
                }
                total_i64 += (group_acc as i64) * (ps_g as i64);
            }
            logits.push((total_i64 as f32) * inv_qx_qs);
        }
        logits
    }

    /// Forward pass using **Omni's BFP-64 (`i8` mantissa `[-127, 127]` + `i16` block accumulator)**.
    ///
    /// Because each block has 64 elements with `|m_k| <= 127` and `w_{i,k} in {-1, 0, +1}`,
    /// the worst-case block sum is `64 * 127 = 8,128 <= 32,767 (i16::MAX)`, enabling
    /// 16-wide `i16` SIMD vectorization with zero overflow risk!
    pub fn forward_bfp_i8(&self, tokens: &[i64], qproj: &QuantizedGptProjection) -> Vec<f32> {
        let hidden = self.compute_hidden(tokens);
        let h = qproj.hidden_size;
        let num_groups = qproj.num_groups;

        // Quantize hidden into 64-element BFP-i8 blocks (`m_k in [-127, 127]` + block scale)
        let mut mantissas = vec![0i8; h];
        let mut block_scales = vec![0.0f32; num_groups];
        for g in 0..num_groups {
            let start = g * 64;
            let end = (start + 64).min(h);
            let max_abs = hidden[start..end]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs > 1e-12 {
                let scale = max_abs / 127.0;
                let inv_scale = 127.0 / max_abs;
                block_scales[g] = scale / 65_536.0;
                for k in start..end {
                    mantissas[k] = (hidden[k] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }

        let mut logits = Vec::with_capacity(qproj.vocab_size);
        for vi in 0..qproj.vocab_size {
            let row_signs = &qproj.ternary_signs[vi * h..(vi + 1) * h];
            let row_scales = &qproj.group_scales_pq[vi * num_groups..(vi + 1) * num_groups];
            let mut logit = 0.0f32;

            for g in 0..num_groups {
                let start = g * 64;
                let end = (start + 64).min(h);
                // Exact i16 accumulation: |acc_i16| <= 64 * 127 = 8,128 <= 32,767
                let mut acc_i16: i16 = 0;
                for (&m_k, &w_k) in mantissas[start..end]
                    .iter()
                    .zip(row_signs[start..end].iter())
                {
                    acc_i16 += (m_k as i16) * (w_k as i16);
                }
                logit += (acc_i16 as f32) * (row_scales[g] as f32) * block_scales[g];
            }
            logits.push(logit);
        }
        logits
    }

    /// Forward pass using **MEM|8 Adaptive Precision (`BFP-i8` fast path + `BFP-i16` promotion)**.
    ///
    /// Starts every 64-element block in `BFP-i8` (`i16` accumulator) and automatically promotes
    /// outlier/high-salience blocks whose relative quantization error exceeds `tau` to `BFP-i16`
    /// (`i32` accumulator).
    pub fn forward_mem8_adaptive(
        &self,
        tokens: &[i64],
        qproj: &QuantizedGptProjection,
        tau: f32,
    ) -> (Vec<f32>, usize) {
        let hidden = self.compute_hidden(tokens);
        let h = qproj.hidden_size;
        let num_groups = qproj.num_groups;

        let mut m_i8 = vec![0i8; h];
        let mut m_i16 = vec![0i16; h];
        let mut promoted = vec![false; num_groups];
        let mut block_scales = vec![0.0f32; num_groups];
        let mut promoted_count = 0usize;

        for g in 0..num_groups {
            let start = g * 64;
            let end = (start + 64).min(h);
            let max_abs = hidden[start..end]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs <= 1e-12 {
                continue;
            }
            let scale8 = max_abs / 127.0;
            let inv8 = 127.0 / max_abs;
            let mut err_sq = 0.0f32;
            let mut sig_sq = 0.0f32;
            for k in start..end {
                let q = (hidden[k] * inv8).round().clamp(-127.0, 127.0) as i8;
                m_i8[k] = q;
                let diff = hidden[k] - (q as f32) * scale8;
                err_sq += diff * diff;
                sig_sq += hidden[k] * hidden[k];
            }
            let rel_err = (err_sq / sig_sq.max(1e-12)).sqrt();
            if rel_err > tau {
                promoted[g] = true;
                promoted_count += 1;
                let scale16 = max_abs / 32_767.0;
                let inv16 = 32_767.0 / max_abs;
                block_scales[g] = scale16 / 65_536.0;
                for k in start..end {
                    m_i16[k] = (hidden[k] * inv16).round().clamp(-32_767.0, 32_767.0) as i16;
                }
            } else {
                block_scales[g] = scale8 / 65_536.0;
            }
        }

        let mut logits = Vec::with_capacity(qproj.vocab_size);
        for vi in 0..qproj.vocab_size {
            let row_signs = &qproj.ternary_signs[vi * h..(vi + 1) * h];
            let row_scales = &qproj.group_scales_pq[vi * num_groups..(vi + 1) * num_groups];
            let mut logit = 0.0f32;

            for g in 0..num_groups {
                let start = g * 64;
                let end = (start + 64).min(h);
                if promoted[g] {
                    let mut acc_i32: i32 = 0;
                    for (&mk, &wk) in m_i16[start..end].iter().zip(row_signs[start..end].iter()) {
                        acc_i32 += (mk as i32) * (wk as i32);
                    }
                    logit += (acc_i32 as f32) * (row_scales[g] as f32) * block_scales[g];
                } else {
                    let mut acc_i16: i16 = 0;
                    for (&mk, &wk) in m_i8[start..end].iter().zip(row_signs[start..end].iter()) {
                        acc_i16 += (mk as i16) * (wk as i16);
                    }
                    logit += (acc_i16 as f32) * (row_scales[g] as f32) * block_scales[g];
                }
            }
            logits.push(logit);
        }

        (logits, promoted_count)
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

    #[test]
    fn test_gpt_pq_bfp_benchmark() {
        use std::time::Instant;

        // Full IndexTTS GPT dimension: hidden_size = 512, vocab_size = 8194 (4,195,328 weights)
        let config = GptConfig::default();
        let model = SimpleGptModel::new_random(config);
        let qproj = model.quantize_output_projection();
        let tokens = vec![10i64, 42, 108, 256, 512, 1024];
        let iters = 20;

        // Warmup
        let ref_logits = model.forward(&tokens);
        let pq_logits = model.forward_pq(&tokens, &qproj);
        let bfp8_logits = model.forward_bfp_i8(&tokens, &qproj);
        let (adapt_logits, promoted_blocks) = model.forward_mem8_adaptive(&tokens, &qproj, 0.0048);

        assert_eq!(ref_logits.len(), 8194);
        assert_eq!(pq_logits.len(), 8194);
        assert_eq!(bfp8_logits.len(), 8194);
        assert_eq!(adapt_logits.len(), 8194);

        // Measure relative error between BFP-i8 / MEM|8 Adaptive and the p/q reference
        let norm_pq: f64 = pq_logits.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let err_bfp8: f64 = pq_logits
            .iter()
            .zip(bfp8_logits.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm_pq.max(1e-12);
        let err_adapt: f64 = pq_logits
            .iter()
            .zip(adapt_logits.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm_pq.max(1e-12);

        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(model.forward(std::hint::black_box(&tokens)));
        }
        let fp32_us = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t1 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(model.forward_pq(std::hint::black_box(&tokens), &qproj));
        }
        let pq_us = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t2 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(model.forward_bfp_i8(std::hint::black_box(&tokens), &qproj));
        }
        let bfp8_us = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t3 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(model.forward_mem8_adaptive(
                std::hint::black_box(&tokens),
                &qproj,
                0.0048,
            ));
        }
        let adapt_us = t3.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let fp32_bytes = 512 * 8194 * 4;
        let packed_bytes = qproj.packed_2bit.len() + qproj.group_scales_pq.len() * 2;

        println!("\n=== INDEXTTS-RUST GPT OUTPUT PROJECTION BENCHMARK (512 x 8194 = 4,195,328 weights) ===");
        println!(
            "  Storage Compression: {:.2} MB (FP32) -> {:.2} MB (2-bit Ternary + Group-64) = {:.2}x smaller",
            (fp32_bytes as f64) / 1e6,
            (packed_bytes as f64) / 1e6,
            (fp32_bytes as f64) / (packed_bytes as f64)
        );
        println!("  Dense FP32 Forward:                    {:>9.2} us (1.00x)", fp32_us);
        println!(
            "  p/q (q=2^20) + i64 Accumulator:        {:>9.2} us ({:.2}x speedup)",
            pq_us,
            fp32_us / pq_us
        );
        println!(
            "  Omni BFP-i8 + i16 Accumulator:         {:>9.2} us ({:.2}x speedup) | rel_err vs p/q: {:.5}",
            bfp8_us,
            fp32_us / bfp8_us,
            err_bfp8
        );
        println!(
            "  MEM|8 Adaptive (i8->i16, {}/{} prom):   {:>9.2} us ({:.2}x speedup) | rel_err vs p/q: {:.5}",
            promoted_blocks,
            qproj.num_groups,
            adapt_us,
            fp32_us / adapt_us,
            err_adapt
        );

        assert!(bfp8_us < fp32_us);
    }
}
