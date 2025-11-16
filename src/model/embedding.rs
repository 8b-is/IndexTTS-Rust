//! Speaker and emotion embedding models

use crate::{Error, Result};
use ndarray::{Array1, Array2, Array, IxDyn};
use std::collections::HashMap;
use std::path::Path;

use super::OnnxSession;

/// Speaker encoder for extracting speaker embeddings from audio
pub struct SpeakerEncoder {
    session: Option<OnnxSession>,
    embedding_dim: usize,
}

impl SpeakerEncoder {
    /// Load speaker encoder from ONNX model
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let session = OnnxSession::load(path)?;
        Ok(Self {
            session: Some(session),
            embedding_dim: 192, // CAMPPlus default
        })
    }

    /// Create placeholder encoder (for testing)
    pub fn new_placeholder(embedding_dim: usize) -> Self {
        Self {
            session: None,
            embedding_dim,
        }
    }

    /// Extract speaker embedding from mel spectrogram
    pub fn encode(&self, mel_spectrogram: &Array2<f32>) -> Result<Array1<f32>> {
        if let Some(ref session) = self.session {
            // Prepare input (add batch dimension)
            let input = mel_spectrogram
                .clone()
                .into_shape(IxDyn(&[1, mel_spectrogram.nrows(), mel_spectrogram.ncols()]))?;

            let mut inputs = HashMap::new();
            inputs.insert("mel".to_string(), input);

            let outputs = session.run(inputs)?;

            let embedding = outputs
                .get("embedding")
                .ok_or_else(|| Error::Model("Missing embedding output".into()))?;

            // Extract 1D embedding
            let flat: Vec<f32> = embedding.iter().cloned().collect();
            Ok(Array1::from_vec(flat))
        } else {
            // Return random embedding for testing
            Ok(Array1::from_vec(vec![0.0f32; self.embedding_dim]))
        }
    }

    /// Extract embedding from audio file
    pub fn encode_audio(&self, audio_path: &str) -> Result<Array1<f32>> {
        use crate::audio::{compute_mel_from_file, AudioConfig};

        let config = AudioConfig::default();
        let mel = compute_mel_from_file(audio_path, &config)?;
        self.encode(&mel)
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Normalize embedding to unit length
    pub fn normalize_embedding(&self, embedding: &Array1<f32>) -> Array1<f32> {
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            embedding / norm
        } else {
            embedding.clone()
        }
    }

    /// Compute cosine similarity between embeddings
    pub fn cosine_similarity(&self, emb1: &Array1<f32>, emb2: &Array1<f32>) -> f32 {
        let norm1 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 < 1e-8 || norm2 < 1e-8 {
            return 0.0;
        }

        let dot: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        dot / (norm1 * norm2)
    }
}

/// Emotion encoder for controlling emotional expression
pub struct EmotionEncoder {
    /// Emotion embedding matrix (num_emotions x embedding_dim)
    emotion_matrix: Array2<f32>,
    /// Number of emotion dimensions
    num_dims: usize,
    /// Values per dimension
    dim_sizes: Vec<usize>,
}

impl EmotionEncoder {
    /// Create emotion encoder with specified dimensions
    pub fn new(num_dims: usize, dim_sizes: Vec<usize>, embedding_dim: usize) -> Self {
        let total_emotions: usize = dim_sizes.iter().sum();
        let emotion_matrix = Array2::zeros((total_emotions, embedding_dim));

        Self {
            emotion_matrix,
            num_dims,
            dim_sizes,
        }
    }

    /// Load emotion matrix from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }

        // Load safetensors file
        let file_data = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&file_data)
            .map_err(|e| Error::ModelLoading(format!("Failed to load safetensors: {}", e)))?;

        // Extract emotion matrix
        let tensor = tensors
            .tensor("emotion_matrix")
            .map_err(|e| Error::ModelLoading(format!("Missing emotion_matrix: {}", e)))?;

        let shape = tensor.shape();
        let data: Vec<f32> = tensor.data().chunks(4).map(|b| {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }).collect();

        let emotion_matrix = Array2::from_shape_vec((shape[0], shape[1]), data)
            .map_err(|e| Error::ModelLoading(format!("Shape mismatch: {}", e)))?;

        // Default configuration
        let num_dims = 8;
        let dim_sizes = vec![5, 6, 8, 6, 5, 4, 7, 6];

        Ok(Self {
            emotion_matrix,
            num_dims,
            dim_sizes,
        })
    }

    /// Encode emotion vector to embedding
    pub fn encode(&self, emotion_vector: &[f32]) -> Result<Array1<f32>> {
        if emotion_vector.len() != self.num_dims {
            return Err(Error::ShapeMismatch {
                expected: format!("{} dimensions", self.num_dims),
                actual: format!("{} dimensions", emotion_vector.len()),
            });
        }

        let embedding_dim = self.emotion_matrix.ncols();
        let mut embedding = vec![0.0f32; embedding_dim];

        let mut offset = 0;
        for (dim_idx, (&value, &dim_size)) in emotion_vector.iter().zip(self.dim_sizes.iter()).enumerate() {
            // Interpolate between discrete emotion levels
            let continuous_idx = value * (dim_size - 1) as f32;
            let lower_idx = continuous_idx.floor() as usize;
            let upper_idx = (lower_idx + 1).min(dim_size - 1);
            let alpha = continuous_idx - lower_idx as f32;

            // Weighted combination
            for i in 0..embedding_dim {
                let lower_val = self.emotion_matrix[[offset + lower_idx, i]];
                let upper_val = self.emotion_matrix[[offset + upper_idx, i]];
                embedding[i] += lower_val * (1.0 - alpha) + upper_val * alpha;
            }

            offset += dim_size;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for e in embedding.iter_mut() {
                *e /= norm;
            }
        }

        Ok(Array1::from_vec(embedding))
    }

    /// Get neutral emotion (all zeros)
    pub fn neutral(&self) -> Vec<f32> {
        vec![0.5f32; self.num_dims]
    }

    /// Get preset emotion vectors
    pub fn preset(&self, name: &str) -> Vec<f32> {
        match name {
            "happy" => vec![0.9, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5],
            "sad" => vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.5, 0.5],
            "angry" => vec![0.8, 0.9, 0.7, 0.5, 0.3, 0.5, 0.5, 0.5],
            "fearful" => vec![0.3, 0.4, 0.8, 0.5, 0.7, 0.5, 0.5, 0.5],
            "surprised" => vec![0.7, 0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5],
            "neutral" | _ => self.neutral(),
        }
    }

    /// Interpolate between two emotion vectors
    pub fn interpolate(&self, emot1: &[f32], emot2: &[f32], alpha: f32) -> Vec<f32> {
        emot1
            .iter()
            .zip(emot2.iter())
            .map(|(&a, &b)| a * (1.0 - alpha) + b * alpha)
            .collect()
    }

    /// Apply emotion strength/alpha
    pub fn apply_strength(&self, emotion: &[f32], strength: f32) -> Vec<f32> {
        let neutral = self.neutral();
        self.interpolate(&neutral, emotion, strength)
    }
}

/// Semantic encoder for extracting semantic codes
pub struct SemanticEncoder {
    session: Option<OnnxSession>,
    embedding_dim: usize,
}

impl SemanticEncoder {
    /// Load semantic encoder
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let session = OnnxSession::load(path)?;
        Ok(Self {
            session: Some(session),
            embedding_dim: 1024,
        })
    }

    /// Create placeholder encoder
    pub fn new_placeholder() -> Self {
        Self {
            session: None,
            embedding_dim: 1024,
        }
    }

    /// Encode audio to semantic codes
    pub fn encode(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<i64>> {
        if let Some(ref session) = self.session {
            let input = Array::from_shape_vec(
                IxDyn(&[1, audio.len()]),
                audio.to_vec(),
            )?;

            let mut inputs = HashMap::new();
            inputs.insert("audio".to_string(), input);

            let outputs = session.run(inputs)?;

            let codes = outputs
                .get("codes")
                .ok_or_else(|| Error::Model("Missing codes output".into()))?;

            Ok(codes.iter().map(|&x| x as i64).collect())
        } else {
            // Return dummy codes for testing
            let num_codes = audio.len() / (sample_rate as usize / 50); // ~50 codes/sec
            Ok(vec![0i64; num_codes.max(1)])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_encoder_placeholder() {
        let encoder = SpeakerEncoder::new_placeholder(192);
        assert_eq!(encoder.embedding_dim(), 192);
    }

    #[test]
    fn test_emotion_encoder() {
        let encoder = EmotionEncoder::new(8, vec![5, 6, 8, 6, 5, 4, 7, 6], 256);
        let neutral = encoder.neutral();
        assert_eq!(neutral.len(), 8);
        assert!(neutral.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_emotion_presets() {
        let encoder = EmotionEncoder::new(8, vec![5, 6, 8, 6, 5, 4, 7, 6], 256);
        let happy = encoder.preset("happy");
        assert_eq!(happy.len(), 8);
        assert!(happy[0] > 0.5); // Happy has high first dimension
    }

    #[test]
    fn test_emotion_interpolation() {
        let encoder = EmotionEncoder::new(8, vec![5, 6, 8, 6, 5, 4, 7, 6], 256);
        let happy = encoder.preset("happy");
        let sad = encoder.preset("sad");
        let mid = encoder.interpolate(&happy, &sad, 0.5);

        // Middle value should be average
        for i in 0..8 {
            assert!((mid[i] - (happy[i] + sad[i]) / 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let encoder = SpeakerEncoder::new_placeholder(3);
        let emb1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let emb2 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let sim = encoder.cosine_similarity(&emb1, &emb2);
        assert!((sim - 1.0).abs() < 1e-6);

        let emb3 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let sim2 = encoder.cosine_similarity(&emb1, &emb3);
        assert!(sim2.abs() < 1e-6);
    }
}
