//! Configuration management for IndexTTS

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Main configuration for IndexTTS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// GPT model configuration
    pub gpt: GptConfig,
    /// Vocoder configuration
    pub vocoder: VocoderConfig,
    /// Semantic-to-Mel configuration
    pub s2mel: S2MelConfig,
    /// Dataset/tokenizer configuration
    pub dataset: DatasetConfig,
    /// Emotion configuration
    pub emotions: EmotionConfig,
    /// General inference settings
    pub inference: InferenceConfig,
    /// Model paths
    pub model_dir: PathBuf,
}

/// GPT model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptConfig {
    /// Number of transformer layers
    pub layers: usize,
    /// Model dimension
    pub model_dim: usize,
    /// Number of attention heads
    pub heads: usize,
    /// Maximum text tokens
    pub max_text_tokens: usize,
    /// Maximum mel tokens
    pub max_mel_tokens: usize,
    /// Stop token for mel generation
    pub stop_mel_token: usize,
    /// Start token for text
    pub start_text_token: usize,
    /// Start token for mel
    pub start_mel_token: usize,
    /// Number of mel codes
    pub num_mel_codes: usize,
    /// Number of text tokens in vocabulary
    pub num_text_tokens: usize,
}

/// Vocoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConfig {
    /// Model name/path
    pub name: String,
    /// Checkpoint path
    pub checkpoint: Option<PathBuf>,
    /// Use FP16 inference
    pub use_fp16: bool,
    /// Use DeepSpeed optimization
    pub use_deepspeed: bool,
}

/// Semantic-to-Mel model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2MelConfig {
    /// Checkpoint path
    pub checkpoint: PathBuf,
    /// Preprocessing parameters
    pub preprocess: PreprocessConfig,
}

/// Audio preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    /// Sample rate
    pub sr: u32,
    /// FFT size
    pub n_fft: usize,
    /// Hop length
    pub hop_length: usize,
    /// Window length
    pub win_length: usize,
    /// Number of mel bands
    pub n_mels: usize,
    /// Minimum frequency for mel filterbank
    pub fmin: f32,
    /// Maximum frequency for mel filterbank
    pub fmax: f32,
}

/// Dataset and tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// BPE model path
    pub bpe_model: PathBuf,
    /// Vocabulary size
    pub vocab_size: usize,
}

/// Emotion control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    /// Number of emotion dimensions
    pub num_dims: usize,
    /// Values per dimension
    pub num: Vec<usize>,
    /// Emotion matrix path
    pub matrix_path: Option<PathBuf>,
}

/// General inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Device to use (cpu, cuda:0, etc.)
    pub device: String,
    /// Use FP16 precision
    pub use_fp16: bool,
    /// Batch size
    pub batch_size: usize,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Top-p (nucleus) sampling parameter
    pub top_p: f32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Length penalty
    pub length_penalty: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gpt: GptConfig::default(),
            vocoder: VocoderConfig::default(),
            s2mel: S2MelConfig::default(),
            dataset: DatasetConfig::default(),
            emotions: EmotionConfig::default(),
            inference: InferenceConfig::default(),
            model_dir: PathBuf::from("models"),
        }
    }
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            layers: 8,
            model_dim: 512,
            heads: 8,
            max_text_tokens: 120,
            max_mel_tokens: 250,
            stop_mel_token: 8193,
            start_text_token: 8192,
            start_mel_token: 8192,
            num_mel_codes: 8194,
            num_text_tokens: 6681,
        }
    }
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            name: "bigvgan_v2_22khz_80band_256x".into(),
            checkpoint: None,
            use_fp16: true,
            use_deepspeed: false,
        }
    }
}

impl Default for S2MelConfig {
    fn default() -> Self {
        Self {
            checkpoint: PathBuf::from("models/s2mel.onnx"),
            preprocess: PreprocessConfig::default(),
        }
    }
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            sr: 22050,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8000.0,
        }
    }
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            bpe_model: PathBuf::from("models/bpe.model"),
            vocab_size: 6681,
        }
    }
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            num_dims: 8,
            num: vec![5, 6, 8, 6, 5, 4, 7, 6],
            matrix_path: Some(PathBuf::from("models/emotion_matrix.safetensors")),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            device: "cpu".into(),
            use_fp16: false,
            batch_size: 1,
            top_k: 50,
            top_p: 0.95,
            temperature: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
        }
    }
}

impl Config {
    /// Load configuration from YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }

        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_yaml::to_string(self)
            .map_err(|e| Error::Config(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }

        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Create default configuration and save to file
    pub fn create_default<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = Config::default();
        config.save(path)?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Check model directory exists
        if !self.model_dir.exists() {
            log::warn!(
                "Model directory does not exist: {}",
                self.model_dir.display()
            );
        }

        // Validate GPT config
        if self.gpt.layers == 0 {
            return Err(Error::Config("GPT layers must be > 0".into()));
        }
        if self.gpt.model_dim == 0 {
            return Err(Error::Config("GPT model_dim must be > 0".into()));
        }
        if self.gpt.heads == 0 {
            return Err(Error::Config("GPT heads must be > 0".into()));
        }
        if self.gpt.model_dim % self.gpt.heads != 0 {
            return Err(Error::Config(
                "GPT model_dim must be divisible by heads".into(),
            ));
        }

        // Validate preprocessing
        if self.s2mel.preprocess.sr == 0 {
            return Err(Error::Config("Sample rate must be > 0".into()));
        }
        if self.s2mel.preprocess.n_fft == 0 {
            return Err(Error::Config("n_fft must be > 0".into()));
        }
        if self.s2mel.preprocess.hop_length == 0 {
            return Err(Error::Config("hop_length must be > 0".into()));
        }

        // Validate inference settings
        if self.inference.temperature <= 0.0 {
            return Err(Error::Config("Temperature must be > 0".into()));
        }
        if self.inference.top_p <= 0.0 || self.inference.top_p > 1.0 {
            return Err(Error::Config("top_p must be in (0, 1]".into()));
        }

        Ok(())
    }
}
