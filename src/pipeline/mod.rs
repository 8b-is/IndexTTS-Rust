//! Main TTS pipeline orchestration
//!
//! Coordinates text processing, model inference, and audio synthesis

mod synthesis;

pub use synthesis::{IndexTTS, SynthesisOptions, SynthesisResult};

use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// Pipeline stage enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    TextNormalization,
    Tokenization,
    SemanticEncoding,
    SpeakerConditioning,
    GptGeneration,
    AcousticExpansion,
    Vocoding,
    PostProcessing,
}

impl PipelineStage {
    /// Get stage name
    pub fn name(&self) -> &'static str {
        match self {
            PipelineStage::TextNormalization => "Text Normalization",
            PipelineStage::Tokenization => "Tokenization",
            PipelineStage::SemanticEncoding => "Semantic Encoding",
            PipelineStage::SpeakerConditioning => "Speaker Conditioning",
            PipelineStage::GptGeneration => "GPT Generation",
            PipelineStage::AcousticExpansion => "Acoustic Expansion",
            PipelineStage::Vocoding => "Vocoding",
            PipelineStage::PostProcessing => "Post Processing",
        }
    }

    /// Get all stages in order
    pub fn all() -> Vec<PipelineStage> {
        vec![
            PipelineStage::TextNormalization,
            PipelineStage::Tokenization,
            PipelineStage::SemanticEncoding,
            PipelineStage::SpeakerConditioning,
            PipelineStage::GptGeneration,
            PipelineStage::AcousticExpansion,
            PipelineStage::Vocoding,
            PipelineStage::PostProcessing,
        ]
    }
}

/// Pipeline progress callback
pub type ProgressCallback = Box<dyn Fn(PipelineStage, f32) + Send + Sync>;

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Model directory
    pub model_dir: PathBuf,
    /// Use FP16 inference
    pub use_fp16: bool,
    /// Device (cpu, cuda:0, etc.)
    pub device: String,
    /// Enable caching
    pub enable_cache: bool,
    /// Maximum text length
    pub max_text_length: usize,
    /// Maximum audio duration (seconds)
    pub max_audio_duration: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models"),
            use_fp16: false,
            device: "cpu".to_string(),
            enable_cache: true,
            max_text_length: 500,
            max_audio_duration: 30.0,
        }
    }
}

impl PipelineConfig {
    /// Create config with model directory
    pub fn with_model_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.model_dir = path.as_ref().to_path_buf();
        self
    }

    /// Enable FP16 inference
    pub fn with_fp16(mut self, enable: bool) -> Self {
        self.use_fp16 = enable;
        self
    }

    /// Set device
    pub fn with_device(mut self, device: &str) -> Self {
        self.device = device.to_string();
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if !self.model_dir.exists() {
            log::warn!(
                "Model directory does not exist: {}",
                self.model_dir.display()
            );
        }

        if self.max_text_length == 0 {
            return Err(Error::Config("max_text_length must be > 0".into()));
        }

        if self.max_audio_duration <= 0.0 {
            return Err(Error::Config("max_audio_duration must be > 0".into()));
        }

        Ok(())
    }
}

/// Text segmentation for long-form synthesis
pub fn segment_text(text: &str, max_segment_len: usize) -> Vec<String> {
    use crate::text::TextNormalizer;

    let normalizer = TextNormalizer::new();
    let sentences = normalizer.split_sentences(text);

    let mut segments = Vec::new();
    let mut current_segment = String::new();

    for sentence in sentences {
        if current_segment.len() + sentence.len() > max_segment_len && !current_segment.is_empty()
        {
            segments.push(current_segment.trim().to_string());
            current_segment = sentence;
        } else {
            if !current_segment.is_empty() {
                current_segment.push(' ');
            }
            current_segment.push_str(&sentence);
        }
    }

    if !current_segment.trim().is_empty() {
        segments.push(current_segment.trim().to_string());
    }

    segments
}

/// Concatenate audio segments with silence
pub fn concatenate_audio(segments: &[Vec<f32>], silence_duration_ms: u32, sample_rate: u32) -> Vec<f32> {
    let silence_samples = (silence_duration_ms as usize * sample_rate as usize) / 1000;
    let silence = vec![0.0f32; silence_samples];

    let mut result = Vec::new();

    for (i, segment) in segments.iter().enumerate() {
        result.extend_from_slice(segment);
        if i < segments.len() - 1 {
            result.extend_from_slice(&silence);
        }
    }

    result
}

/// Estimate synthesis duration
pub fn estimate_duration(text: &str, chars_per_second: f32) -> f32 {
    text.chars().count() as f32 / chars_per_second
}
