//! Core TTS synthesis implementation

use crate::{
    audio::{load_audio, save_audio, AudioConfig, AudioData},
    config::Config,
    model::{EmotionEncoder, SamplingStrategy, SemanticEncoder, SpeakerEncoder},
    text::{TextNormalizer, TextTokenizer, TokenizerConfig},
    vocoder::{BigVGAN, BigVGANConfig, Vocoder}, Result,
};
use ndarray::Array1;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Synthesis options
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Emotion vector (8 dimensions, 0-1)
    pub emotion_vector: Option<Vec<f32>>,
    /// Emotion audio reference path
    pub emotion_audio: Option<PathBuf>,
    /// Emotion alpha (strength)
    pub emotion_alpha: f32,
    /// Sampling strategy
    pub sampling: SamplingStrategy,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Maximum generation length
    pub max_length: usize,
    /// Silence between segments (ms)
    pub segment_silence_ms: u32,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            emotion_vector: None,
            emotion_audio: None,
            emotion_alpha: 1.0,
            sampling: SamplingStrategy::TopKP { k: 50, p: 0.95 },
            repetition_penalty: 1.1,
            max_length: 250,
            segment_silence_ms: 200,
        }
    }
}

/// Synthesis result
#[derive(Debug)]
pub struct SynthesisResult {
    /// Generated audio samples
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Processing time in seconds
    pub processing_time: f32,
    /// Real-time factor
    pub rtf: f32,
}

impl SynthesisResult {
    /// Save to WAV file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let audio_data = AudioData::new(self.audio.clone(), self.sample_rate);
        save_audio(path, &audio_data)
    }

    /// Get duration formatted as MM:SS
    pub fn duration_formatted(&self) -> String {
        let minutes = (self.duration / 60.0) as u32;
        let seconds = (self.duration % 60.0) as u32;
        format!("{:02}:{:02}", minutes, seconds)
    }
}

/// Main IndexTTS synthesizer
pub struct IndexTTS {
    /// Text normalizer
    normalizer: TextNormalizer,
    /// Tokenizer
    tokenizer: TextTokenizer,
    /// Speaker encoder
    speaker_encoder: SpeakerEncoder,
    /// Emotion encoder
    emotion_encoder: EmotionEncoder,
    /// Semantic encoder
    semantic_encoder: SemanticEncoder,
    /// Vocoder
    vocoder: BigVGAN,
    /// Audio configuration
    audio_config: AudioConfig,
    /// Model configuration
    config: Config,
}

impl IndexTTS {
    /// Create new IndexTTS from configuration
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;

        log::info!("Initializing IndexTTS...");

        // Initialize text processing
        let normalizer = TextNormalizer::new();
        let tokenizer = TextTokenizer::new(TokenizerConfig {
            model_path: config.dataset.bpe_model.display().to_string(),
            vocab_size: config.dataset.vocab_size,
            ..Default::default()
        })?;

        // Initialize encoders (using placeholders for now)
        let speaker_encoder = SpeakerEncoder::new_placeholder(192);
        let emotion_encoder = EmotionEncoder::new(
            config.emotions.num_dims,
            config.emotions.num.clone(),
            256,
        );
        let semantic_encoder = SemanticEncoder::new_placeholder();

        // Initialize vocoder
        let vocoder_config = BigVGANConfig {
            sample_rate: config.s2mel.preprocess.sr,
            num_mels: config.s2mel.preprocess.n_mels,
            ..Default::default()
        };
        let vocoder = BigVGAN::new_fallback(vocoder_config);

        // Audio configuration
        let audio_config = AudioConfig {
            sample_rate: config.s2mel.preprocess.sr,
            n_fft: config.s2mel.preprocess.n_fft,
            hop_length: config.s2mel.preprocess.hop_length,
            win_length: config.s2mel.preprocess.win_length,
            n_mels: config.s2mel.preprocess.n_mels,
            fmin: config.s2mel.preprocess.fmin,
            fmax: config.s2mel.preprocess.fmax,
        };

        log::info!("IndexTTS initialized successfully");

        Ok(Self {
            normalizer,
            tokenizer,
            speaker_encoder,
            emotion_encoder,
            semantic_encoder,
            vocoder,
            audio_config,
            config,
        })
    }

    /// Load from configuration file
    pub fn load<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let config = Config::load(config_path)?;
        Self::new(config)
    }

    /// Synthesize speech from text
    pub fn synthesize(
        &self,
        text: &str,
        speaker_audio_path: &str,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        let start_time = Instant::now();

        log::info!("Starting synthesis for: {}", &text[..text.len().min(50)]);

        // 1. Text normalization
        log::debug!("Normalizing text...");
        let normalized_text = self.normalizer.normalize(text)?;

        // 2. Tokenization
        log::debug!("Tokenizing text...");
        let tokens = self.tokenizer.encode(&normalized_text)?;
        log::debug!("Generated {} tokens", tokens.len());

        // 3. Load speaker audio
        log::debug!("Loading speaker audio...");
        let speaker_audio = load_audio(speaker_audio_path, Some(self.audio_config.sample_rate))?;

        // 4. Extract speaker embedding
        log::debug!("Extracting speaker embedding...");
        let mel_spec = crate::audio::mel_spectrogram(&speaker_audio.samples, &self.audio_config)?;
        let speaker_embedding = self.speaker_encoder.encode(&mel_spec)?;

        // 5. Extract semantic codes
        log::debug!("Extracting semantic codes...");
        let semantic_codes = self
            .semantic_encoder
            .encode(&speaker_audio.samples, self.audio_config.sample_rate)?;

        // 6. Prepare emotion conditioning
        log::debug!("Preparing emotion conditioning...");
        let emotion_embedding = if let Some(ref emo_vec) = options.emotion_vector {
            let emo = self.emotion_encoder.apply_strength(emo_vec, options.emotion_alpha);
            self.emotion_encoder.encode(&emo)?
        } else {
            let neutral = self.emotion_encoder.neutral();
            self.emotion_encoder.encode(&neutral)?
        };

        // 7. Generate mel tokens (simplified - directly create mel spectrogram)
        log::debug!("Generating mel spectrogram...");
        let mel_length = (tokens.len() as f32 * 2.5) as usize; // Approximate
        let mel_spec = self.generate_mel_spectrogram(
            &tokens,
            &semantic_codes,
            &speaker_embedding,
            &emotion_embedding,
            mel_length,
        )?;

        // 8. Vocoding
        log::debug!("Running vocoder...");
        let audio = self.vocoder.synthesize(&mel_spec)?;

        // 9. Post-processing
        log::debug!("Post-processing...");
        let audio = self.post_process(&audio);

        let processing_time = start_time.elapsed().as_secs_f32();
        let duration = audio.len() as f32 / self.vocoder.sample_rate() as f32;
        let rtf = processing_time / duration;

        log::info!(
            "Synthesis complete: {:.2}s audio in {:.2}s (RTF: {:.3})",
            duration,
            processing_time,
            rtf
        );

        Ok(SynthesisResult {
            audio,
            sample_rate: self.vocoder.sample_rate(),
            duration,
            processing_time,
            rtf,
        })
    }

    /// Synthesize and save to file
    pub fn synthesize_to_file(
        &self,
        text: &str,
        speaker_audio_path: &str,
        output_path: &str,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        let result = self.synthesize(text, speaker_audio_path, options)?;
        result.save(output_path)?;
        log::info!("Saved audio to: {}", output_path);
        Ok(result)
    }

    /// Generate mel spectrogram (simplified version)
    fn generate_mel_spectrogram(
        &self,
        _tokens: &[i64],
        _semantic_codes: &[i64],
        _speaker_embedding: &Array1<f32>,
        _emotion_embedding: &Array1<f32>,
        mel_length: usize,
    ) -> Result<ndarray::Array2<f32>> {
        // This is a placeholder - in production, would use the GPT model
        // For now, generate a simple mel spectrogram based on input characteristics

        use rand::Rng;
        let mut rng = rand::thread_rng();

        let n_mels = self.audio_config.n_mels;
        let mut mel = ndarray::Array2::zeros((n_mels, mel_length));

        // Generate synthetic mel spectrogram with some structure
        for t in 0..mel_length {
            for freq in 0..n_mels {
                // Create frequency-dependent pattern
                let base_value = -4.0 + (freq as f32 / n_mels as f32) * 2.0;
                let time_mod = ((t as f32 * 0.1).sin() + 1.0) * 0.5;
                let noise = rng.gen_range(-0.5..0.5);
                mel[[freq, t]] = base_value + time_mod + noise;
            }
        }

        Ok(mel)
    }

    /// Post-process audio
    fn post_process(&self, audio: &[f32]) -> Vec<f32> {
        use crate::audio::{normalize_audio_peak, apply_fade};

        // Normalize to -1dB peak
        let normalized = normalize_audio_peak(audio, 0.89);

        // Apply fade
        let fade_samples = (self.audio_config.sample_rate as f32 * 0.005) as usize; // 5ms
        apply_fade(&normalized, fade_samples, fade_samples)
    }

    /// Synthesize long text by splitting into segments
    pub fn synthesize_long(
        &self,
        text: &str,
        speaker_audio_path: &str,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        let start_time = Instant::now();

        // Segment text
        let segments = super::segment_text(text, 100);
        log::info!("Split text into {} segments", segments.len());

        // Synthesize each segment
        let mut audio_segments = Vec::new();
        for (i, segment) in segments.iter().enumerate() {
            log::info!("Synthesizing segment {}/{}", i + 1, segments.len());
            let result = self.synthesize(segment, speaker_audio_path, options)?;
            audio_segments.push(result.audio);
        }

        // Concatenate with silence
        let audio = super::concatenate_audio(
            &audio_segments,
            options.segment_silence_ms,
            self.vocoder.sample_rate(),
        );

        let processing_time = start_time.elapsed().as_secs_f32();
        let duration = audio.len() as f32 / self.vocoder.sample_rate() as f32;
        let rtf = processing_time / duration;

        Ok(SynthesisResult {
            audio,
            sample_rate: self.vocoder.sample_rate(),
            duration,
            processing_time,
            rtf,
        })
    }

    /// Get vocoder sample rate
    pub fn sample_rate(&self) -> u32 {
        self.vocoder.sample_rate()
    }

    /// Get configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.emotion_alpha, 1.0);
        assert!(matches!(options.sampling, SamplingStrategy::TopKP { .. }));
    }

    #[test]
    fn test_synthesis_result_duration() {
        let result = SynthesisResult {
            audio: vec![0.0; 22050 * 125], // 125 seconds
            sample_rate: 22050,
            duration: 125.0,
            processing_time: 10.0,
            rtf: 0.08,
        };

        assert_eq!(result.duration_formatted(), "02:05");
    }

    #[test]
    fn test_segment_text() {
        let text = "This is sentence one. This is sentence two. This is sentence three.";
        let segments = super::super::segment_text(text, 50);
        assert!(segments.len() >= 2);
    }

    #[test]
    fn test_concatenate_audio() {
        let seg1 = vec![1.0f32; 100];
        let seg2 = vec![2.0f32; 100];
        let result = super::super::concatenate_audio(&[seg1, seg2], 10, 1000);
        // Should have seg1 (100) + silence (10) + seg2 (100) = 210
        assert_eq!(result.len(), 210);
    }
}
