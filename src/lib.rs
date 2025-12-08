//! IndexTTS - High-performance Text-to-Speech Engine in Pure Rust
//!
//! This is a Rust implementation of the IndexTTS system, providing
//! zero-shot multi-lingual text-to-speech synthesis with emotion control.
//!
//! # Features
//! - High-performance audio processing with SIMD optimizations
//! - Multi-language support (Chinese, English, mixed)
//! - Emotion control via vectors or text
//! - Speaker voice cloning from reference audio
//! - Efficient memory usage with zero-copy operations
//!
//! # Example
//! ```no_run
//! use indextts::{IndexTTS, Config};
//! use indextts::pipeline::SynthesisOptions;
//!
//! let config = Config::load("config.yaml").unwrap();
//! let tts = IndexTTS::new(config).unwrap();
//!
//! let options = SynthesisOptions::default();
//! tts.synthesize("Hello world", "speaker.wav", &options).unwrap();
//! ```

// Allow dead code for utility functions intended for future use or library consumers
#![allow(dead_code)]
// Allow traditional for loops - often clearer for audio DSP code
#![allow(clippy::needless_range_loop)]
// Allow enumeration discards in iterator patterns
#![allow(clippy::unused_enumerate_index)]
// Allow wildcard patterns in matches for exhaustive handling
#![allow(clippy::match_wildcard_for_single_variants)]

pub mod audio;
pub mod config;
pub mod error;
pub mod model;
pub mod pipeline;
pub mod quality;
pub mod text;
pub mod vocoder;

pub use config::Config;
pub use error::{Error, Result};
pub use pipeline::IndexTTS;

// Re-export Marine quality validation
pub use quality::{
    ComfortLevel, ConversationAffectSummary, MarineProsodyConditioner, MarineProsodyVector,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default sample rate for audio processing
pub const SAMPLE_RATE: u32 = 22050;

/// Default number of mel filterbank channels
pub const N_MELS: usize = 80;

/// Default FFT size
pub const N_FFT: usize = 1024;

/// Default hop length for STFT
pub const HOP_LENGTH: usize = 256;

/// Default window size
pub const WIN_LENGTH: usize = 1024;
