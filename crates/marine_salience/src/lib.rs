//! # Marine Salience - O(1) Jitter-Based Authenticity Detection
//!
//! "Marines are not just jarheads - they are actually very intelligent"
//!
//! This crate provides a universal salience primitive that can detect the
//! "authenticity" of audio signals by measuring timing and amplitude jitter.
//!
//! ## Why "Marine"?
//! - Marines are stable and reliable under pressure
//! - Low jitter = authentic/stable signal
//! - High jitter = damaged/synthetic signal
//!
//! ## Use Cases
//! - **TTS Quality Validation** - Is synthesized speech authentic?
//! - **Prosody Extraction** - Extract 8D interpretable emotion vectors
//! - **Conversation Affect** - Track comfort level over sessions
//! - **Real-time Monitoring** - O(1) per sample processing
//!
//! ## Core Insight
//! Human voice has NATURAL jitter patterns. Perfect smoothness = synthetic.
//! The Marine algorithm detects these patterns to distinguish authentic
//! from damaged or artificial audio.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod config;
pub mod ema;
pub mod packet;
pub mod processor;

// Re-export main types
pub use config::MarineConfig;
pub use packet::SaliencePacket;
pub use processor::MarineProcessor;

/// Marine algorithm version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default jitter thresholds tuned for speech
/// These values accommodate natural musical/speech variation
pub const DEFAULT_JITTER_LOW: f32 = 0.02;  // Below = very stable
pub const DEFAULT_JITTER_HIGH: f32 = 0.60; // Above = heavily damaged
