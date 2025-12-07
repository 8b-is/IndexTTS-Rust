//! Quality validation module using Marine salience
//!
//! Provides TTS output validation, prosody extraction, and conversation
//! affect tracking using the Marine algorithm.
//!
//! "Marines are not just jarheads - they are actually very intelligent"

pub mod prosody;
pub mod affect;

pub use prosody::{MarineProsodyConditioner, MarineProsodyVector};
pub use affect::{ComfortLevel, ConversationAffectSummary};
