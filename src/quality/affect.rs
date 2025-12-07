//! Conversation Affect Tracking - Session-level comfort analysis
//!
//! After a conversation, Aye can determine: "This felt uneasy / ok / happy"
//! based on Marine prosody patterns over time.
//!
//! The key insight: jitter patterns reveal emotional state
//! - Rising jitter over conversation = increasing tension
//! - Stable low jitter = calm exchange
//! - High energy + low jitter = positive/confident

use super::prosody::MarineProsodyVector;

/// Comfort level classification
///
/// After a conversation, this represents the overall emotional tone.
/// Used by Aye to self-assess: "How did I make you feel?"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComfortLevel {
    /// High jitter AND rising over session - tension/nervousness
    Uneasy,
    /// Stable but low energy, or mildly jittery but not escalating
    Neutral,
    /// Good energy, low/stable jitter - positive interaction
    Happy,
}

impl ComfortLevel {
    /// Convert to emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            ComfortLevel::Uneasy => "😟",
            ComfortLevel::Neutral => "😐",
            ComfortLevel::Happy => "😊",
        }
    }

    /// Convert to descriptive string
    pub fn description(&self) -> &'static str {
        match self {
            ComfortLevel::Uneasy => "uneasy or tense",
            ComfortLevel::Neutral => "neutral or stable",
            ComfortLevel::Happy => "comfortable and positive",
        }
    }

    /// Convert to numeric score (-1 = uneasy, 0 = neutral, 1 = happy)
    pub fn score(&self) -> i8 {
        match self {
            ComfortLevel::Uneasy => -1,
            ComfortLevel::Neutral => 0,
            ComfortLevel::Happy => 1,
        }
    }
}

/// Conversation affect summary
///
/// Aggregates Marine prosody data over an entire conversation to
/// provide session-level emotional assessment.
#[derive(Debug, Clone)]
pub struct ConversationAffectSummary {
    /// Comfort level of the human speaker (if analyzed)
    pub human_state: Option<ComfortLevel>,
    /// Comfort level of Aye's output
    pub aye_state: ComfortLevel,
    /// Overall audio/structure quality (0..1)
    pub quality_score: f32,
    /// Number of utterances analyzed
    pub utterance_count: usize,
    /// Session duration in seconds
    pub duration_seconds: f32,
    /// Mean prosody statistics
    pub mean_prosody: MarineProsodyVector,
    /// Jitter trend (positive = rising, negative = falling)
    pub jitter_trend: f32,
    /// Energy trend (positive = rising, negative = falling)
    pub energy_trend: f32,
}

impl ConversationAffectSummary {
    /// Generate Aye's self-assessment message
    pub fn aye_assessment(&self) -> String {
        let emoji = self.aye_state.emoji();
        let desc = self.aye_state.description();

        let quality_desc = if self.quality_score > 0.8 {
            "very good"
        } else if self.quality_score > 0.6 {
            "good"
        } else if self.quality_score > 0.4 {
            "moderate"
        } else {
            "low"
        };

        format!(
            "{} Aye thinks this conversation felt {}. Audio quality was {} ({:.0}%). \
             {} {} utterances over {:.1} seconds.",
            emoji,
            desc,
            quality_desc,
            self.quality_score * 100.0,
            if self.jitter_trend > 0.05 {
                "Tension seemed to increase."
            } else if self.jitter_trend < -0.05 {
                "Tension seemed to decrease."
            } else {
                "Emotional tone stayed consistent."
            },
            self.utterance_count,
            self.duration_seconds
        )
    }

    /// Generate prompt for asking human for feedback
    pub fn feedback_prompt(&self) -> String {
        format!(
            "Aye would like to improve. How did this conversation make you feel?\n\
             A) Uneasy or tense 😟\n\
             B) Neutral or okay 😐\n\
             C) Comfortable and positive 😊\n\n\
             Aye's self-assessment: {} ({})",
            self.aye_state.emoji(),
            self.aye_state.description()
        )
    }
}

/// Conversation affect analyzer
///
/// Collects prosody vectors over a conversation and computes
/// session-level emotional state.
pub struct ConversationAffectAnalyzer {
    /// Collected prosody vectors
    utterances: Vec<MarineProsodyVector>,
    /// Total audio duration
    total_duration_seconds: f32,
    /// Configuration thresholds
    config: AffectAnalyzerConfig,
}

/// Configuration for affect classification
#[derive(Debug, Clone, Copy)]
pub struct AffectAnalyzerConfig {
    /// Threshold for "high" combined jitter
    pub high_jitter_threshold: f32,
    /// Threshold for "rising" jitter trend
    pub rising_jitter_threshold: f32,
    /// Threshold for "high" energy (happy indicator)
    pub high_energy_threshold: f32,
}

impl Default for AffectAnalyzerConfig {
    fn default() -> Self {
        Self {
            high_jitter_threshold: 0.4,
            rising_jitter_threshold: 0.1,
            high_energy_threshold: 0.5,
        }
    }
}

impl ConversationAffectAnalyzer {
    /// Create new analyzer with default config
    pub fn new() -> Self {
        Self {
            utterances: Vec::new(),
            total_duration_seconds: 0.0,
            config: AffectAnalyzerConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AffectAnalyzerConfig) -> Self {
        Self {
            utterances: Vec::new(),
            total_duration_seconds: 0.0,
            config,
        }
    }

    /// Add an utterance's prosody to the conversation
    pub fn add_utterance(&mut self, prosody: MarineProsodyVector, duration_seconds: f32) {
        self.utterances.push(prosody);
        self.total_duration_seconds += duration_seconds;
    }

    /// Reset analyzer for new conversation
    pub fn reset(&mut self) {
        self.utterances.clear();
        self.total_duration_seconds = 0.0;
    }

    /// Analyze conversation and produce affect summary
    pub fn analyze(&self) -> Option<ConversationAffectSummary> {
        if self.utterances.is_empty() {
            return None;
        }

        let n = self.utterances.len() as f32;

        // Calculate mean prosody
        let mut mean_prosody = MarineProsodyVector::zeros();
        for p in &self.utterances {
            mean_prosody.jp_mean += p.jp_mean;
            mean_prosody.jp_std += p.jp_std;
            mean_prosody.ja_mean += p.ja_mean;
            mean_prosody.ja_std += p.ja_std;
            mean_prosody.h_mean += p.h_mean;
            mean_prosody.s_mean += p.s_mean;
            mean_prosody.peak_density += p.peak_density;
            mean_prosody.energy_mean += p.energy_mean;
        }
        mean_prosody.jp_mean /= n;
        mean_prosody.jp_std /= n;
        mean_prosody.ja_mean /= n;
        mean_prosody.ja_std /= n;
        mean_prosody.h_mean /= n;
        mean_prosody.s_mean /= n;
        mean_prosody.peak_density /= n;
        mean_prosody.energy_mean /= n;

        // Calculate trends (first vs last)
        let jitter_trend = if self.utterances.len() >= 2 {
            let first = self.utterances.first().unwrap().combined_jitter();
            let last = self.utterances.last().unwrap().combined_jitter();
            last - first
        } else {
            0.0
        };

        let energy_trend = if self.utterances.len() >= 2 {
            let first = self.utterances.first().unwrap().energy_mean;
            let last = self.utterances.last().unwrap().energy_mean;
            last - first
        } else {
            0.0
        };

        // Classify comfort level
        let aye_state = self.classify_comfort(
            mean_prosody.combined_jitter(),
            jitter_trend,
            mean_prosody.energy_mean,
        );

        let quality_score = mean_prosody.s_mean;

        Some(ConversationAffectSummary {
            human_state: None, // Would require analyzing human audio
            aye_state,
            quality_score,
            utterance_count: self.utterances.len(),
            duration_seconds: self.total_duration_seconds,
            mean_prosody,
            jitter_trend,
            energy_trend,
        })
    }

    /// Classify comfort level based on jitter and energy patterns
    fn classify_comfort(
        &self,
        mean_jitter: f32,
        trend_jitter: f32,
        mean_energy: f32,
    ) -> ComfortLevel {
        let high_jitter = mean_jitter > self.config.high_jitter_threshold;
        let rising_jitter = trend_jitter > self.config.rising_jitter_threshold;

        if high_jitter && rising_jitter {
            // Jitter is high AND getting worse = tension/unease
            ComfortLevel::Uneasy
        } else if mean_energy > self.config.high_energy_threshold && !high_jitter {
            // Good energy with stable jitter = positive/happy
            ComfortLevel::Happy
        } else {
            // In-between: stable but low energy, or slightly jittery but stable
            ComfortLevel::Neutral
        }
    }

    /// Get number of utterances collected
    pub fn utterance_count(&self) -> usize {
        self.utterances.len()
    }

    /// Get total duration
    pub fn total_duration(&self) -> f32 {
        self.total_duration_seconds
    }
}

impl Default for ConversationAffectAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comfort_level_descriptions() {
        assert_eq!(ComfortLevel::Uneasy.emoji(), "😟");
        assert_eq!(ComfortLevel::Neutral.emoji(), "😐");
        assert_eq!(ComfortLevel::Happy.emoji(), "😊");

        assert_eq!(ComfortLevel::Uneasy.score(), -1);
        assert_eq!(ComfortLevel::Neutral.score(), 0);
        assert_eq!(ComfortLevel::Happy.score(), 1);
    }

    #[test]
    fn test_analyzer_empty_conversation() {
        let analyzer = ConversationAffectAnalyzer::new();
        assert!(analyzer.analyze().is_none());
    }

    #[test]
    fn test_analyzer_single_utterance() {
        let mut analyzer = ConversationAffectAnalyzer::new();
        let prosody = MarineProsodyVector {
            jp_mean: 0.1,
            jp_std: 0.05,
            ja_mean: 0.1,
            ja_std: 0.05,
            h_mean: 1.0,
            s_mean: 0.8,
            peak_density: 50.0,
            energy_mean: 0.6,
        };
        analyzer.add_utterance(prosody, 2.0);

        let summary = analyzer.analyze().unwrap();
        assert_eq!(summary.utterance_count, 1);
        assert_eq!(summary.duration_seconds, 2.0);
    }

    #[test]
    fn test_uneasy_classification() {
        let mut analyzer = ConversationAffectAnalyzer::new();

        // First utterance: moderate jitter
        analyzer.add_utterance(
            MarineProsodyVector {
                jp_mean: 0.3,
                jp_std: 0.1,
                ja_mean: 0.3,
                ja_std: 0.1,
                h_mean: 1.0,
                s_mean: 0.5,
                peak_density: 50.0,
                energy_mean: 0.3,
            },
            1.0,
        );

        // Second utterance: HIGH jitter (rising trend)
        analyzer.add_utterance(
            MarineProsodyVector {
                jp_mean: 0.6,
                jp_std: 0.2,
                ja_mean: 0.5,
                ja_std: 0.2,
                h_mean: 0.8,
                s_mean: 0.3,
                peak_density: 60.0,
                energy_mean: 0.4,
            },
            1.0,
        );

        let summary = analyzer.analyze().unwrap();
        assert_eq!(summary.aye_state, ComfortLevel::Uneasy);
        assert!(summary.jitter_trend > 0.0); // Rising jitter
    }

    #[test]
    fn test_happy_classification() {
        let mut analyzer = ConversationAffectAnalyzer::new();

        // High energy, low jitter = happy
        analyzer.add_utterance(
            MarineProsodyVector {
                jp_mean: 0.1,
                jp_std: 0.05,
                ja_mean: 0.1,
                ja_std: 0.05,
                h_mean: 1.0,
                s_mean: 0.9,
                peak_density: 80.0,
                energy_mean: 0.7,
            },
            2.0,
        );

        let summary = analyzer.analyze().unwrap();
        assert_eq!(summary.aye_state, ComfortLevel::Happy);
    }

    #[test]
    fn test_neutral_classification() {
        let mut analyzer = ConversationAffectAnalyzer::new();

        // Low energy, moderate jitter = neutral
        analyzer.add_utterance(
            MarineProsodyVector {
                jp_mean: 0.2,
                jp_std: 0.1,
                ja_mean: 0.2,
                ja_std: 0.1,
                h_mean: 1.0,
                s_mean: 0.7,
                peak_density: 40.0,
                energy_mean: 0.3,
            },
            1.5,
        );

        let summary = analyzer.analyze().unwrap();
        assert_eq!(summary.aye_state, ComfortLevel::Neutral);
    }

    #[test]
    fn test_aye_assessment_message() {
        let summary = ConversationAffectSummary {
            human_state: None,
            aye_state: ComfortLevel::Happy,
            quality_score: 0.85,
            utterance_count: 5,
            duration_seconds: 30.0,
            mean_prosody: MarineProsodyVector::zeros(),
            jitter_trend: -0.1,
            energy_trend: 0.2,
        };

        let message = summary.aye_assessment();
        assert!(message.contains("😊"));
        assert!(message.contains("comfortable"));
        assert!(message.contains("85%"));
        assert!(message.contains("5 utterances"));
    }
}
