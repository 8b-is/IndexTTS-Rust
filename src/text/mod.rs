//! Text processing module for IndexTTS
//!
//! Provides text normalization, tokenization, and phoneme conversion.

mod normalizer;
mod phoneme;
mod tokenizer;

pub use normalizer::{Language, TextNormalizer};
pub use phoneme::{g2p_english, pinyin_to_phones};
pub use tokenizer::{TextTokenizer, TokenizerConfig};

use crate::Result;

/// Process text through the complete frontend pipeline
pub fn process_text(text: &str, tokenizer: &TextTokenizer) -> Result<Vec<i64>> {
    // Normalize text
    let normalizer = TextNormalizer::new();
    let normalized = normalizer.normalize(text)?;

    // Tokenize
    let tokens = tokenizer.encode(&normalized)?;

    Ok(tokens)
}

/// Detect language of text
pub fn detect_language(text: &str) -> Language {
    let mut chinese_count = 0;
    let mut english_count = 0;

    for ch in text.chars() {
        if is_chinese_char(ch) {
            chinese_count += 1;
        } else if ch.is_ascii_alphabetic() {
            english_count += 1;
        }
    }

    if chinese_count > 0 && english_count == 0 {
        Language::Chinese
    } else if english_count > 0 && chinese_count == 0 {
        Language::English
    } else if chinese_count > 0 && english_count > 0 {
        Language::Mixed
    } else {
        // Default to English for pure punctuation or empty
        Language::English
    }
}

/// Check if character is Chinese
pub fn is_chinese_char(ch: char) -> bool {
    matches!(ch as u32,
        0x4E00..=0x9FFF |     // CJK Unified Ideographs
        0x3400..=0x4DBF |     // CJK Unified Ideographs Extension A
        0x20000..=0x2A6DF |   // CJK Unified Ideographs Extension B
        0x2A700..=0x2B73F |   // CJK Unified Ideographs Extension C
        0x2B740..=0x2B81F |   // CJK Unified Ideographs Extension D
        0xF900..=0xFAFF |     // CJK Compatibility Ideographs
        0x2F800..=0x2FA1F     // CJK Compatibility Ideographs Supplement
    )
}

/// Check if text contains Chinese characters
pub fn contains_chinese(text: &str) -> bool {
    text.chars().any(is_chinese_char)
}

/// Check if text contains only ASCII
pub fn is_ascii_only(text: &str) -> bool {
    text.is_ascii()
}

/// Split text into segments by language
pub fn split_by_language(text: &str) -> Vec<(String, Language)> {
    let mut segments = Vec::new();
    let mut current_segment = String::new();
    let mut current_lang = None;

    for ch in text.chars() {
        let char_lang = if is_chinese_char(ch) {
            Some(Language::Chinese)
        } else if ch.is_ascii_alphabetic() {
            Some(Language::English)
        } else {
            None // Punctuation or other
        };

        match (current_lang, char_lang) {
            (None, Some(lang)) => {
                current_lang = Some(lang);
                current_segment.push(ch);
            }
            (Some(curr), Some(lang)) if curr == lang => {
                current_segment.push(ch);
            }
            (Some(curr), Some(lang)) if curr != lang => {
                if !current_segment.trim().is_empty() {
                    segments.push((current_segment.clone(), curr));
                }
                current_segment = ch.to_string();
                current_lang = Some(lang);
            }
            (Some(_), None) => {
                // Punctuation - add to current segment
                current_segment.push(ch);
            }
            (None, None) => {
                // Pure punctuation
                if !current_segment.is_empty() {
                    current_segment.push(ch);
                }
            }
            _ => {}
        }
    }

    if !current_segment.trim().is_empty() {
        if let Some(lang) = current_lang {
            segments.push((current_segment, lang));
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_chinese_char() {
        assert!(is_chinese_char('中'));
        assert!(is_chinese_char('文'));
        assert!(!is_chinese_char('a'));
        assert!(!is_chinese_char('1'));
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("Hello world"), Language::English);
        assert_eq!(detect_language("你好世界"), Language::Chinese);
        assert_eq!(detect_language("Hello 世界"), Language::Mixed);
    }

    #[test]
    fn test_contains_chinese() {
        assert!(contains_chinese("Hello 世界"));
        assert!(contains_chinese("你好"));
        assert!(!contains_chinese("Hello world"));
    }
}
