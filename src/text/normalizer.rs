//! Text normalization for TTS

use crate::Result;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Mixed,
}

#[derive(Debug)]
pub struct TextNormalizer {
    punct_map: HashMap<char, char>,
    number_words: HashMap<u64, &'static str>,
}

lazy_static! {
    static ref NUMBER_REGEX: Regex = Regex::new(r"\d+").unwrap();
    static ref WHITESPACE_REGEX: Regex = Regex::new(r"\s+").unwrap();
}

impl TextNormalizer {
    pub fn new() -> Self {
        let mut punct_map = HashMap::new();
        punct_map.insert('\u{FF0C}', ',');
        punct_map.insert('\u{3002}', '.');
        punct_map.insert('\u{FF01}', '!');
        punct_map.insert('\u{FF1F}', '?');
        punct_map.insert('\u{FF1B}', ';');
        punct_map.insert('\u{FF1A}', ':');
        punct_map.insert('\u{201C}', '\u{0022}');
        punct_map.insert('\u{201D}', '\u{0022}');
        punct_map.insert('\u{2018}', '\'');
        punct_map.insert('\u{2019}', '\'');

        let mut number_words = HashMap::new();
        number_words.insert(0, "zero");
        number_words.insert(1, "one");
        number_words.insert(2, "two");
        number_words.insert(3, "three");
        number_words.insert(4, "four");
        number_words.insert(5, "five");
        number_words.insert(6, "six");
        number_words.insert(7, "seven");
        number_words.insert(8, "eight");
        number_words.insert(9, "nine");
        number_words.insert(10, "ten");
        number_words.insert(20, "twenty");
        number_words.insert(30, "thirty");

        Self { punct_map, number_words }
    }

    pub fn normalize(&self, text: &str) -> Result<String> {
        let mut result = self.normalize_punctuation(text);
        result = self.normalize_whitespace(&result);
        Ok(result)
    }

    pub fn normalize_punctuation(&self, text: &str) -> String {
        text.chars()
            .map(|c| *self.punct_map.get(&c).unwrap_or(&c))
            .collect()
    }

    pub fn normalize_whitespace(&self, text: &str) -> String {
        WHITESPACE_REGEX.replace_all(text, " ").trim().to_string()
    }

    pub fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer() {
        let n = TextNormalizer::new();
        let r = n.normalize_whitespace("  a  b  ");
        assert_eq!(r.len(), 3);
    }
}
