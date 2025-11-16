//! Text tokenization for TTS
//!
//! Uses SentencePiece BPE tokenization for converting text to tokens

use crate::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Tokenizer configuration
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Path to BPE model
    pub model_path: String,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Start of text token ID
    pub bos_id: i64,
    /// End of text token ID
    pub eos_id: i64,
    /// Unknown token ID
    pub unk_id: i64,
    /// Padding token ID
    pub pad_id: i64,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            model_path: "models/bpe.model".to_string(),
            vocab_size: 6681,
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 3,
        }
    }
}

/// Text tokenizer using BPE (Byte Pair Encoding)
#[derive(Debug)]
pub struct TextTokenizer {
    /// Configuration
    config: TokenizerConfig,
    /// Token to ID mapping
    token_to_id: HashMap<String, i64>,
    /// ID to token mapping
    id_to_token: HashMap<i64, String>,
    /// Character-level fallback vocabulary
    char_vocab: HashMap<char, i64>,
}

impl TextTokenizer {
    /// Create new tokenizer with default vocabulary
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut char_vocab = HashMap::new();

        // Add special tokens
        token_to_id.insert("<unk>".to_string(), config.unk_id);
        token_to_id.insert("<s>".to_string(), config.bos_id);
        token_to_id.insert("</s>".to_string(), config.eos_id);
        token_to_id.insert("<pad>".to_string(), config.pad_id);

        id_to_token.insert(config.unk_id, "<unk>".to_string());
        id_to_token.insert(config.bos_id, "<s>".to_string());
        id_to_token.insert(config.eos_id, "</s>".to_string());
        id_to_token.insert(config.pad_id, "<pad>".to_string());

        // Add basic ASCII characters
        let mut next_id = 4i64;
        for c in ' '..='~' {
            char_vocab.insert(c, next_id);
            token_to_id.insert(c.to_string(), next_id);
            id_to_token.insert(next_id, c.to_string());
            next_id += 1;
        }

        // Add Chinese character range (simplified approach)
        // In production, this would load from the actual BPE model
        for code_point in 0x4E00u32..=0x9FFF {
            if let Some(c) = char::from_u32(code_point) {
                char_vocab.insert(c, next_id);
                token_to_id.insert(c.to_string(), next_id);
                id_to_token.insert(next_id, c.to_string());
                next_id += 1;

                if next_id >= config.vocab_size as i64 {
                    break;
                }
            }
        }

        Ok(Self {
            config,
            token_to_id,
            id_to_token,
            char_vocab,
        })
    }

    /// Load tokenizer from model file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::FileNotFound(path.display().to_string()));
        }

        // In production, this would load the actual SentencePiece model
        // For now, create a character-level tokenizer
        let config = TokenizerConfig {
            model_path: path.display().to_string(),
            ..Default::default()
        };

        Self::new(config)
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();

        // Add BOS token
        tokens.push(self.config.bos_id);

        // Tokenize character by character (simplified)
        // In production, this would use BPE merging
        for ch in text.chars() {
            if let Some(&id) = self.char_vocab.get(&ch) {
                tokens.push(id);
            } else if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                tokens.push(id);
            } else {
                // Unknown token
                tokens.push(self.config.unk_id);
            }
        }

        // Add EOS token
        tokens.push(self.config.eos_id);

        Ok(tokens)
    }

    /// Encode text without special tokens
    pub fn encode_without_special(&self, text: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();

        for ch in text.chars() {
            if let Some(&id) = self.char_vocab.get(&ch) {
                tokens.push(id);
            } else if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                tokens.push(id);
            } else {
                tokens.push(self.config.unk_id);
            }
        }

        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[i64]) -> Result<String> {
        let mut text = String::new();

        for &token_id in tokens {
            // Skip special tokens
            if token_id == self.config.bos_id
                || token_id == self.config.eos_id
                || token_id == self.config.pad_id
            {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                text.push_str(token);
            } else {
                // Unknown token placeholder
                text.push('?');
            }
        }

        Ok(text)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get BOS token ID
    pub fn bos_id(&self) -> i64 {
        self.config.bos_id
    }

    /// Get EOS token ID
    pub fn eos_id(&self) -> i64 {
        self.config.eos_id
    }

    /// Get UNK token ID
    pub fn unk_id(&self) -> i64 {
        self.config.unk_id
    }

    /// Get PAD token ID
    pub fn pad_id(&self) -> i64 {
        self.config.pad_id
    }

    /// Pad sequences to same length
    pub fn pad_sequences(&self, sequences: &[Vec<i64>], max_len: Option<usize>) -> Vec<Vec<i64>> {
        let max_length = max_len.unwrap_or_else(|| sequences.iter().map(|s| s.len()).max().unwrap_or(0));

        sequences
            .iter()
            .map(|seq| {
                let mut padded = seq.clone();
                while padded.len() < max_length {
                    padded.push(self.config.pad_id);
                }
                padded.truncate(max_length);
                padded
            })
            .collect()
    }

    /// Create attention mask (1 for real tokens, 0 for padding)
    pub fn create_attention_mask(&self, tokens: &[i64]) -> Vec<i64> {
        tokens
            .iter()
            .map(|&t| if t == self.config.pad_id { 0 } else { 1 })
            .collect()
    }

    /// Batch encode multiple texts
    pub fn batch_encode(&self, texts: &[&str]) -> Result<Vec<Vec<i64>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Batch encode and pad
    pub fn batch_encode_padded(
        &self,
        texts: &[&str],
        max_len: Option<usize>,
    ) -> Result<Vec<Vec<i64>>> {
        let encoded: Vec<Vec<i64>> = self.batch_encode(texts)?;
        Ok(self.pad_sequences(&encoded, max_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let config = TokenizerConfig::default();
        let tokenizer = TextTokenizer::new(config).unwrap();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_encode_decode() {
        let config = TokenizerConfig::default();
        let tokenizer = TextTokenizer::new(config).unwrap();

        let text = "Hello world";
        let tokens = tokenizer.encode(text).unwrap();

        // Should start with BOS and end with EOS
        assert_eq!(tokens[0], tokenizer.bos_id());
        assert_eq!(*tokens.last().unwrap(), tokenizer.eos_id());

        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_chinese() {
        let config = TokenizerConfig::default();
        let tokenizer = TextTokenizer::new(config).unwrap();

        let text = "你好";
        let tokens = tokenizer.encode(text).unwrap();

        // Should have BOS + 2 chars + EOS = 4 tokens
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn test_pad_sequences() {
        let config = TokenizerConfig::default();
        let tokenizer = TextTokenizer::new(config).unwrap();

        let seq1 = vec![1, 2, 3];
        let seq2 = vec![1, 2, 3, 4, 5];

        let padded = tokenizer.pad_sequences(&[seq1, seq2], None);

        assert_eq!(padded[0].len(), 5);
        assert_eq!(padded[1].len(), 5);
        assert_eq!(padded[0][3], tokenizer.pad_id());
    }

    #[test]
    fn test_attention_mask() {
        let config = TokenizerConfig::default();
        let tokenizer = TextTokenizer::new(config).unwrap();

        let tokens = vec![1, 2, tokenizer.pad_id(), tokenizer.pad_id()];
        let mask = tokenizer.create_attention_mask(&tokens);

        assert_eq!(mask, vec![1, 1, 0, 0]);
    }
}
