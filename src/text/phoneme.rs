//! Phoneme conversion for TTS
//!
//! Provides grapheme-to-phoneme (G2P) conversion for English
//! and Pinyin handling for Chinese

use crate::Result;
use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    /// English grapheme-to-phoneme dictionary (simplified)
    static ref G2P_DICT: HashMap<&'static str, Vec<&'static str>> = {
        let mut m = HashMap::new();
        // Common words - in production, this would be much larger
        m.insert("hello", vec!["HH", "AH0", "L", "OW1"]);
        m.insert("world", vec!["W", "ER1", "L", "D"]);
        m.insert("the", vec!["DH", "AH0"]);
        m.insert("a", vec!["AH0"]);
        m.insert("is", vec!["IH1", "Z"]);
        m.insert("to", vec!["T", "UW1"]);
        m.insert("and", vec!["AH0", "N", "D"]);
        m.insert("in", vec!["IH0", "N"]);
        m.insert("that", vec!["DH", "AE1", "T"]);
        m.insert("have", vec!["HH", "AE1", "V"]);
        m.insert("for", vec!["F", "AO1", "R"]);
        m.insert("not", vec!["N", "AA1", "T"]);
        m.insert("with", vec!["W", "IH1", "DH"]);
        m.insert("you", vec!["Y", "UW1"]);
        m.insert("this", vec!["DH", "IH1", "S"]);
        m.insert("but", vec!["B", "AH1", "T"]);
        m.insert("from", vec!["F", "R", "AH1", "M"]);
        m.insert("they", vec!["DH", "EY1"]);
        m.insert("we", vec!["W", "IY1"]);
        m.insert("say", vec!["S", "EY1"]);
        m.insert("she", vec!["SH", "IY1"]);
        m.insert("or", vec!["AO1", "R"]);
        m.insert("an", vec!["AE1", "N"]);
        m.insert("will", vec!["W", "IH1", "L"]);
        m.insert("my", vec!["M", "AY1"]);
        m.insert("one", vec!["W", "AH1", "N"]);
        m.insert("all", vec!["AO1", "L"]);
        m.insert("would", vec!["W", "UH1", "D"]);
        m.insert("there", vec!["DH", "EH1", "R"]);
        m.insert("their", vec!["DH", "EH1", "R"]);
        m
    };

    /// Pinyin to initial-final mapping
    static ref PINYIN_MAP: HashMap<&'static str, (&'static str, &'static str)> = {
        let mut m = HashMap::new();
        // Initial + Final decomposition
        m.insert("ba", ("b", "a"));
        m.insert("pa", ("p", "a"));
        m.insert("ma", ("m", "a"));
        m.insert("fa", ("f", "a"));
        m.insert("da", ("d", "a"));
        m.insert("ta", ("t", "a"));
        m.insert("na", ("n", "a"));
        m.insert("la", ("l", "a"));
        m.insert("ga", ("g", "a"));
        m.insert("ka", ("k", "a"));
        m.insert("ha", ("h", "a"));
        m.insert("zha", ("zh", "a"));
        m.insert("cha", ("ch", "a"));
        m.insert("sha", ("sh", "a"));
        m.insert("za", ("z", "a"));
        m.insert("ca", ("c", "a"));
        m.insert("sa", ("s", "a"));
        m.insert("ni", ("n", "i"));
        m.insert("hao", ("h", "ao"));
        m.insert("shi", ("sh", "i"));
        m.insert("jie", ("j", "ie"));
        m.insert("zhong", ("zh", "ong"));
        m.insert("guo", ("g", "uo"));
        m.insert("ren", ("r", "en"));
        m.insert("ming", ("m", "ing"));
        m.insert("de", ("d", "e"));
        m.insert("yi", ("", "i"));
        m.insert("er", ("", "er"));
        m.insert("san", ("s", "an"));
        m.insert("si", ("s", "i"));
        m.insert("wu", ("", "u"));
        m.insert("liu", ("l", "iu"));
        m.insert("qi", ("q", "i"));
        m.insert("jiu", ("j", "iu"));
        m
    };
}

/// Convert English word to phonemes using dictionary lookup
pub fn g2p_english(word: &str) -> Vec<String> {
    let lower = word.to_lowercase();

    if let Some(phones) = G2P_DICT.get(lower.as_str()) {
        phones.iter().map(|s| s.to_string()).collect()
    } else {
        // Fallback: spell out letters
        word.chars()
            .map(|c| c.to_uppercase().to_string())
            .collect()
    }
}

/// Convert text to phonemes
pub fn text_to_phonemes(text: &str) -> Vec<String> {
    let mut phonemes = Vec::new();

    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let clean_word: String = word
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect();

        if !clean_word.is_empty() {
            phonemes.extend(g2p_english(&clean_word));
        }

        // Add word boundary
        if i < words.len() - 1 {
            phonemes.push(" ".to_string());
        }
    }

    phonemes
}

/// Pinyin tone extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    First,  // ā
    Second, // á
    Third,  // ǎ
    Fourth, // à
    Neutral,
}

/// Extract tone from pinyin with tone marks
pub fn extract_tone(pinyin: &str) -> (String, Tone) {
    let tone_marks = [
        ('ā', 'a', Tone::First),
        ('á', 'a', Tone::Second),
        ('ǎ', 'a', Tone::Third),
        ('à', 'a', Tone::Fourth),
        ('ē', 'e', Tone::First),
        ('é', 'e', Tone::Second),
        ('ě', 'e', Tone::Third),
        ('è', 'e', Tone::Fourth),
        ('ī', 'i', Tone::First),
        ('í', 'i', Tone::Second),
        ('ǐ', 'i', Tone::Third),
        ('ì', 'i', Tone::Fourth),
        ('ō', 'o', Tone::First),
        ('ó', 'o', Tone::Second),
        ('ǒ', 'o', Tone::Third),
        ('ò', 'o', Tone::Fourth),
        ('ū', 'u', Tone::First),
        ('ú', 'u', Tone::Second),
        ('ǔ', 'u', Tone::Third),
        ('ù', 'u', Tone::Fourth),
        ('ǖ', 'ü', Tone::First),
        ('ǘ', 'ü', Tone::Second),
        ('ǚ', 'ü', Tone::Third),
        ('ǜ', 'ü', Tone::Fourth),
    ];

    let mut result = pinyin.to_string();
    let mut tone = Tone::Neutral;

    for (marked, plain, t) in tone_marks.iter() {
        if result.contains(*marked) {
            result = result.replace(*marked, &plain.to_string());
            tone = *t;
            break;
        }
    }

    // Check for numeric tone (e.g., "ma1")
    if let Some(last_char) = result.chars().last() {
        if last_char.is_ascii_digit() {
            let tone_num = last_char.to_digit(10).unwrap_or(5);
            tone = match tone_num {
                1 => Tone::First,
                2 => Tone::Second,
                3 => Tone::Third,
                4 => Tone::Fourth,
                _ => Tone::Neutral,
            };
            result.pop();
        }
    }

    (result, tone)
}

/// Convert pinyin to phonetic representation
pub fn pinyin_to_phones(pinyin: &str) -> Vec<String> {
    let (base, tone) = extract_tone(pinyin);
    let lower = base.to_lowercase();

    let mut phones = Vec::new();

    if let Some(&(initial, final_part)) = PINYIN_MAP.get(lower.as_str()) {
        if !initial.is_empty() {
            phones.push(initial.to_string());
        }
        phones.push(final_part.to_string());
    } else {
        // Fallback: return as-is
        phones.push(lower);
    }

    // Add tone marker
    let tone_str = match tone {
        Tone::First => "1",
        Tone::Second => "2",
        Tone::Third => "3",
        Tone::Fourth => "4",
        Tone::Neutral => "5",
    };
    phones.push(tone_str.to_string());

    phones
}

/// Convert Chinese character to pinyin (simplified)
pub fn char_to_pinyin(ch: char) -> Option<String> {
    // This is a simplified version
    // In production, would use a full pinyin dictionary
    let pinyin_map: HashMap<char, &str> = [
        ('你', "ni3"),
        ('好', "hao3"),
        ('世', "shi4"),
        ('界', "jie4"),
        ('中', "zhong1"),
        ('国', "guo2"),
        ('人', "ren2"),
        ('我', "wo3"),
        ('是', "shi4"),
        ('的', "de5"),
        ('了', "le5"),
        ('在', "zai4"),
        ('有', "you3"),
        ('个', "ge4"),
        ('这', "zhe4"),
        ('他', "ta1"),
        ('说', "shuo1"),
        ('来', "lai2"),
        ('要', "yao4"),
        ('就', "jiu4"),
        ('出', "chu1"),
        ('会', "hui4"),
        ('可', "ke3"),
        ('以', "yi3"),
        ('时', "shi2"),
        ('大', "da4"),
        ('看', "kan4"),
        ('地', "di4"),
        ('不', "bu4"),
        ('对', "dui4"),
    ]
    .iter()
    .cloned()
    .collect();

    pinyin_map.get(&ch).map(|s| s.to_string())
}

/// Segment Chinese text into words using jieba
pub fn segment_chinese(text: &str) -> Vec<String> {
    use jieba_rs::Jieba;

    let jieba = Jieba::new();
    let words = jieba.cut(text, false);
    words.into_iter().map(|s| s.to_string()).collect()
}

/// Convert Chinese text to pinyin sequence
pub fn chinese_to_pinyin(text: &str) -> Vec<String> {
    let mut pinyin_seq = Vec::new();

    for ch in text.chars() {
        if super::is_chinese_char(ch) {
            if let Some(py) = char_to_pinyin(ch) {
                pinyin_seq.push(py);
            } else {
                // Unknown character
                pinyin_seq.push(format!("_{}_", ch));
            }
        } else if !ch.is_whitespace() {
            pinyin_seq.push(ch.to_string());
        }
    }

    pinyin_seq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g2p_english() {
        let phones = g2p_english("hello");
        assert_eq!(phones, vec!["HH", "AH0", "L", "OW1"]);
    }

    #[test]
    fn test_g2p_unknown() {
        let phones = g2p_english("xyz");
        // Should spell out
        assert_eq!(phones, vec!["X", "Y", "Z"]);
    }

    #[test]
    fn test_extract_tone() {
        let (base, tone) = extract_tone("nǐ");
        assert_eq!(base, "ni");
        assert_eq!(tone, Tone::Third);

        let (base, tone) = extract_tone("hao3");
        assert_eq!(base, "hao");
        assert_eq!(tone, Tone::Third);
    }

    #[test]
    fn test_pinyin_to_phones() {
        let phones = pinyin_to_phones("hao3");
        assert!(phones.contains(&"h".to_string()));
        assert!(phones.contains(&"ao".to_string()));
        assert!(phones.contains(&"3".to_string()));
    }

    #[test]
    fn test_char_to_pinyin() {
        assert_eq!(char_to_pinyin('你'), Some("ni3".to_string()));
        assert_eq!(char_to_pinyin('好'), Some("hao3".to_string()));
    }

    #[test]
    fn test_segment_chinese() {
        let segments = segment_chinese("你好世界");
        assert!(segments.len() >= 2);
    }
}
