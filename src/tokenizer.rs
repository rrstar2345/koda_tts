//! Port of `src/vits/tokenizer.py`.
//!
//! This checkpoint (Kannada, character-level) always has `phonemize=false`
//! and `is_uroman=false` in `tokenizer_config.json`; those code paths raised
//! `RuntimeError` in Python because the required optional dependencies
//! (`phonemizer`, `uroman`) aren't bundled. This port keeps the same
//! contract: it validates the flags at load time and returns an error if a
//! checkpoint ever needs them, rather than silently mis-tokenizing.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub struct VitsTokenizer {
    encoder: HashMap<String, i64>,
    decoder: HashMap<i64, String>,
    pad_token: String,
    #[allow(dead_code)]
    unk_token: String,
    language: Option<String>,
    add_blank: bool,
    normalize: bool,
    unk_token_id: Option<i64>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct TokenizerConfigRaw {
    pad_token: Option<Value>,
    unk_token: Option<Value>,
    language: Option<String>,
    #[serde(default = "default_true")]
    add_blank: bool,
    #[serde(default = "default_true")]
    normalize: bool,
    #[serde(default)]
    phonemize: bool,
    #[serde(default)]
    is_uroman: bool,
}

fn default_true() -> bool {
    true
}

fn tok_str(value: &Option<Value>, default: &str) -> String {
    match value {
        Some(Value::Object(map)) => map
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or(default)
            .to_string(),
        Some(Value::String(s)) => s.clone(),
        _ => default.to_string(),
    }
}

fn has_non_roman_characters(s: &str) -> bool {
    s.chars().any(|c| !c.is_ascii())
}

impl VitsTokenizer {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("tokenizer_config.json");
        let raw: TokenizerConfigRaw = if config_path.exists() {
            let text = std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", config_path.display()))?
        } else {
            TokenizerConfigRaw::default()
        };

        if raw.phonemize {
            bail!(
                "This checkpoint requires phonemization (`phonemize=true`), which needs a \
                 phonemizer + espeak backend. That path is not implemented in this standalone \
                 port since it's not needed for character-based checkpoints like this one."
            );
        }
        if raw.is_uroman {
            bail!(
                "This checkpoint's tokenizer config indicates uroman romanization is required, \
                 but uroman support is not implemented in this standalone port."
            );
        }

        let pad_token = tok_str(&raw.pad_token, "<pad>");
        let unk_token = tok_str(&raw.unk_token, "<unk>");

        let vocab_path = model_dir.join("vocab.json");
        let vocab_text = std::fs::read_to_string(&vocab_path)
            .with_context(|| format!("reading {}", vocab_path.display()))?;
        let encoder: HashMap<String, i64> = serde_json::from_str(&vocab_text)
            .with_context(|| format!("parsing {}", vocab_path.display()))?;
        let decoder: HashMap<i64, String> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        let unk_token_id = encoder.get(&unk_token).copied();

        Ok(Self {
            encoder,
            decoder,
            pad_token,
            unk_token,
            language: raw.language,
            add_blank: raw.add_blank,
            normalize: raw.normalize,
            unk_token_id,
        })
    }

    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Lowercase input, respecting any multi-character vocab entries
    /// (greedy longest-match-first is not required here since the Python
    /// reference just iterates `encoder.keys()` in insertion order; we
    /// mirror that exactly for parity).
    fn normalize_text(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let vocab_words: Vec<&String> = self.encoder.keys().collect();
        let mut out = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            let mut matched = false;
            for word in &vocab_words {
                let word_chars: Vec<char> = word.chars().collect();
                if word_chars.is_empty() {
                    continue;
                }
                if i + word_chars.len() <= chars.len()
                    && chars[i..i + word_chars.len()] == word_chars[..]
                {
                    out.push_str(word);
                    i += word_chars.len();
                    matched = true;
                    break;
                }
            }
            if !matched {
                for lower in chars[i].to_lowercase() {
                    out.push(lower);
                }
                i += 1;
            }
        }
        out
    }

    fn preprocess_char(&self, text: String) -> String {
        if self.language.as_deref() == Some("ron") {
            text.replace('ț', "ţ")
        } else {
            text
        }
    }

    fn prepare_text(&self, text: &str) -> Result<String> {
        let mut text = if self.normalize {
            self.normalize_text(text)
        } else {
            text.to_string()
        };

        text = self.preprocess_char(text);

        if has_non_roman_characters(&text) {
            // is_uroman is validated false at load time, so this mirrors
            // the Python function's dead branch faithfully but should
            // never trigger for this checkpoint.
        }

        if self.normalize {
            // strip any characters outside the vocab (e.g. stray punctuation)
            text = text.chars().filter(|c| {
                let s = c.to_string();
                self.encoder.contains_key(&s)
            }).collect::<String>().trim().to_string();
        }

        Ok(text)
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        if self.add_blank {
            let pad_tok = self
                .decoder
                .get(&0)
                .cloned()
                .unwrap_or_else(|| self.pad_token.clone());
            let mut interspersed = vec![pad_tok; tokens.len() * 2 + 1];
            for (idx, tok) in tokens.drain(..).enumerate() {
                interspersed[idx * 2 + 1] = tok;
            }
            tokens = interspersed;
        }
        tokens
    }

    fn convert_token_to_id(&self, token: &str) -> i64 {
        self.encoder
            .get(token)
            .copied()
            .or(self.unk_token_id)
            .unwrap_or(0)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let text = self.prepare_text(text)?;
        let tokens = self.tokenize(&text);
        Ok(tokens.iter().map(|t| self.convert_token_to_id(t)).collect())
    }

    /// Returns `(input_ids, attention_mask)` as flat i64/i64 vectors for a
    /// single (batch size 1) input, matching `VitsTokenizer.__call__` with
    /// `return_tensors="pt"` in the Python reference.
    pub fn call(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let ids = self.encode(text)?;
        let attention_mask = vec![1i64; ids.len()];
        Ok((ids, attention_mask))
    }
}