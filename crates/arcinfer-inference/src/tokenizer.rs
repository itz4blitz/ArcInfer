/// Tokenization wrapper for all-MiniLM-L6-v2.
///
/// This module wraps HuggingFace's tokenizers crate to provide a simple
/// interface: text in, token IDs + attention mask out.
///
/// WHY A WRAPPER?
/// The tokenizers crate has a large API surface. We only need one path:
/// load from JSON file, encode a single sentence, extract IDs and mask.
/// Wrapping it gives us a clean interface and isolates the dependency.
///
/// The tokenizer uses WordPiece (same as BERT):
///   "unbelievable" → ["un", "##believe", "##able"]
/// Plus special tokens: [CLS] at start (ID 101), [SEP] at end (ID 102).

use tokenizers::Tokenizer;

/// The output of tokenization: token IDs and attention mask.
///
/// These two arrays are the inputs to the ONNX model.
/// - `token_ids`: Integer IDs from the WordPiece vocabulary
/// - `attention_mask`: 1 for real tokens, 0 for padding
pub struct EncodedInput {
    ids: Vec<i64>,
    mask: Vec<i64>,
}

impl EncodedInput {
    /// The token IDs as i64 (ONNX models expect i64 inputs).
    pub fn token_ids(&self) -> &[i64] {
        &self.ids
    }

    /// The attention mask as i64.
    pub fn attention_mask(&self) -> &[i64] {
        &self.mask
    }

    /// Number of tokens (including [CLS] and [SEP]).
    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Thin wrapper around the HuggingFace tokenizer.
///
/// Loads from a tokenizer.json file (exported from Python) and provides
/// a single `encode` method that returns token IDs + attention mask.
pub struct SentenceTokenizer {
    inner: Tokenizer,
}

impl SentenceTokenizer {
    /// Load from a tokenizer.json file.
    ///
    /// Disables padding and truncation — we handle sequence length
    /// ourselves when building tensors for tract. Padding would waste
    /// computation inside the ONNX model.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut inner = Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer from {}: {}", path, e))?;
        inner.with_padding(None);
        inner.with_truncation(None).ok();
        Ok(Self { inner })
    }

    /// Tokenize a sentence. Returns token IDs and attention mask.
    ///
    /// The tokenizer automatically adds [CLS] (101) at the start and
    /// [SEP] (102) at the end, as required by BERT-based models.
    pub fn encode(&self, text: &str) -> EncodedInput {
        let encoding = self.inner.encode(text, true)
            .expect("Tokenization should not fail for valid UTF-8");

        EncodedInput {
            ids: encoding.get_ids().iter().map(|&id| id as i64).collect(),
            mask: encoding.get_attention_mask().iter().map(|&m| m as i64).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the downloaded tokenizer.json
    fn tokenizer_path() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/tokenizer.json")
    }

    // =========================================================================
    // TEST GROUP 1: Loading the tokenizer
    // =========================================================================

    #[test]
    fn test_load_tokenizer() {
        // Should load without error from our downloaded tokenizer.json
        let tok = SentenceTokenizer::from_file(tokenizer_path());
        assert!(tok.is_ok(), "Failed to load tokenizer: {:?}", tok.err());
    }

    #[test]
    fn test_load_tokenizer_bad_path() {
        let tok = SentenceTokenizer::from_file("/nonexistent/tokenizer.json");
        assert!(tok.is_err());
    }

    // =========================================================================
    // TEST GROUP 2: Basic tokenization
    // =========================================================================
    //
    // all-MiniLM-L6-v2 uses WordPiece tokenization with [CLS] and [SEP].
    // Token 101 = [CLS], Token 102 = [SEP].

    #[test]
    fn test_encode_simple_sentence() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let encoded = tok.encode("Hello world");

        // Should have token IDs: [CLS] Hello world [SEP]
        let ids = encoded.token_ids();
        assert!(ids.len() >= 3, "Expected at least [CLS] + tokens + [SEP]");
        assert_eq!(ids[0], 101, "First token should be [CLS] (101)");
        assert_eq!(*ids.last().unwrap(), 102, "Last token should be [SEP] (102)");
    }

    #[test]
    fn test_encode_returns_attention_mask() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let encoded = tok.encode("This is a test");

        let mask = encoded.attention_mask();
        // All tokens should have mask=1 (no padding)
        assert!(mask.iter().all(|&m| m == 1));
        // Mask length should match token IDs length
        assert_eq!(mask.len(), encoded.token_ids().len());
    }

    #[test]
    fn test_encode_empty_string() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let encoded = tok.encode("");

        // Even empty string should get [CLS] and [SEP]
        let ids = encoded.token_ids();
        assert!(ids.len() >= 2);
        assert_eq!(ids[0], 101);
        assert_eq!(*ids.last().unwrap(), 102);
    }

    // =========================================================================
    // TEST GROUP 3: Token ID consistency
    // =========================================================================
    //
    // Same input should always produce the same tokens (deterministic).
    // Different inputs should produce different tokens.

    #[test]
    fn test_encode_deterministic() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let enc1 = tok.encode("The movie was great");
        let enc2 = tok.encode("The movie was great");

        assert_eq!(enc1.token_ids(), enc2.token_ids());
        assert_eq!(enc1.attention_mask(), enc2.attention_mask());
    }

    #[test]
    fn test_different_inputs_different_tokens() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let enc_pos = tok.encode("I love this movie");
        let enc_neg = tok.encode("I hate this movie");

        // "love" and "hate" should produce different token IDs
        assert_ne!(enc_pos.token_ids(), enc_neg.token_ids());
    }

    // =========================================================================
    // TEST GROUP 4: Sequence length properties
    // =========================================================================

    #[test]
    fn test_longer_input_more_tokens() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let short = tok.encode("Good");
        let long = tok.encode("This movie was absolutely wonderful and I loved every minute of it");

        assert!(
            long.token_ids().len() > short.token_ids().len(),
            "Longer input should produce more tokens"
        );
    }

    #[test]
    fn test_subword_tokenization() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        // "unconstitutionally" is long enough to require subword splitting
        let encoded = tok.encode("unconstitutionally");

        // WordPiece splits into subwords: [CLS] un ##con ##stitution ##ally [SEP] or similar
        // Should be more than 3 tokens (CLS + at least 2 subwords + SEP)
        assert!(
            encoded.token_ids().len() >= 4,
            "Expected subword tokenization, got {} tokens",
            encoded.token_ids().len()
        );
    }
}
