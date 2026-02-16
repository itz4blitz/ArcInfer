/// ONNX-based sentence embedding extraction using all-MiniLM-L6-v2.
///
/// This module loads the ONNX model via tract-onnx and runs inference
/// to produce 384-dimensional sentence embeddings.
///
/// THE MEAN POOLING STEP:
/// The transformer outputs per-token embeddings: [batch, seq_len, 384].
/// To get a single sentence embedding, we average the token embeddings,
/// weighted by the attention mask (so padding tokens don't contribute).
/// This is "mean pooling" — the standard approach for sentence-transformers.
///
/// WHY TRACT?
/// tract is pure Rust with no C/C++ FFI. It can run on any platform Rust
/// compiles to, including WASM. No CUDA, no system-level ONNX Runtime install.
/// For a single-inference demo, the performance is more than adequate.

use tract_onnx::prelude::*;
use crate::tokenizer::EncodedInput;

/// The expected embedding dimension for all-MiniLM-L6-v2.
pub const EMBED_DIM: usize = 384;

/// Mean pool over token embeddings, weighted by attention mask.
///
/// `hidden_states` is a flat f32 buffer of shape [seq_len, embed_dim].
/// `mask` indicates which tokens are real (1) vs padding (0).
///
/// Returns a single embedding vector of length `embed_dim`.
///
/// This is a standalone function (not a method) so we can test it
/// independently without loading the ONNX model.
pub fn mean_pool(hidden_states: &[f32], seq_len: usize, embed_dim: usize, mask: &[i64]) -> Vec<f32> {
    let mut result = vec![0.0f32; embed_dim];
    let mut mask_sum = 0.0f32;

    for t in 0..seq_len {
        if mask[t] == 1 {
            for d in 0..embed_dim {
                result[d] += hidden_states[t * embed_dim + d];
            }
            mask_sum += 1.0;
        }
    }

    if mask_sum > 0.0 {
        for d in 0..embed_dim {
            result[d] /= mask_sum;
        }
    }

    result
}

/// ONNX model wrapper for all-MiniLM-L6-v2.
///
/// Loads the model once and can run inference on multiple inputs.
/// Each call to `embed` creates a new optimized plan for the input's
/// sequence length, since tract needs concrete shapes.
pub struct EmbeddingModel {
    /// Raw ONNX model (before shape specialization)
    model_bytes: Vec<u8>,
}

impl EmbeddingModel {
    /// Load an ONNX model from file.
    pub fn from_onnx(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model_bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read ONNX model from {}: {}", path, e))?;
        Ok(Self { model_bytes })
    }

    /// Extract a 384-dim embedding from tokenized input.
    ///
    /// Steps:
    /// 1. Build a tract model specialized to this input's sequence length
    /// 2. Run inference → [1, seq_len, 384] tensor
    /// 3. Mean pool with attention mask → [384] vector
    pub fn embed(&self, input: &EncodedInput) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let seq_len = input.len();

        // Build a model specialized to this sequence length.
        // tract needs concrete shapes to optimize the computation graph.
        // The ONNX model has 3 inputs: input_ids, attention_mask, token_type_ids
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(&self.model_bytes))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_len as i64)),
            )?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_len as i64)),
            )?
            .with_input_fact(
                2,
                InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_len as i64)),
            )?
            .into_optimized()?
            .into_runnable()?;

        // Create input tensors (ndarray → Tensor → TValue)
        let ids_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, seq_len),
            input.token_ids().to_vec(),
        )?.into();
        let mask_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, seq_len),
            input.attention_mask().to_vec(),
        )?.into();
        // token_type_ids: all zeros for single-sentence input
        let type_ids_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, seq_len),
            vec![0i64; seq_len],
        )?.into();

        // Run inference
        let outputs = model.run(tvec!(
            ids_tensor.into(),
            mask_tensor.into(),
            type_ids_tensor.into(),
        ))?;

        // Extract hidden states: [1, seq_len, 384]
        let hidden = outputs[0].to_array_view::<f32>()?;
        let flat: Vec<f32> = hidden.iter().copied().collect();

        // Mean pool
        Ok(mean_pool(&flat, seq_len, EMBED_DIM, input.attention_mask()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::SentenceTokenizer;

    fn tokenizer_path() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/tokenizer.json")
    }

    fn model_path() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/model.onnx")
    }

    // =========================================================================
    // TEST GROUP 1: Model loading
    // =========================================================================

    #[test]
    fn test_load_model() {
        let model = EmbeddingModel::from_onnx(model_path());
        assert!(model.is_ok(), "Failed to load ONNX model: {:?}", model.err());
    }

    // =========================================================================
    // TEST GROUP 2: Mean pooling (pure math, no model needed)
    // =========================================================================
    //
    // Mean pooling can be tested independently with synthetic data.
    // This isolates the averaging logic from the model.

    #[test]
    fn test_mean_pool_uniform() {
        // 2 tokens, 4-dim embeddings, all mask=1
        // Token 0: [1.0, 2.0, 3.0, 4.0]
        // Token 1: [3.0, 4.0, 5.0, 6.0]
        // Mean:    [2.0, 3.0, 4.0, 5.0]
        let hidden = vec![
            1.0f32, 2.0, 3.0, 4.0,
            3.0, 4.0, 5.0, 6.0,
        ];
        let mask = vec![1i64, 1];
        let result = mean_pool(&hidden, 2, 4, &mask);

        assert_eq!(result.len(), 4);
        assert!((result[0] - 2.0).abs() < 0.001);
        assert!((result[1] - 3.0).abs() < 0.001);
        assert!((result[2] - 4.0).abs() < 0.001);
        assert!((result[3] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_mean_pool_with_padding() {
        // 3 tokens, 2-dim embeddings, but token 2 is padding (mask=0)
        // Token 0: [10.0, 20.0]  mask=1
        // Token 1: [30.0, 40.0]  mask=1
        // Token 2: [99.0, 99.0]  mask=0 (should be ignored)
        // Mean of tokens 0,1: [20.0, 30.0]
        let hidden = vec![
            10.0f32, 20.0,
            30.0, 40.0,
            99.0, 99.0,
        ];
        let mask = vec![1i64, 1, 0];
        let result = mean_pool(&hidden, 3, 2, &mask);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 20.0).abs() < 0.001);
        assert!((result[1] - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_mean_pool_single_token() {
        // 1 token = no averaging needed
        let hidden = vec![5.0f32, 10.0, 15.0];
        let mask = vec![1i64];
        let result = mean_pool(&hidden, 1, 3, &mask);

        assert_eq!(result, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_mean_pool_all_masked() {
        // Edge case: all tokens masked out (shouldn't happen in practice,
        // but we should handle it gracefully — return zeros, not NaN/Inf)
        let hidden = vec![99.0f32, 99.0, 99.0, 99.0];
        let mask = vec![0i64, 0];
        let result = mean_pool(&hidden, 2, 2, &mask);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
    }

    // =========================================================================
    // TEST GROUP 3: Full embedding extraction
    // =========================================================================
    //
    // These tests require the actual ONNX model file.
    // They verify the complete pipeline: tokens → model → mean pool → 384-dim.

    #[test]
    fn test_embed_produces_384_dims() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let model = EmbeddingModel::from_onnx(model_path()).unwrap();

        let encoded = tok.encode("Hello world");
        let embedding = model.embed(&encoded).unwrap();

        assert_eq!(embedding.len(), 384, "Expected 384-dim embedding");
    }

    #[test]
    fn test_embed_deterministic() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let model = EmbeddingModel::from_onnx(model_path()).unwrap();

        let enc = tok.encode("The weather is nice today");
        let emb1 = model.embed(&enc).unwrap();
        let emb2 = model.embed(&enc).unwrap();

        // Same input should produce identical embeddings
        for i in 0..384 {
            assert_eq!(emb1[i], emb2[i], "Embedding mismatch at dim {}", i);
        }
    }

    #[test]
    fn test_embed_different_inputs_differ() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let model = EmbeddingModel::from_onnx(model_path()).unwrap();

        let emb_pos = model.embed(&tok.encode("I love this")).unwrap();
        let emb_neg = model.embed(&tok.encode("I hate this")).unwrap();

        // Different sentences should produce different embeddings
        let diff: f32 = emb_pos.iter().zip(emb_neg.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(diff > 0.1, "Expected different embeddings, L2 distance: {}", diff);
    }

    #[test]
    fn test_embed_values_are_reasonable() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let model = EmbeddingModel::from_onnx(model_path()).unwrap();

        let embedding = model.embed(&tok.encode("Test sentence")).unwrap();

        // Sentence-transformer embeddings are typically in [-1, 1] range
        // but not strictly bounded. Values should be reasonable, not NaN/Inf.
        for (i, &val) in embedding.iter().enumerate() {
            assert!(val.is_finite(), "Embedding dim {} is not finite: {}", i, val);
            assert!(
                val.abs() < 10.0,
                "Embedding dim {} unexpectedly large: {}",
                i, val
            );
        }
    }

    #[test]
    fn test_similar_sentences_closer_than_different() {
        let tok = SentenceTokenizer::from_file(tokenizer_path()).unwrap();
        let model = EmbeddingModel::from_onnx(model_path()).unwrap();

        let emb_a = model.embed(&tok.encode("The food was delicious")).unwrap();
        let emb_b = model.embed(&tok.encode("The meal was tasty")).unwrap();
        let emb_c = model.embed(&tok.encode("The car broke down on the highway")).unwrap();

        // Cosine similarity: similar sentences should be closer
        let sim_ab = cosine_similarity(&emb_a, &emb_b);
        let sim_ac = cosine_similarity(&emb_a, &emb_c);

        assert!(
            sim_ab > sim_ac,
            "Similar sentences should have higher cosine similarity: ab={}, ac={}",
            sim_ab, sim_ac
        );
    }

    /// Helper: cosine similarity between two f32 vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }
}
