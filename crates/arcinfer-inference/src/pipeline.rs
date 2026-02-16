/// The complete client-side inference pipeline.
///
/// FLOW:
/// 1. Tokenize text → token IDs + attention mask
/// 2. Run ONNX model → 384-dim f32 embedding
/// 3. Apply PCA → 16-dim f32 embedding
/// 4. Quantize to Q16.16 → [i32; 16] ready for MPC encryption
///
/// This module ties everything together. The output of this pipeline
/// is exactly what gets encrypted and sent to the Arcium MPC cluster.
///
/// NOTE ON PCA:
/// We don't have a trained PCA matrix yet (that requires running PCA on the
/// training set in Python). For now, the pipeline accepts PCA parameters
/// as inputs. In production, these would be loaded from a file exported
/// by the Python training script.

use crate::tokenizer::SentenceTokenizer;
use crate::embedding::EmbeddingModel;
use arcinfer_core::fixed_point::from_f64;

/// The complete client-side pipeline: text → quantized 16-dim input for MPC.
///
/// Holds the tokenizer and ONNX model, loaded once at startup.
pub struct InferencePipeline {
    tokenizer: SentenceTokenizer,
    model: EmbeddingModel,
}

impl InferencePipeline {
    /// Load the tokenizer and ONNX model from disk.
    pub fn load(
        tokenizer_path: &str,
        model_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = SentenceTokenizer::from_file(tokenizer_path)?;
        let model = EmbeddingModel::from_onnx(model_path)?;
        Ok(Self { tokenizer, model })
    }

    /// Extract a raw 384-dim f32 embedding (before PCA).
    ///
    /// Useful for debugging and for generating training data for PCA.
    pub fn embed_f32(&self, text: &str) -> Vec<f32> {
        let encoded = self.tokenizer.encode(text);
        self.model.embed(&encoded)
            .expect("Embedding extraction should not fail")
    }

    /// Extract a 16-dim f64 embedding after PCA reduction.
    ///
    /// `mean`: the 384-dim training set mean (for centering)
    /// `projection`: the 16×384 PCA projection matrix
    ///
    /// Both are in f64 because they come from Python (numpy/sklearn).
    /// The PCA math happens in f64 to preserve precision before quantization.
    pub fn embed_pca(
        &self,
        text: &str,
        mean: &[f64; 384],
        projection: &[[f64; 384]; 16],
    ) -> [f64; 16] {
        let raw = self.embed_f32(text);

        // Center: subtract mean (in f64 for precision)
        let mut centered = [0.0f64; 384];
        for i in 0..384 {
            centered[i] = raw[i] as f64 - mean[i];
        }

        // Project: multiply by projection matrix
        let mut output = [0.0f64; 16];
        for j in 0..16 {
            let mut sum = 0.0f64;
            for i in 0..384 {
                sum += projection[j][i] * centered[i];
            }
            output[j] = sum;
        }

        output
    }

    /// The full pipeline: text → 16-dim Q16.16 fixed-point, ready for MPC.
    ///
    /// This is the final output that gets encrypted and sent to the
    /// Arcium MPC cluster.
    pub fn embed_quantized(
        &self,
        text: &str,
        mean: &[f64; 384],
        projection: &[[f64; 384]; 16],
    ) -> [i32; 16] {
        let pca_output = self.embed_pca(text, mean, projection);

        let mut quantized = [0i32; 16];
        for i in 0..16 {
            quantized[i] = from_f64(pca_output[i]);
        }
        quantized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcinfer_core::fixed_point::to_f64;

    fn tokenizer_path() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/tokenizer.json")
    }

    fn model_path() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/model.onnx")
    }

    // =========================================================================
    // TEST GROUP 1: Pipeline construction
    // =========================================================================

    #[test]
    fn test_pipeline_loads() {
        let pipeline = InferencePipeline::load(tokenizer_path(), model_path());
        assert!(pipeline.is_ok(), "Pipeline failed to load: {:?}", pipeline.err());
    }

    // =========================================================================
    // TEST GROUP 2: Embedding extraction (384-dim f32)
    // =========================================================================
    //
    // Before PCA, verify the pipeline can produce raw 384-dim embeddings.

    #[test]
    fn test_pipeline_embed_f32() {
        let pipeline = InferencePipeline::load(tokenizer_path(), model_path()).unwrap();
        let embedding = pipeline.embed_f32("I love this movie");

        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_pipeline_embed_f32_sentiment_difference() {
        let pipeline = InferencePipeline::load(tokenizer_path(), model_path()).unwrap();

        let emb_pos = pipeline.embed_f32("This product is amazing and wonderful");
        let emb_neg = pipeline.embed_f32("This product is terrible and awful");

        // Verify they produce different embeddings
        let diff: f32 = emb_pos.iter().zip(emb_neg.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(diff > 0.1, "Positive and negative should differ: L2={}", diff);
    }

    // =========================================================================
    // TEST GROUP 3: PCA reduction (384 → 16)
    // =========================================================================
    //
    // Uses a simple test PCA (identity-like: pick first 16 dims) to verify
    // the pipeline can reduce dimensions. Real PCA comes from Python training.

    #[test]
    fn test_pipeline_embed_with_pca() {
        let pipeline = InferencePipeline::load(tokenizer_path(), model_path()).unwrap();

        // Simple PCA: pick first 16 dimensions (identity-like)
        let mut projection = [[0.0f64; 384]; 16];
        for i in 0..16 {
            projection[i][i] = 1.0;
        }
        let mean = [0.0f64; 384]; // zero mean = no centering

        let result = pipeline.embed_pca("Hello world", &mean, &projection);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_pipeline_pca_output_matches_raw_embedding() {
        let pipeline = InferencePipeline::load(tokenizer_path(), model_path()).unwrap();

        // Identity PCA (first 16 dims) with zero mean should match raw embedding
        let mut projection = [[0.0f64; 384]; 16];
        for i in 0..16 {
            projection[i][i] = 1.0;
        }
        let mean = [0.0f64; 384];

        let raw = pipeline.embed_f32("Test sentence");
        let pca = pipeline.embed_pca("Test sentence", &mean, &projection);

        // PCA output should match first 16 dims of raw embedding
        for i in 0..16 {
            let error = (pca[i] - raw[i] as f64).abs();
            assert!(error < 0.001, "Dim {} mismatch: pca={}, raw={}", i, pca[i], raw[i]);
        }
    }

    // =========================================================================
    // TEST GROUP 4: Full quantized output ([i32; 16])
    // =========================================================================
    //
    // The final output: 16-dim Q16.16 fixed-point, ready for MPC encryption.

    #[test]
    fn test_pipeline_full_quantized() {
        let pipeline = InferencePipeline::load(tokenizer_path(), model_path()).unwrap();

        // Identity PCA for testing
        let mut projection = [[0.0f64; 384]; 16];
        for i in 0..16 {
            projection[i][i] = 1.0;
        }
        let mean = [0.0f64; 384];

        let quantized: [i32; 16] = pipeline.embed_quantized("Hello world", &mean, &projection);

        // All values should be valid Q16.16 (non-zero for a real sentence)
        let nonzero_count = quantized.iter().filter(|&&v| v != 0).count();
        assert!(
            nonzero_count > 5,
            "Expected many non-zero Q16.16 values, got {}",
            nonzero_count
        );

        // Roundtrip: quantized values should approximate the f32 embedding
        let raw = pipeline.embed_f32("Hello world");
        for i in 0..16 {
            let recovered = to_f64(quantized[i]);
            let error = (recovered - raw[i] as f64).abs();
            assert!(
                error < 0.001,
                "Dim {} quantization error too large: {} vs {}",
                i, recovered, raw[i]
            );
        }
    }

    #[test]
    fn test_pipeline_quantized_different_for_different_inputs() {
        let pipeline = InferencePipeline::load(tokenizer_path(), model_path()).unwrap();

        let mut projection = [[0.0f64; 384]; 16];
        for i in 0..16 {
            projection[i][i] = 1.0;
        }
        let mean = [0.0f64; 384];

        let q1 = pipeline.embed_quantized("Great movie", &mean, &projection);
        let q2 = pipeline.embed_quantized("Terrible movie", &mean, &projection);

        assert_ne!(q1, q2, "Different inputs should produce different quantized outputs");
    }
}
