/// The production sentiment classifier: Linear(16→16) → x² → Linear(16→8) → x² → Linear(8→2)
///
/// This module defines the exact architecture that will run inside the Arcis circuit.
/// The dimensions are chosen based on:
///   - Input: 16-dim (after PCA from 384-dim sentence embedding)
///   - Hidden 1: 16 neurons
///   - Hidden 2: 8 neurons
///   - Output: 2 logits (positive/negative sentiment)
///
/// TOTAL PARAMETERS:
///   Layer 1: 16×16 weights + 16 biases = 272
///   Layer 2: 16×8 weights + 8 biases = 136
///   Layer 3: 8×2 weights + 2 biases   = 18
///   Total: 426 parameters
///
/// MPC DEPTH: 10 rounds (2 per layer + 2 per activation)
///
/// This fits comfortably in Arcium's constraints:
///   - All arrays are fixed-size (compile-time constants)
///   - Output is 2 × i32 = 8 bytes, well under the 1232-byte callback limit
///   - Total computation is ~2,642 multiplications (parallel within each layer)

use crate::nn::{Linear, square_activate, argmax};

/// The three layers of the classifier, sized for our architecture.
pub struct SentimentClassifier {
    pub layer1: Linear<16, 16>,
    pub layer2: Linear<16, 8>,
    pub layer3: Linear<8, 2>,
}

impl SentimentClassifier {
    /// Run the full forward pass. Returns raw logits [i32; 2].
    ///
    /// In the Arcis circuit, this exact sequence runs on secret-shared values.
    /// Client-side, we use it to verify correctness before deploying to MPC.
    pub fn forward(&self, input: &[i32; 16]) -> [i32; 2] {
        let h1 = square_activate(&self.layer1.forward(input));
        let h2 = square_activate(&self.layer2.forward(&h1));
        self.layer3.forward(&h2)
    }

    /// Classify: returns 0 (negative) or 1 (positive).
    /// Runs argmax on the logits — client-side only, after MPC decryption.
    pub fn classify(&self, input: &[i32; 16]) -> usize {
        argmax(&self.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::{from_f64, to_f64};
    use crate::quantize::quantize_linear;

    // =========================================================================
    // Helper: build a classifier with small known weights for testing
    // =========================================================================

    /// Create a classifier where layer1 and layer2 are near-identity
    /// (pass through first N dims, zero out the rest) and layer3 uses
    /// hand-picked weights to produce a known classification.
    fn test_classifier() -> SentimentClassifier {
        // Layer 1 (16→16): Identity-like, scale by 0.5
        // to keep values small after squaring
        let mut w1 = [[from_f64(0.0); 16]; 16];
        for i in 0..16 {
            w1[i][i] = from_f64(0.5);
        }
        let b1 = [from_f64(0.0); 16];
        let layer1 = Linear { weights: w1, biases: b1 };

        // Layer 2 (16→8): Pass first 8 dims through, scale by 0.5
        let mut w2 = [[from_f64(0.0); 16]; 8];
        for i in 0..8 {
            w2[i][i] = from_f64(0.5);
        }
        let b2 = [from_f64(0.0); 8];
        let layer2 = Linear { weights: w2, biases: b2 };

        // Layer 3 (8→2): Sum first 4 dims for class 0, sum last 4 for class 1
        let mut w3 = [[from_f64(0.0); 8]; 2];
        for i in 0..4 {
            w3[0][i] = from_f64(1.0);      // class 0 looks at dims 0-3
            w3[1][i + 4] = from_f64(1.0);  // class 1 looks at dims 4-7
        }
        let b3 = [from_f64(0.0); 2];
        let layer3 = Linear { weights: w3, biases: b3 };

        SentimentClassifier { layer1, layer2, layer3 }
    }

    // =========================================================================
    // TEST GROUP 1: Basic forward pass dimensions
    // =========================================================================

    #[test]
    fn test_forward_returns_two_logits() {
        let classifier = test_classifier();
        let input = [from_f64(0.0); 16];
        let logits = classifier.forward(&input);

        // Should return exactly 2 logits
        assert_eq!(logits.len(), 2);
    }

    #[test]
    fn test_forward_zero_input() {
        let classifier = test_classifier();
        let input = [from_f64(0.0); 16];
        let logits = classifier.forward(&input);

        // All zeros in → should get zeros out (0 through all layers)
        assert_eq!(to_f64(logits[0]), 0.0);
        assert_eq!(to_f64(logits[1]), 0.0);
    }

    // =========================================================================
    // TEST GROUP 2: Classification correctness
    // =========================================================================

    #[test]
    fn test_classify_class_0() {
        let classifier = test_classifier();

        // Input with energy in first 4 dims (class 0 signal)
        let mut input = [from_f64(0.0); 16];
        for i in 0..4 {
            input[i] = from_f64(2.0);
        }

        let class = classifier.classify(&input);
        assert_eq!(class, 0, "Expected class 0 (energy in first 4 dims)");
    }

    #[test]
    fn test_classify_class_1() {
        let classifier = test_classifier();

        // Input with energy in dims 4-7 (class 1 signal)
        let mut input = [from_f64(0.0); 16];
        for i in 4..8 {
            input[i] = from_f64(2.0);
        }

        let class = classifier.classify(&input);
        assert_eq!(class, 1, "Expected class 1 (energy in dims 4-7)");
    }

    #[test]
    fn test_classify_stronger_signal_wins() {
        let classifier = test_classifier();

        // Both class regions have signal, but class 1 is stronger
        let mut input = [from_f64(0.0); 16];
        for i in 0..4 {
            input[i] = from_f64(1.0);   // weak class 0 signal
        }
        for i in 4..8 {
            input[i] = from_f64(3.0);   // strong class 1 signal
        }

        let class = classifier.classify(&input);
        assert_eq!(class, 1, "Stronger class 1 signal should win");
    }

    // =========================================================================
    // TEST GROUP 3: Verify intermediate values make sense
    // =========================================================================
    //
    // These tests trace through the computation to ensure each layer
    // does what we expect. Important for debugging if the Arcis circuit
    // ever produces different results from the client-side reference.

    #[test]
    fn test_forward_trace_values() {
        let classifier = test_classifier();

        // Input: dim 0 = 4.0, everything else = 0
        let mut input = [from_f64(0.0); 16];
        input[0] = from_f64(4.0);

        // Trace:
        // Layer 1: w1[0][0] = 0.5, so neuron 0 = 0.5 * 4.0 = 2.0
        //          All other neurons = 0
        // After square: [4.0, 0, 0, ..., 0]  (2.0² = 4.0)
        //
        // Layer 2: w2[0][0] = 0.5, so neuron 0 = 0.5 * 4.0 = 2.0
        //          All other neurons = 0
        // After square: [4.0, 0, 0, ..., 0]  (2.0² = 4.0)
        //
        // Layer 3: w3[0][0] = 1.0, so class 0 = 1.0 * 4.0 = 4.0
        //          w3[1][0] = 0.0, so class 1 = 0.0
        //
        // Expected logits: [4.0, 0.0]
        let logits = classifier.forward(&input);

        assert!(
            (to_f64(logits[0]) - 4.0).abs() < 0.1,
            "Expected logit[0] ~4.0, got {}",
            to_f64(logits[0])
        );
        assert!(
            to_f64(logits[1]).abs() < 0.1,
            "Expected logit[1] ~0.0, got {}",
            to_f64(logits[1])
        );
    }

    // =========================================================================
    // TEST GROUP 4: Quantized weight loading
    // =========================================================================
    //
    // Verify that we can build a classifier from f64 weights
    // (as they'd come from PyTorch export) via the quantize module.

    #[test]
    fn test_build_from_quantized_weights() {
        // Tiny example: build a classifier where we manually specify f64 weights
        // and verify the forward pass works after quantization.

        // Layer 1 (16→16): identity-like, scale by 0.5
        let mut w1_f64: Vec<Vec<f64>> = vec![vec![0.0; 16]; 16];
        for i in 0..16 {
            w1_f64[i][i] = 0.5;
        }
        let w1_refs: Vec<&[f64]> = w1_f64.iter().map(|r| r.as_slice()).collect();
        let b1_f64 = vec![0.0; 16];

        // Layer 2 (16→8): pass first 8 dims, scale by 0.5
        let mut w2_f64: Vec<Vec<f64>> = vec![vec![0.0; 16]; 8];
        for i in 0..8 {
            w2_f64[i][i] = 0.5;
        }
        let w2_refs: Vec<&[f64]> = w2_f64.iter().map(|r| r.as_slice()).collect();
        let b2_f64 = vec![0.0; 8];

        // Layer 3 (8→2): first 4 dims → class 0, last 4 → class 1
        let mut w3_f64: Vec<Vec<f64>> = vec![vec![0.0; 8]; 2];
        for i in 0..4 {
            w3_f64[0][i] = 1.0;
            w3_f64[1][i + 4] = 1.0;
        }
        let w3_refs: Vec<&[f64]> = w3_f64.iter().map(|r| r.as_slice()).collect();
        let b3_f64 = vec![0.0; 2];

        let classifier = SentimentClassifier {
            layer1: quantize_linear(&w1_refs, &b1_f64),
            layer2: quantize_linear(&w2_refs, &b2_f64),
            layer3: quantize_linear(&w3_refs, &b3_f64),
        };

        // Should classify the same as test_classifier()
        let mut input = [from_f64(0.0); 16];
        for i in 4..8 {
            input[i] = from_f64(2.0);
        }
        assert_eq!(classifier.classify(&input), 1);
    }
}
