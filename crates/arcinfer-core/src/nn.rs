/// Neural network layers using fixed-point arithmetic.
///
/// DESIGN CONSTRAINT: Everything here must be expressible in Arcis.
/// That means:
///   - Fixed-size arrays only (no Vec, no heap allocation)
///   - All dimensions known at compile time (const generics)
///   - Only operations available in MPC: add, multiply, square
///   - No branching on secret values (no ReLU, no max, no if-on-data)
///
/// WHY CONST GENERICS?
/// In Arcis, array sizes must be compile-time constants. Rust const generics
/// let us write `Linear<IN, OUT>` where IN and OUT are known at compile time.
/// This means the compiler enforces dimension correctness — if you try to
/// feed a 16-dim vector into a layer expecting 8-dim, it won't compile.
/// The Arcis circuit will use fixed arrays like `[i32; 16]` directly.

use crate::fixed_point::{fp_add, fp_dot, fp_square};

/// A fully-connected (dense) linear layer: output = input * W^T + bias
///
/// Generic over:
///   - IN: input dimension (e.g., 16 for our PCA-reduced embedding)
///   - OUT: output dimension (e.g., 16 for the first hidden layer)
///
/// `weights` is stored as [OUT][IN] — each row is the weight vector for one output neuron.
/// This matches how neural nets are typically stored and how Arcis arrays work.
///
/// MPC COST: One multiplication "depth" (all dot products are independent and parallel).
/// Total MPC rounds for this layer: 2 (1 multiply + 1 truncation).
pub struct Linear<const IN: usize, const OUT: usize> {
    pub weights: [[i32; IN]; OUT],
    pub biases: [i32; OUT],
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// Compute the forward pass: output[j] = dot(weights[j], input) + biases[j]
    ///
    /// Returns a fixed-size array [i32; OUT].
    pub fn forward(&self, input: &[i32; IN]) -> [i32; OUT] {
        let mut output = [0i32; OUT];
        for j in 0..OUT {
            // fp_dot handles the multiply-accumulate-truncate pattern
            output[j] = fp_add(fp_dot(&self.weights[j], input.as_slice()), self.biases[j]);
        }
        output
    }
}

/// Apply square activation element-wise: f(x) = x²
///
/// This replaces ReLU for MPC efficiency. Each element costs 1 fp_mul.
/// All elements are independent, so in MPC they execute in parallel.
/// MPC depth: 1 multiplication round.
pub fn square_activate<const N: usize>(input: &[i32; N]) -> [i32; N] {
    let mut output = [0i32; N];
    for i in 0..N {
        output[i] = fp_square(input[i]);
    }
    output
}

/// Argmax over fixed-point logits. Returns the index of the largest value.
///
/// RUNS CLIENT-SIDE ONLY — after decrypting the MPC output.
/// In the Arcis circuit, we output raw logits. The client decrypts them,
/// then calls argmax to determine the classification.
///
/// This avoids softmax entirely in MPC (softmax needs exp() and division,
/// costing ~50+ MPC rounds). Since argmax(softmax(x)) == argmax(x),
/// we get the same classification result for free.
pub fn argmax<const N: usize>(logits: &[i32; N]) -> usize {
    let mut max_idx = 0;
    let mut max_val = logits[0];
    for i in 1..N {
        if logits[i] > max_val {
            max_val = logits[i];
            max_idx = i;
        }
    }
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::{from_f64, to_f64};

    // =========================================================================
    // TEST GROUP 1: Linear layer (single layer, no activation)
    // =========================================================================
    //
    // A Linear layer computes: output[j] = sum(input[i] * weights[j][i]) + bias[j]
    // This is matrix-vector multiplication + bias addition.
    //
    // In MPC terms:
    //   - Each weight multiplication is 1 fp_mul (2 MPC rounds)
    //   - But within a dot product, multiplications are INDEPENDENT, so they
    //     can happen in parallel across the cluster. The depth is still 1 mul.
    //   - Bias addition is free (local operation on shares)
    //   - Total MPC depth per layer: 1 multiplication round + 1 truncation round

    #[test]
    fn test_linear_layer_identity() {
        // A 2->2 linear layer with identity weights and zero bias
        // should pass input through unchanged.
        let weights = [
            [from_f64(1.0), from_f64(0.0)],
            [from_f64(0.0), from_f64(1.0)],
        ];
        let biases = [from_f64(0.0), from_f64(0.0)];
        let layer = Linear { weights, biases };

        let input = [from_f64(3.0), from_f64(7.0)];
        let output = layer.forward(&input);

        assert_eq!(to_f64(output[0]), 3.0);
        assert_eq!(to_f64(output[1]), 7.0);
    }

    #[test]
    fn test_linear_layer_with_bias() {
        // 2->1 layer: output = 0.5*x0 + 0.5*x1 + 1.0
        // Input: [2.0, 4.0] -> 0.5*2 + 0.5*4 + 1 = 4.0
        let weights = [[from_f64(0.5), from_f64(0.5)]];
        let biases = [from_f64(1.0)];
        let layer = Linear { weights, biases };

        let input = [from_f64(2.0), from_f64(4.0)];
        let output = layer.forward(&input);

        assert_eq!(to_f64(output[0]), 4.0);
    }

    #[test]
    fn test_linear_layer_negative_weights() {
        // Tests that negative weights work correctly in fixed-point
        // 2->1: output = -1.0*x0 + 2.0*x1 + 0.0
        // Input: [3.0, 1.0] -> -3 + 2 = -1.0
        let weights = [[from_f64(-1.0), from_f64(2.0)]];
        let biases = [from_f64(0.0)];
        let layer = Linear { weights, biases };

        let input = [from_f64(3.0), from_f64(1.0)];
        let output = layer.forward(&input);

        assert_eq!(to_f64(output[0]), -1.0);
    }

    // =========================================================================
    // TEST GROUP 2: Square activation applied to arrays
    // =========================================================================

    #[test]
    fn test_square_activation_array() {
        let input = [from_f64(2.0), from_f64(-3.0), from_f64(0.5)];
        let output = square_activate(&input);

        assert_eq!(to_f64(output[0]), 4.0);
        assert_eq!(to_f64(output[1]), 9.0);
        assert_eq!(to_f64(output[2]), 0.25);
    }

    #[test]
    fn test_square_activation_zeros() {
        let input = [from_f64(0.0), from_f64(0.0)];
        let output = square_activate(&input);

        assert_eq!(to_f64(output[0]), 0.0);
        assert_eq!(to_f64(output[1]), 0.0);
    }

    // =========================================================================
    // TEST GROUP 3: Classifier (stacked layers = the full MPC circuit)
    // =========================================================================
    //
    // Our sentiment classifier architecture:
    //   Input (16-dim embedding after PCA)
    //   -> Linear(16 -> 16) -> Square
    //   -> Linear(16 -> 8) -> Square
    //   -> Linear(8 -> 2)  -> raw logits (no softmax — too expensive in MPC)
    //
    // For testing, we use a tiny 2->3->2 network to verify the wiring.

    #[test]
    fn test_classifier_two_layers() {
        // 2->3 (+ square) -> 3->2
        // This tests the full forward pass with intermediate activation.
        //
        // Layer 1: 2->3
        let w1 = [
            [from_f64(1.0), from_f64(0.0)],   // neuron 0: passes x0
            [from_f64(0.0), from_f64(1.0)],   // neuron 1: passes x1
            [from_f64(0.5), from_f64(0.5)],   // neuron 2: average
        ];
        let b1 = [from_f64(0.0), from_f64(0.0), from_f64(0.0)];
        let layer1 = Linear { weights: w1, biases: b1 };

        // Layer 2: 3->2
        let w2 = [
            [from_f64(1.0), from_f64(0.0), from_f64(0.0)],  // output 0 = hidden 0
            [from_f64(0.0), from_f64(1.0), from_f64(0.0)],  // output 1 = hidden 1
        ];
        let b2 = [from_f64(0.0), from_f64(0.0)];
        let layer2 = Linear { weights: w2, biases: b2 };

        // Input: [2.0, 3.0]
        // Layer 1 output: [2.0, 3.0, 2.5]
        // After square: [4.0, 9.0, 6.25]
        // Layer 2 output: [4.0, 9.0]
        let input = [from_f64(2.0), from_f64(3.0)];

        let hidden = layer1.forward(&input);
        let activated = square_activate(&hidden);
        let output = layer2.forward(&activated);

        let out0 = to_f64(output[0]);
        let out1 = to_f64(output[1]);

        assert!((out0 - 4.0).abs() < 0.01, "Expected ~4.0, got {}", out0);
        assert!((out1 - 9.0).abs() < 0.01, "Expected ~9.0, got {}", out1);
    }

    #[test]
    fn test_classifier_determines_sentiment() {
        // Simulate a classifier that's been "trained" to output:
        //   logit[0] > logit[1]  => NEGATIVE
        //   logit[1] > logit[0]  => POSITIVE
        //
        // We manually set weights so that input [1.0, 0.0] => POSITIVE
        // and input [0.0, 1.0] => NEGATIVE.
        //
        // This proves the argmax-on-logits pattern works without softmax.

        // Layer: 2->2 with hand-picked weights
        let weights = [
            [from_f64(-1.0), from_f64(1.0)],  // neg logit: high when x1 > x0
            [from_f64(1.0), from_f64(-1.0)],   // pos logit: high when x0 > x1
        ];
        let biases = [from_f64(0.0), from_f64(0.0)];
        let layer = Linear { weights, biases };

        // "Positive" input
        let pos_input = [from_f64(1.0), from_f64(0.0)];
        let pos_output = layer.forward(&pos_input);
        // pos_output[1] should be > pos_output[0]
        assert!(
            pos_output[1] > pos_output[0],
            "Expected POSITIVE: logit[1]={} > logit[0]={}",
            to_f64(pos_output[1]), to_f64(pos_output[0])
        );

        // "Negative" input
        let neg_input = [from_f64(0.0), from_f64(1.0)];
        let neg_output = layer.forward(&neg_input);
        // neg_output[0] should be > neg_output[1]
        assert!(
            neg_output[0] > neg_output[1],
            "Expected NEGATIVE: logit[0]={} > logit[1]={}",
            to_f64(neg_output[0]), to_f64(neg_output[1])
        );
    }

    // =========================================================================
    // TEST GROUP 4: Argmax (how we pick the class without softmax)
    // =========================================================================
    //
    // In standard ML: softmax converts logits to probabilities, then argmax.
    // Softmax = exp(x_i) / sum(exp(x_j)) -- requires exp() and division,
    // both EXTREMELY expensive in MPC.
    //
    // Key insight: argmax(softmax(x)) == argmax(x)
    // Softmax is monotonic, so it preserves ordering. We skip it entirely
    // and just find the max logit. The classification result is the same.
    //
    // In the Arcis circuit, we'll output the raw logits as Enc<Shared, [i32; 2]>.
    // The CLIENT does argmax after decryption. No MPC cost for classification.

    #[test]
    fn test_argmax_positive_wins() {
        let logits = [from_f64(-1.5), from_f64(2.3)];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_negative_wins() {
        let logits = [from_f64(5.0), from_f64(-3.0)];
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_close_values() {
        // Even tiny differences should be distinguishable
        let logits = [from_f64(1.001), from_f64(1.002)];
        assert_eq!(argmax(&logits), 1);
    }

    // =========================================================================
    // TEST GROUP 5: End-to-end sentiment classification
    // =========================================================================

    #[test]
    fn test_end_to_end_classify() {
        // Full pipeline: input -> layer1 -> square -> layer2 -> argmax
        // Using a 2->2->2 network with hand-tuned weights.
        //
        // KEY LESSON: Square activation destroys sign info ((-x)² == x²).
        // So layer 1 must ROUTE dimensions separately, not compute differences.
        // After squaring, the magnitudes will differ, and layer 2 can classify.

        // Layer 1: pass-through (routes each input dimension to its own neuron)
        let w1 = [
            [from_f64(1.0), from_f64(0.0)],   // neuron 0 = x0
            [from_f64(0.0), from_f64(1.0)],   // neuron 1 = x1
        ];
        let b1 = [from_f64(0.0), from_f64(0.0)];
        let layer1 = Linear { weights: w1, biases: b1 };

        // After squaring: [x0², x1²] — now magnitude differences are preserved
        //   pos [0.8, 0.2] -> [0.64, 0.04]  (x0² >> x1²)
        //   neg [0.2, 0.8] -> [0.04, 0.64]  (x1² >> x0²)

        // Layer 2: compare squared magnitudes
        let w2 = [
            [from_f64(1.0), from_f64(-1.0)],  // class 0: high when x0² > x1²
            [from_f64(-1.0), from_f64(1.0)],   // class 1: high when x1² > x0²
        ];
        let b2 = [from_f64(0.0), from_f64(0.0)];
        let layer2 = Linear { weights: w2, biases: b2 };

        // "Positive" signal: x0 > x1 => after square, x0² > x1² => class 0
        let pos = [from_f64(0.8), from_f64(0.2)];
        let h = square_activate(&layer1.forward(&pos));
        let logits = layer2.forward(&h);
        let class = argmax(&logits);
        assert_eq!(class, 0, "Expected class 0 for pos signal");

        // "Negative" signal: x1 > x0 => after square, x1² > x0² => class 1
        let neg = [from_f64(0.2), from_f64(0.8)];
        let h = square_activate(&layer1.forward(&neg));
        let logits = layer2.forward(&h);
        let class = argmax(&logits);
        assert_eq!(class, 1, "Expected class 1 for neg signal");
    }
}
