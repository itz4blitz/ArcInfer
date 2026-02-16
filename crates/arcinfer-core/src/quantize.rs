/// Batch quantization of neural network weights from f64 to Q16.16 fixed-point.
///
/// WORKFLOW:
/// 1. Train the classifier in Python (PyTorch) with square activations
/// 2. Export weights as JSON/binary arrays of f64 values
/// 3. This module converts them to fixed-point arrays sized for our Arcis circuit
/// 4. The fixed-point weights get hardcoded into the Arcis #[instruction] function
///
/// WHY NOT QUANTIZE IN THE CIRCUIT?
/// Because the circuit runs on encrypted data. The weights are PUBLIC constants
/// baked into the circuit at compile time — they're part of the program, not part
/// of the secret input. Quantization happens once, offline, before deployment.

use crate::fixed_point::from_f64;
use crate::nn::Linear;

/// The safe range for Q16.16: values beyond this will overflow i32
const MAX_SAFE_VALUE: f64 = 32767.0;
const MIN_SAFE_VALUE: f64 = -32768.0;

/// Quantize a slice of f64 values into a fixed-size array of Q16.16 i32.
///
/// Panics if:
///   - `values.len() != N` (length mismatch)
///   - Any value is outside [-32768, 32767] (out of Q16.16 range)
pub fn quantize_vec<const N: usize>(values: &[f64]) -> [i32; N] {
    assert_eq!(values.len(), N, "length mismatch: expected {}, got {}", N, values.len());
    let mut result = [0i32; N];
    for i in 0..N {
        assert!(
            values[i] >= MIN_SAFE_VALUE && values[i] <= MAX_SAFE_VALUE,
            "value {} ({}) out of Q16.16 range [{}, {}]",
            i, values[i], MIN_SAFE_VALUE, MAX_SAFE_VALUE
        );
        result[i] = from_f64(values[i]);
    }
    result
}

/// Quantize a 2D slice of f64 values into a fixed-size weight matrix.
///
/// `rows` must have exactly OUT elements, each with exactly IN elements.
pub fn quantize_matrix<const IN: usize, const OUT: usize>(rows: &[&[f64]]) -> [[i32; IN]; OUT] {
    assert_eq!(rows.len(), OUT, "row count mismatch: expected {}, got {}", OUT, rows.len());
    let mut result = [[0i32; IN]; OUT];
    for j in 0..OUT {
        result[j] = quantize_vec::<IN>(rows[j]);
    }
    result
}

/// Quantize f64 weights and biases directly into a Linear layer.
///
/// This is the main entry point for loading PyTorch-exported weights.
pub fn quantize_linear<const IN: usize, const OUT: usize>(
    weight_rows: &[&[f64]],
    biases: &[f64],
) -> Linear<IN, OUT> {
    Linear {
        weights: quantize_matrix::<IN, OUT>(weight_rows),
        biases: quantize_vec::<OUT>(biases),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::to_f64;

    // =========================================================================
    // TEST GROUP 1: Quantize a 1D weight vector
    // =========================================================================

    #[test]
    fn test_quantize_vec_simple() {
        let weights_f64 = &[1.0, -0.5, 0.25, 0.0];
        let quantized: [i32; 4] = quantize_vec(weights_f64);

        assert_eq!(to_f64(quantized[0]), 1.0);
        assert_eq!(to_f64(quantized[1]), -0.5);
        assert_eq!(to_f64(quantized[2]), 0.25);
        assert_eq!(to_f64(quantized[3]), 0.0);
    }

    #[test]
    fn test_quantize_vec_fractional() {
        let weights_f64 = &[0.3427, -0.8912, 1.5003];
        let quantized: [i32; 3] = quantize_vec(weights_f64);

        // Should roundtrip within Q16.16 precision (~0.0001)
        for i in 0..3 {
            let error = (weights_f64[i] - to_f64(quantized[i])).abs();
            assert!(error < 0.0001, "Element {} error too large: {}", i, error);
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_quantize_vec_wrong_length() {
        let weights_f64 = &[1.0, 2.0, 3.0]; // 3 elements
        let _: [i32; 2] = quantize_vec(weights_f64); // expects 2
    }

    // =========================================================================
    // TEST GROUP 2: Quantize a 2D weight matrix (for Linear layers)
    // =========================================================================

    #[test]
    fn test_quantize_matrix_2x3() {
        // A 2-output, 3-input weight matrix
        let weights_f64: &[&[f64]] = &[
            &[0.5, -0.3, 0.1],
            &[0.2, 0.8, -0.4],
        ];
        let quantized: [[i32; 3]; 2] = quantize_matrix(weights_f64);

        assert_eq!(to_f64(quantized[0][0]), 0.5);
        assert!((to_f64(quantized[0][1]) - (-0.3)).abs() < 0.0001);
        assert!((to_f64(quantized[1][2]) - (-0.4)).abs() < 0.0001);
    }

    #[test]
    fn test_quantize_matrix_roundtrip_accuracy() {
        // Simulate realistic neural net weights
        let weights_f64: &[&[f64]] = &[
            &[0.0142, -0.2831, 0.5012, -0.0003],
            &[-1.2345, 0.9876, 0.0001, 0.7654],
        ];
        let quantized: [[i32; 4]; 2] = quantize_matrix(weights_f64);

        for row in 0..2 {
            for col in 0..4 {
                let error = (weights_f64[row][col] - to_f64(quantized[row][col])).abs();
                assert!(
                    error < 0.0001,
                    "weights[{}][{}] error {}: {} vs {}",
                    row, col, error,
                    weights_f64[row][col], to_f64(quantized[row][col])
                );
            }
        }
    }

    // =========================================================================
    // TEST GROUP 3: Quantize into a Linear layer directly
    // =========================================================================
    //
    // This is the primary use case: take f64 weights from PyTorch,
    // produce a ready-to-use Linear<IN, OUT> struct.

    #[test]
    fn test_quantize_to_linear() {
        use crate::nn::Linear;
        use crate::fixed_point::from_f64;

        let weights_f64: &[&[f64]] = &[
            &[0.5, -0.3],
            &[0.2, 0.8],
        ];
        let biases_f64 = &[0.1, -0.1];

        let layer: Linear<2, 2> = quantize_linear(weights_f64, biases_f64);

        // Run a forward pass with known input
        let input = [from_f64(1.0), from_f64(2.0)];
        let output = layer.forward(&input);

        // neuron 0: 1.0*0.5 + 2.0*(-0.3) + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
        // neuron 1: 1.0*0.2 + 2.0*0.8 + (-0.1) = 0.2 + 1.6 - 0.1 = 1.7
        let out0 = to_f64(output[0]);
        let out1 = to_f64(output[1]);

        assert!(out0.abs() < 0.01, "Expected ~0.0, got {}", out0);
        assert!((out1 - 1.7).abs() < 0.01, "Expected ~1.7, got {}", out1);
    }

    // =========================================================================
    // TEST GROUP 4: Clipping / range validation
    // =========================================================================
    //
    // Q16.16 can represent [-32768, 32767]. If a weight is outside this range,
    // we should catch it before it silently wraps around.

    #[test]
    fn test_quantize_warns_on_large_values() {
        // Values within safe neural net range should be fine
        let safe = &[9.99, -9.99, 0.001];
        let result: [i32; 3] = quantize_vec(safe);
        assert!((to_f64(result[0]) - 9.99).abs() < 0.001);
    }

    #[test]
    #[should_panic(expected = "out of Q16.16 range")]
    fn test_quantize_panics_on_overflow() {
        // 40000.0 exceeds Q16.16 i32 range (max ~32767)
        let overflow = &[40000.0];
        let _: [i32; 1] = quantize_vec(overflow);
    }
}
