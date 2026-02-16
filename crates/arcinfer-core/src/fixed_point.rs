/// Q16.16 fixed-point arithmetic for MPC-compatible neural network inference.
///
/// WHY Q16.16?
/// - 16 fractional bits gives us precision to ~0.0000153 (1/65536)
/// - 16 integer bits gives us range [-32768, 32767]
/// - Neural network weights typically live in [-10, 10], so this is plenty
/// - Total: 32 bits, fits in Arcis i32/u32. We use i64 for intermediate products
///   to avoid overflow during multiplication (32-bit * 32-bit = 64-bit result)
///
/// IN THE ARCIS CIRCUIT:
/// The exact same operations happen on secret-shared values. Each Arx node holds
/// a share of the fixed-point integer. Addition of shares = addition of values.
/// Multiplication requires a round of communication between nodes, plus a
/// truncation protocol to maintain the fixed-point scale.

/// Number of fractional bits in our Q16.16 representation
pub const FRAC_BITS: u32 = 16;

/// The scaling factor: 2^16 = 65536
pub const SCALE: i64 = 1 << FRAC_BITS;

/// Convert a floating-point value to Q16.16 fixed-point.
///
/// This runs CLIENT-SIDE only (not in MPC). We use it to quantize model weights
/// and input embeddings before encrypting them for Arcium.
pub fn from_f64(value: f64) -> i32 {
    (value * SCALE as f64).round() as i32
}

/// Convert Q16.16 fixed-point back to f64.
///
/// This runs CLIENT-SIDE only, after decrypting the MPC output.
pub fn to_f64(fp: i32) -> f64 {
    fp as f64 / SCALE as f64
}

/// Fixed-point addition. Just integer addition — the scale factors cancel.
///
/// In MPC: addition of secret shares is LOCAL (no communication needed).
/// Each Arx node just adds its share. This is "free" in terms of MPC rounds.
pub fn fp_add(a: i32, b: i32) -> i32 {
    a + b
}

/// Fixed-point multiplication with truncation.
///
/// We widen to i64 to prevent overflow: (i32 * i32) can exceed i32 range.
/// Then right-shift by FRAC_BITS to maintain the Q16.16 scale.
///
/// In MPC: this is the EXPENSIVE operation. It requires:
///   1. Beaver triple consumption (pre-processed correlated randomness)
///   2. One round of communication between all Arx nodes
///   3. A truncation protocol (another round) to right-shift the shared value
///
/// Every fp_mul costs ~2 communication rounds. Minimizing multiplications
/// is the #1 optimization for MPC performance.
pub fn fp_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> FRAC_BITS) as i32
}

/// Square activation: f(x) = x²
///
/// This replaces ReLU in our MPC-friendly neural network.
/// Cost: exactly 1 fp_mul (= 2 MPC communication rounds).
/// Compare to ReLU which needs bit-decomposition (~20-40 rounds).
pub fn fp_square(x: i32) -> i32 {
    fp_mul(x, x)
}

/// Dot product of two fixed-point vectors.
///
/// Computes sum(a[i] * b[i]) — the core operation of every neural net layer.
/// Accumulates in i64 to avoid intermediate overflow, then truncates once at the end.
///
/// Why accumulate-then-truncate instead of truncating each multiplication?
/// - Fewer truncation operations = fewer MPC rounds
/// - Less accumulated rounding error
/// - In the Arcis circuit, we'll use the same pattern
pub fn fp_dot(a: &[i32], b: &[i32]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot product vectors must have equal lengths");
    let sum: i64 = a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai as i64 * bi as i64)
        .sum();
    (sum >> FRAC_BITS) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TEST GROUP 1: Conversion (f64 <-> fixed-point)
    // =========================================================================

    #[test]
    fn test_from_f64_positive() {
        // 3.5 * 65536 = 229376
        let fp = from_f64(3.5);
        assert_eq!(fp, 229376);
    }

    #[test]
    fn test_from_f64_negative() {
        // -2.25 * 65536 = -147456
        let fp = from_f64(-2.25);
        assert_eq!(fp, -147456);
    }

    #[test]
    fn test_from_f64_zero() {
        assert_eq!(from_f64(0.0), 0);
    }

    #[test]
    fn test_from_f64_small_fraction() {
        // 0.001 * 65536 = 65.536 → rounds to 66
        let fp = from_f64(0.001);
        // Allow rounding: should be 65 or 66
        assert!((fp - 66).abs() <= 1, "Expected ~66, got {}", fp);
    }

    #[test]
    fn test_to_f64_roundtrip() {
        // Convert to fixed, then back. Should be close (within 1/65536 precision)
        let original = 3.14159;
        let fp = from_f64(original);
        let recovered = to_f64(fp);
        let error = (original - recovered).abs();
        assert!(error < 0.0001, "Roundtrip error too large: {}", error);
    }

    #[test]
    fn test_to_f64_negative_roundtrip() {
        let original = -7.777;
        let fp = from_f64(original);
        let recovered = to_f64(fp);
        let error = (original - recovered).abs();
        assert!(error < 0.0001, "Roundtrip error too large: {}", error);
    }

    // =========================================================================
    // TEST GROUP 2: Addition (trivial in fixed-point — just integer add)
    // =========================================================================

    #[test]
    fn test_add_positive() {
        let a = from_f64(1.5);
        let b = from_f64(2.5);
        let result = fp_add(a, b);
        assert_eq!(to_f64(result), 4.0);
    }

    #[test]
    fn test_add_negative() {
        let a = from_f64(3.0);
        let b = from_f64(-5.0);
        let result = fp_add(a, b);
        assert_eq!(to_f64(result), -2.0);
    }

    // =========================================================================
    // TEST GROUP 3: Multiplication (the critical one for MPC)
    // =========================================================================
    //
    // In MPC: multiplication of two secret-shared values requires:
    //   1. Each node multiplies its local shares (gives a 64-bit intermediate)
    //   2. Nodes communicate to re-share the product
    //   3. A truncation protocol right-shifts by FRAC_BITS to maintain scale
    //
    // If truncation is wrong, values explode or vanish. These tests catch that.

    #[test]
    fn test_mul_simple() {
        // 2.0 * 3.0 = 6.0
        let a = from_f64(2.0);
        let b = from_f64(3.0);
        let result = fp_mul(a, b);
        assert_eq!(to_f64(result), 6.0);
    }

    #[test]
    fn test_mul_fractions() {
        // 0.5 * 0.5 = 0.25
        let a = from_f64(0.5);
        let b = from_f64(0.5);
        let result = fp_mul(a, b);
        assert_eq!(to_f64(result), 0.25);
    }

    #[test]
    fn test_mul_negative() {
        // 3.0 * -2.0 = -6.0
        let a = from_f64(3.0);
        let b = from_f64(-2.0);
        let result = fp_mul(a, b);
        assert_eq!(to_f64(result), -6.0);
    }

    #[test]
    fn test_mul_by_weight() {
        // Simulates multiplying a feature by a neural net weight
        // 0.7832 * 0.3145 = 0.24631...
        let feature = from_f64(0.7832);
        let weight = from_f64(0.3145);
        let result = fp_mul(feature, weight);
        let recovered = to_f64(result);
        let expected = 0.7832 * 0.3145;
        let error = (expected - recovered).abs();
        assert!(error < 0.001, "Expected ~{}, got {}, error={}", expected, recovered, error);
    }

    // =========================================================================
    // TEST GROUP 4: Square activation (x² — the MPC-friendly ReLU replacement)
    // =========================================================================
    //
    // WHY x² INSTEAD OF ReLU?
    // ReLU = max(0, x) requires a COMPARISON on secret-shared values.
    // Comparisons in MPC require bit-decomposition: converting the shared integer
    // into individual shared bits, performing the comparison in a Boolean circuit,
    // then converting back. This costs 10-100x more than a multiplication.
    //
    // x² is just one multiplication. Same cost as any other fp_mul. The neural
    // network is trained with x² activations from the start, so accuracy is
    // maintained. Research (PolyMPCNet, CrypTen) shows <2% accuracy loss vs ReLU.

    #[test]
    fn test_square_activation_positive() {
        // 3.0² = 9.0
        let x = from_f64(3.0);
        let result = fp_square(x);
        assert_eq!(to_f64(result), 9.0);
    }

    #[test]
    fn test_square_activation_negative() {
        // (-4.0)² = 16.0 — always positive, which is the key property
        let x = from_f64(-4.0);
        let result = fp_square(x);
        assert_eq!(to_f64(result), 16.0);
    }

    #[test]
    fn test_square_activation_fraction() {
        // 0.5² = 0.25
        let x = from_f64(0.5);
        let result = fp_square(x);
        assert_eq!(to_f64(result), 0.25);
    }

    #[test]
    fn test_square_activation_zero() {
        let x = from_f64(0.0);
        let result = fp_square(x);
        assert_eq!(to_f64(result), 0.0);
    }

    // =========================================================================
    // TEST GROUP 5: Dot product (the core of neural net layers)
    // =========================================================================
    //
    // A single neuron computes: output = sum(input[i] * weight[i]) + bias
    // This is a dot product followed by addition. In fixed-point, each
    // multiplication needs truncation, and we accumulate in i64 to avoid overflow.

    #[test]
    fn test_dot_product_simple() {
        // [1.0, 2.0, 3.0] · [4.0, 5.0, 6.0] = 4 + 10 + 18 = 32.0
        let a = vec![from_f64(1.0), from_f64(2.0), from_f64(3.0)];
        let b = vec![from_f64(4.0), from_f64(5.0), from_f64(6.0)];
        let result = fp_dot(&a, &b);
        assert_eq!(to_f64(result), 32.0);
    }

    #[test]
    fn test_dot_product_with_bias() {
        // Simulates a single neuron: dot([0.5, -0.3], [0.7, 0.2]) + 0.1
        // = (0.35 + -0.06) + 0.1 = 0.39
        let inputs = vec![from_f64(0.5), from_f64(-0.3)];
        let weights = vec![from_f64(0.7), from_f64(0.2)];
        let bias = from_f64(0.1);
        let dot = fp_dot(&inputs, &weights);
        let result = fp_add(dot, bias);
        let recovered = to_f64(result);
        let expected = 0.5 * 0.7 + (-0.3) * 0.2 + 0.1;
        let error = (expected - recovered).abs();
        assert!(error < 0.001, "Expected ~{}, got {}", expected, recovered);
    }

    #[test]
    #[should_panic(expected = "mismatched")]
    fn test_dot_product_mismatched_lengths() {
        let a = vec![from_f64(1.0), from_f64(2.0)];
        let b = vec![from_f64(3.0)];
        fp_dot(&a, &b);
    }

    // =========================================================================
    // TEST GROUP 6: Full forward pass of a tiny network
    // =========================================================================
    //
    // This simulates what will run inside the Arcis circuit:
    // Layer 1: Linear(2 -> 2) + square activation
    // Layer 2: Linear(2 -> 1)
    //
    // If this test passes, we know our fixed-point math can execute a neural net.

    #[test]
    fn test_tiny_forward_pass() {
        // Input: [1.0, 2.0]
        //
        // Layer 1 weights: [[0.5, -0.3], [0.2, 0.8]]
        // Layer 1 bias: [0.1, -0.1]
        //
        // neuron_0 = (1.0*0.5 + 2.0*-0.3) + 0.1 = (0.5 - 0.6) + 0.1 = 0.0
        // neuron_1 = (1.0*0.2 + 2.0*0.8) + -0.1 = (0.2 + 1.6) - 0.1 = 1.7
        //
        // After square activation:
        // hidden_0 = 0.0² = 0.0
        // hidden_1 = 1.7² = 2.89
        //
        // Layer 2 weights: [[0.6], [0.4]]
        // Layer 2 bias: [-0.5]
        //
        // output = (0.0*0.6 + 2.89*0.4) + -0.5 = 1.156 - 0.5 = 0.656
        //
        let input = [from_f64(1.0), from_f64(2.0)];

        // Layer 1
        let w1 = [[from_f64(0.5), from_f64(-0.3)],
                   [from_f64(0.2), from_f64(0.8)]];
        let b1 = [from_f64(0.1), from_f64(-0.1)];

        let mut hidden = [0i32; 2];
        for i in 0..2 {
            let dot = fp_dot(&input, &w1[i]);
            hidden[i] = fp_square(fp_add(dot, b1[i])); // square activation
        }

        // Layer 2
        let w2 = [from_f64(0.6), from_f64(0.4)];
        let b2 = from_f64(-0.5);

        let output = fp_add(fp_dot(&hidden, &w2), b2);
        let result = to_f64(output);

        // Expected: 0.656 (allowing for fixed-point rounding)
        let expected = 0.656;
        let error = (expected - result).abs();
        assert!(error < 0.01, "Forward pass expected ~{}, got {}, error={}", expected, result, error);
    }
}
