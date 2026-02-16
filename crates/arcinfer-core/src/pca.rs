/// PCA (Principal Component Analysis) dimensionality reduction.
///
/// WHY PCA?
/// The sentence embedding model (all-MiniLM-L6-v2) outputs 384-dim vectors.
/// Running a classifier on 384-dim input inside MPC means Linear(384→N) has
/// 384 multiplications per output neuron. Reducing to 16-dim cuts this to 16 muls.
/// That's a 24x reduction in the most expensive layer.
///
/// HOW IT WORKS:
/// 1. Collect a training set of embeddings (e.g., 10K sentences from SST-2)
/// 2. Compute PCA in Python: fit a 384→16 projection matrix
/// 3. Export the projection matrix (16 rows × 384 cols) and mean vector (384-dim)
/// 4. Apply client-side: output = projection_matrix × (input - mean)
///
/// PCA runs CLIENT-SIDE only. It's a dimensionality reduction step before
/// encryption. The MPC circuit only ever sees 16-dim inputs.
///
/// The projection matrix and mean vector are PUBLIC — they're part of the model,
/// not sensitive data. Only the input embedding and classification result are secret.

use crate::fixed_point::fp_dot;

/// Subtract the training-set mean from an input vector (centering).
///
/// Standard PCA requires centered data: output = W × (x - μ)
/// This computes the (x - μ) part.
///
/// In fixed-point, subtraction is just integer subtraction — same as
/// fp_add but with negation. No MPC cost since this runs client-side.
pub fn subtract_mean<const N: usize>(input: &[i32; N], mean: &[i32; N]) -> [i32; N] {
    let mut result = [0i32; N];
    for i in 0..N {
        result[i] = input[i] - mean[i];
    }
    result
}

/// Project a vector through the PCA matrix (matrix-vector multiply).
///
/// `projection` is OUT rows × IN cols. Each row is a principal component.
/// Output dimension i = dot(projection[i], input).
///
/// This is structurally identical to Linear::forward without a bias term.
/// We keep it separate because PCA runs client-side (not in MPC) and
/// operates on different dimension pairs (384→16 vs the classifier's 16→16→8→2).
pub fn project<const IN: usize, const OUT: usize>(
    projection: &[[i32; IN]; OUT],
    input: &[i32; IN],
) -> [i32; OUT] {
    let mut output = [0i32; OUT];
    for j in 0..OUT {
        output[j] = fp_dot(&projection[j], input.as_slice());
    }
    output
}

/// Full PCA transform: center the input, then project.
///
/// output = projection_matrix × (input - mean)
///
/// This is the complete client-side preprocessing pipeline:
/// 1. Take a 384-dim embedding from the sentence transformer
/// 2. Subtract the training-set mean (384-dim)
/// 3. Multiply by the projection matrix (16×384)
/// 4. Get a 16-dim vector ready for the MPC classifier
pub fn pca_transform<const IN: usize, const OUT: usize>(
    input: &[i32; IN],
    mean: &[i32; IN],
    projection: &[[i32; IN]; OUT],
) -> [i32; OUT] {
    let centered = subtract_mean(input, mean);
    project(projection, &centered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::{from_f64, to_f64};

    // =========================================================================
    // TEST GROUP 1: Mean subtraction
    // =========================================================================
    //
    // PCA requires centering: subtract the training set mean from each input.
    // In standard PCA: output = W × (x - μ)
    // This step is trivial but important — without it, the projection is wrong.

    #[test]
    fn test_subtract_mean() {
        let input = [from_f64(3.0), from_f64(5.0), from_f64(1.0)];
        let mean = [from_f64(1.0), from_f64(2.0), from_f64(0.5)];
        let centered: [i32; 3] = subtract_mean(&input, &mean);

        assert_eq!(to_f64(centered[0]), 2.0);
        assert_eq!(to_f64(centered[1]), 3.0);
        assert_eq!(to_f64(centered[2]), 0.5);
    }

    #[test]
    fn test_subtract_mean_negative_result() {
        let input = [from_f64(1.0), from_f64(1.0)];
        let mean = [from_f64(3.0), from_f64(2.0)];
        let centered: [i32; 2] = subtract_mean(&input, &mean);

        assert_eq!(to_f64(centered[0]), -2.0);
        assert_eq!(to_f64(centered[1]), -1.0);
    }

    #[test]
    fn test_subtract_zero_mean() {
        // Zero mean = no change (useful for testing without PCA preprocessing)
        let input = [from_f64(4.2), from_f64(-1.3)];
        let mean = [from_f64(0.0), from_f64(0.0)];
        let centered: [i32; 2] = subtract_mean(&input, &mean);

        assert!((to_f64(centered[0]) - 4.2).abs() < 0.001);
        assert!((to_f64(centered[1]) - (-1.3)).abs() < 0.001);
    }

    // =========================================================================
    // TEST GROUP 2: PCA projection (the matrix-vector multiply)
    // =========================================================================

    #[test]
    fn test_project_identity() {
        // 3→3 identity projection should pass through (no dimensionality change)
        let projection = [
            [from_f64(1.0), from_f64(0.0), from_f64(0.0)],
            [from_f64(0.0), from_f64(1.0), from_f64(0.0)],
            [from_f64(0.0), from_f64(0.0), from_f64(1.0)],
        ];
        let input = [from_f64(2.0), from_f64(3.0), from_f64(4.0)];
        let output: [i32; 3] = project(&projection, &input);

        assert_eq!(to_f64(output[0]), 2.0);
        assert_eq!(to_f64(output[1]), 3.0);
        assert_eq!(to_f64(output[2]), 4.0);
    }

    #[test]
    fn test_project_reduces_dimensions() {
        // 3→2 projection: takes 3-dim input, outputs 2-dim
        // Row 0 of projection: [1, 0, 0] → extracts first component
        // Row 1 of projection: [0, 0.5, 0.5] → averages last two components
        let projection = [
            [from_f64(1.0), from_f64(0.0), from_f64(0.0)],
            [from_f64(0.0), from_f64(0.5), from_f64(0.5)],
        ];
        let input = [from_f64(6.0), from_f64(4.0), from_f64(2.0)];
        let output: [i32; 2] = project(&projection, &input);

        assert_eq!(to_f64(output[0]), 6.0);
        assert_eq!(to_f64(output[1]), 3.0); // (4+2)/2 = 3
    }

    #[test]
    fn test_project_with_negative_components() {
        // PCA principal components can have negative values
        let projection = [
            [from_f64(0.707), from_f64(0.707)],   // ~1/sqrt(2), 1/sqrt(2)
            [from_f64(0.707), from_f64(-0.707)],   // ~1/sqrt(2), -1/sqrt(2)
        ];
        let input = [from_f64(3.0), from_f64(1.0)];
        let output: [i32; 2] = project(&projection, &input);

        // output[0] = 0.707*3 + 0.707*1 = 2.828
        // output[1] = 0.707*3 - 0.707*1 = 1.414
        assert!((to_f64(output[0]) - 2.828).abs() < 0.01);
        assert!((to_f64(output[1]) - 1.414).abs() < 0.01);
    }

    // =========================================================================
    // TEST GROUP 3: Full PCA pipeline (mean subtraction + projection)
    // =========================================================================

    #[test]
    fn test_pca_full_pipeline() {
        // Input: 4-dim vector
        // Output: 2-dim vector
        // Steps: center by subtracting mean, then project

        let input = [from_f64(5.0), from_f64(3.0), from_f64(7.0), from_f64(1.0)];
        let mean = [from_f64(2.0), from_f64(1.0), from_f64(4.0), from_f64(0.0)];

        // After centering: [3.0, 2.0, 3.0, 1.0]

        let projection = [
            [from_f64(1.0), from_f64(0.0), from_f64(0.0), from_f64(0.0)], // extracts dim 0
            [from_f64(0.0), from_f64(0.0), from_f64(1.0), from_f64(0.0)], // extracts dim 2
        ];

        let output: [i32; 2] = pca_transform(&input, &mean, &projection);

        assert_eq!(to_f64(output[0]), 3.0);
        assert_eq!(to_f64(output[1]), 3.0);
    }

    #[test]
    fn test_pca_preserves_information() {
        // Two inputs that differ should produce different outputs
        let mean = [from_f64(0.0), from_f64(0.0), from_f64(0.0)];
        let projection = [
            [from_f64(1.0), from_f64(0.0), from_f64(0.0)],
            [from_f64(0.0), from_f64(1.0), from_f64(0.0)],
        ];

        let input_a = [from_f64(1.0), from_f64(2.0), from_f64(9.0)];
        let input_b = [from_f64(2.0), from_f64(1.0), from_f64(9.0)];

        let out_a: [i32; 2] = pca_transform(&input_a, &mean, &projection);
        let out_b: [i32; 2] = pca_transform(&input_b, &mean, &projection);

        // dim 2 (value 9.0) is dropped by PCA — only dims 0 and 1 survive
        // But dims 0 and 1 differ between inputs, so outputs should differ
        assert_ne!(out_a[0], out_b[0]);
        assert_ne!(out_a[1], out_b[1]);
    }

    // =========================================================================
    // TEST GROUP 4: Realistic dimensions (small scale test of 8→3)
    // =========================================================================
    //
    // Can't test the full 384→16 here (would need a real PCA matrix),
    // but we can verify the pattern at a smaller scale.

    #[test]
    fn test_pca_8_to_3() {
        let input: [i32; 8] = [
            from_f64(0.1), from_f64(0.2), from_f64(0.3), from_f64(0.4),
            from_f64(0.5), from_f64(0.6), from_f64(0.7), from_f64(0.8),
        ];
        let mean: [i32; 8] = [from_f64(0.0); 8]; // zero mean for simplicity

        // Projection: 3 rows × 8 cols
        // Each row just picks one dimension (simplest possible projection)
        let projection: [[i32; 8]; 3] = [
            {
                let mut row = [from_f64(0.0); 8];
                row[0] = from_f64(1.0);
                row
            },
            {
                let mut row = [from_f64(0.0); 8];
                row[3] = from_f64(1.0);
                row
            },
            {
                let mut row = [from_f64(0.0); 8];
                row[7] = from_f64(1.0);
                row
            },
        ];

        let output: [i32; 3] = pca_transform(&input, &mean, &projection);

        assert!((to_f64(output[0]) - 0.1).abs() < 0.001);  // picked dim 0
        assert!((to_f64(output[1]) - 0.4).abs() < 0.001);  // picked dim 3
        assert!((to_f64(output[2]) - 0.8).abs() < 0.001);  // picked dim 7
    }
}
