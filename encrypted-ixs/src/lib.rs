/// ArcInfer Encrypted Sentiment Classifier
///
/// This is the MPC circuit that runs on Arcium's network. It takes a
/// Q16.16 fixed-point embedding vector (16 dims, already PCA-reduced
/// by the client) and classifies it as positive (1) or negative (0).
///
/// Architecture: Linear(16→16) → x² → Linear(16→8) → x² → Linear(8→2) → argmax
///
/// The weights are PUBLIC compile-time constants (426 parameters).
/// Only the input features and output classification are encrypted.
///
/// MPC cost: ~10 rounds (2 per linear layer + 2 per square activation)

use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    // =========================================================================
    // Layer 1: 16 → 16 (256 weights + 16 biases)
    // =========================================================================

    const L1_WEIGHTS: [[i32; 16]; 16] = [
        [28846, 13857, -19431, 15041, -14823, 17251, -50414, 402, 34778, 14376, 31630, -7093, -12360, 1439, -22207, -13880],
        [16366, 1687, -62261, 29357, 9792, -46006, 154, 40636, 23258, -37992, 27993, -31891, 30295, -56350, -5847, -9454],
        [31662, 26135, -47280, 30751, -2272, 9895, 24374, -18496, 5297, -1142, -19755, 15705, 7957, 5499, 8453, 4942],
        [19408, -13706, -45598, -20054, -25179, 2299, -2312, 10904, 17398, 22943, -9015, -57774, -3555, -5580, -6428, -29114],
        [7529, -5206, 53664, -5255, -17691, -22177, -11877, 2307, 17289, 15335, -7856, 17538, -53109, -25624, -15821, 52013],
        [-12242, -41886, 39352, 1307, 34513, -10428, 2584, -22582, -37128, -6616, 28730, 8450, 18230, -9044, -15909, -1407],
        [17774, -14619, -29814, -50114, -12975, 4601, -23955, 6606, -923, 7740, -8287, -6633, -9151, 5370, 10556, -16712],
        [34290, -37932, 11234, 48929, -64516, -23860, -4844, -35147, 1414, -56451, 38290, 1262, 2749, -13653, -7176, 44768],
        [5442, 16489, 50529, -4829, -8470, 28424, -55342, 22499, -18162, -31772, 12332, 27837, 24915, -33320, 14915, -16850],
        [-8351, -31242, -14426, -7101, 36317, -13249, -880, 9301, -17507, -4742, -12169, -22866, -5968, 2786, -22618, -30207],
        [35770, 7072, 33183, -57854, -770, 4660, -12503, 47988, 18389, -12495, -16104, 19693, 20224, 34831, 7612, 31430],
        [-25816, 345, 30482, -28584, -23972, -10422, -19339, -7002, 17005, 4401, 25579, 9527, -6208, -3756, 6842, -12862],
        [36458, 39716, 4019, 13897, -26406, 12212, 7556, -48077, -11419, 8163, 18496, 35582, 9468, -27825, 15058, -7537],
        [-8779, 25466, 43210, 33330, -29449, -90, -8111, 19700, 36297, -11465, 6139, -17974, -18290, 10965, 17570, 16719],
        [-53606, 22823, -15339, -21904, -61319, -24924, 7584, 12314, 38208, 27938, -37167, -12320, -17088, -32203, -3670, -50907],
        [18767, -10910, 42443, -1194, -101, 403, -46266, -29476, -6912, 29285, 542, -51335, 20321, -28089, 5156, -27528],
    ];

    const L1_BIASES: [i32; 16] = [25866, -20128, 22129, 24503, 20775, -28192, 23496, -27924, 27145, 23672, -18141, -14906, -19111, -23460, 17371, -25348];

    // =========================================================================
    // Layer 2: 16 → 8 (128 weights + 8 biases)
    // =========================================================================

    const L2_WEIGHTS: [[i32; 16]; 8] = [
        [-12526, 53997, -9960, -30960, 14641, -21275, -15584, -30340, 29461, -30302, -35445, -29775, -35231, -34201, -47879, -10660],
        [36979, -40108, 21772, 17984, -39239, 9828, 28043, 9159, -45685, 19032, 31349, 13605, 31531, 21749, 26633, 16499],
        [-22459, -8040, -4216, 10928, -33865, 238, 3632, -3602, -46240, -7631, 25468, 16062, 8878, 3872, -9381, -502],
        [-17444, -34786, 22415, 29385, -39913, -9903, -20768, -35606, -46066, -11702, 34287, 22154, -3716, -24006, 3465, -10375],
        [-7029, -18046, -383, 18025, -20646, 164, -11143, -35335, -43300, -15814, 33954, 10249, -2662, -3595, 6572, 10185],
        [-885, 3876, -9706, -120, 2762, 604, -15210, 51241, -33012, -1604, 7267, -4440, 29047, 14426, -391, -333],
        [-21720, -22503, 22323, 24769, -29049, 747, -15264, -27888, -26354, 3259, 30824, -2354, 13963, -19272, 8600, 12732],
        [-25993, 54339, -40928, -30173, 29594, -33135, -25897, -27347, 41597, -9757, -27695, -29207, -41161, -21193, -26305, -37094],
    ];

    const L2_BIASES: [i32; 8] = [-14886, 19776, -18498, -16501, -15659, -19867, -22930, -19017];

    // =========================================================================
    // Layer 3: 8 → 2 (16 weights + 2 biases)
    // =========================================================================

    const L3_WEIGHTS: [[i32; 8]; 2] = [
        [-37176, -16235, 9118, 14677, 39649, 27496, 29371, -39079],
        [15934, 31970, -31302, -38889, -18677, -27900, -30223, 41605],
    ];

    const L3_BIASES: [i32; 2] = [21052, 1340];

    // =========================================================================
    // Structs
    // =========================================================================

    pub struct SentimentInput {
        pub features: [i32; 16],
    }

    pub struct ClassificationResult {
        pub class: u8,
        pub logit_positive: i32,
        pub logit_negative: i32,
    }

    // =========================================================================
    // Instructions
    // =========================================================================

    /// Encrypted sentiment classification.
    ///
    /// Takes an encrypted 16-dim Q16.16 feature vector, runs the 3-layer
    /// classifier with square activations, and returns the encrypted
    /// classification result.
    #[instruction]
    pub fn classify(input: Enc<Shared, SentimentInput>) -> Enc<Shared, ClassificationResult> {
        let features = input.to_arcis();

        // Layer 1: Linear(16 → 16) + bias
        let mut h1 = [0i32; 16];
        for j in 0..16 {
            let mut acc: i64 = 0;
            for i in 0..16 {
                acc += (L1_WEIGHTS[j][i] as i64) * (features.features[i] as i64);
            }
            h1[j] = ((acc >> 16) as i32) + L1_BIASES[j];
        }

        // Square activation
        for j in 0..16 {
            let val = h1[j] as i64;
            h1[j] = ((val * val) >> 16) as i32;
        }

        // Layer 2: Linear(16 → 8) + bias
        let mut h2 = [0i32; 8];
        for j in 0..8 {
            let mut acc: i64 = 0;
            for i in 0..16 {
                acc += (L2_WEIGHTS[j][i] as i64) * (h1[i] as i64);
            }
            h2[j] = ((acc >> 16) as i32) + L2_BIASES[j];
        }

        // Square activation
        for j in 0..8 {
            let val = h2[j] as i64;
            h2[j] = ((val * val) >> 16) as i32;
        }

        // Layer 3: Linear(8 → 2) + bias (output logits)
        let mut logits = [0i32; 2];
        for j in 0..2 {
            let mut acc: i64 = 0;
            for i in 0..8 {
                acc += (L3_WEIGHTS[j][i] as i64) * (h2[i] as i64);
            }
            logits[j] = ((acc >> 16) as i32) + L3_BIASES[j];
        }

        // Argmax: determine class (0=negative, 1=positive)
        let class = if logits[1] > logits[0] { 1u8 } else { 0u8 };

        // Re-encrypt result for the original client
        input.owner.from_arcis(ClassificationResult {
            class,
            logit_positive: logits[1],
            logit_negative: logits[0],
        })
    }

    /// Classify and reveal — returns the class label as plaintext.
    #[instruction]
    pub fn classify_reveal(input: Enc<Shared, SentimentInput>) -> u8 {
        let features = input.to_arcis();

        // Layer 1: Linear(16 → 16) + square activation
        let mut h1 = [0i32; 16];
        for j in 0..16 {
            let mut acc: i64 = 0;
            for i in 0..16 {
                acc += (L1_WEIGHTS[j][i] as i64) * (features.features[i] as i64);
            }
            h1[j] = ((acc >> 16) as i32) + L1_BIASES[j];
        }
        for j in 0..16 {
            let val = h1[j] as i64;
            h1[j] = ((val * val) >> 16) as i32;
        }

        // Layer 2: Linear(16 → 8) + square activation
        let mut h2 = [0i32; 8];
        for j in 0..8 {
            let mut acc: i64 = 0;
            for i in 0..16 {
                acc += (L2_WEIGHTS[j][i] as i64) * (h1[i] as i64);
            }
            h2[j] = ((acc >> 16) as i32) + L2_BIASES[j];
        }
        for j in 0..8 {
            let val = h2[j] as i64;
            h2[j] = ((val * val) >> 16) as i32;
        }

        // Layer 3: Linear(8 → 2) (output logits)
        let mut logits = [0i32; 2];
        for j in 0..2 {
            let mut acc: i64 = 0;
            for i in 0..8 {
                acc += (L3_WEIGHTS[j][i] as i64) * (h2[i] as i64);
            }
            logits[j] = ((acc >> 16) as i32) + L3_BIASES[j];
        }

        // Argmax and reveal
        let class = if logits[1] > logits[0] { 1u8 } else { 0u8 };
        class.reveal()
    }
}
