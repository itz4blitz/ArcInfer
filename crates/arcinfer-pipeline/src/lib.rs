/// End-to-end pipeline: load trained models, classify text.
///
/// This crate ties everything together:
/// - Loads PCA parameters from JSON (exported by Python training)
/// - Loads classifier weights from JSON (exported by Python training)
/// - Uses arcinfer-inference to generate embeddings
/// - Uses arcinfer-core to quantize and classify
///
/// This is the final integration point. If these tests pass with real
/// trained weights, the entire pipeline is proven correct.

pub mod weights;

#[cfg(test)]
mod tests {
    use crate::weights::{load_pca_params, load_classifier_weights};
    use arcinfer_core::classifier::SentimentClassifier;
    use arcinfer_core::fixed_point::from_f64;
    use arcinfer_core::quantize::quantize_linear;
    use arcinfer_inference::pipeline::InferencePipeline;

    fn models_dir() -> String {
        format!("{}/../../models", env!("CARGO_MANIFEST_DIR"))
    }

    fn tokenizer_path() -> String {
        format!("{}/tokenizer.json", models_dir())
    }

    fn model_path() -> String {
        format!("{}/model.onnx", models_dir())
    }

    fn pca_path() -> String {
        format!("{}/pca.json", models_dir())
    }

    fn weights_path() -> String {
        format!("{}/classifier_weights.json", models_dir())
    }

    // =========================================================================
    // TEST GROUP 0: Error paths for weight loading
    // =========================================================================

    #[test]
    fn test_load_pca_bad_path() {
        let result = load_pca_params("/nonexistent/pca.json");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Failed to read"), "Expected read error, got: {}", err);
    }

    #[test]
    fn test_load_pca_invalid_json() {
        let dir = std::env::temp_dir();
        let path = dir.join("bad_pca.json");
        std::fs::write(&path, "not valid json").unwrap();
        let result = load_pca_params(path.to_str().unwrap());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Failed to parse"), "Expected parse error, got: {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_classifier_bad_path() {
        let result = load_classifier_weights("/nonexistent/weights.json");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Failed to read"), "Expected read error, got: {}", err);
    }

    #[test]
    fn test_load_classifier_invalid_json() {
        let dir = std::env::temp_dir();
        let path = dir.join("bad_weights.json");
        std::fs::write(&path, "not valid json").unwrap();
        let result = load_classifier_weights(path.to_str().unwrap());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Failed to parse"), "Expected parse error, got: {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_classifier_missing_key() {
        let dir = std::env::temp_dir();
        let path = dir.join("partial_weights.json");
        // Valid JSON but missing required keys
        std::fs::write(&path, r#"{"net.0.weight": [[1.0]], "net.0.bias": [1.0]}"#).unwrap();
        let result = load_classifier_weights(path.to_str().unwrap());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Missing key"), "Expected missing key error, got: {}", err);
        std::fs::remove_file(&path).ok();
    }

    // =========================================================================
    // TEST GROUP 1: Weight loading from JSON
    // =========================================================================

    #[test]
    fn test_load_pca_params() {
        let pca = load_pca_params(&pca_path()).unwrap();

        assert_eq!(pca.mean.len(), 384, "PCA mean should be 384-dim");
        assert_eq!(pca.components.len(), 16, "PCA should have 16 components");
        assert_eq!(pca.components[0].len(), 384, "Each component should be 384-dim");
    }

    #[test]
    fn test_load_classifier_weights() {
        let w = load_classifier_weights(&weights_path()).unwrap();

        // Layer 1: 16→16
        assert_eq!(w.layer1_weights.len(), 16);
        assert_eq!(w.layer1_weights[0].len(), 16);
        assert_eq!(w.layer1_biases.len(), 16);
        // Layer 2: 16→8
        assert_eq!(w.layer2_weights.len(), 8);
        assert_eq!(w.layer2_weights[0].len(), 16);
        assert_eq!(w.layer2_biases.len(), 8);
        // Layer 3: 8→2
        assert_eq!(w.layer3_weights.len(), 2);
        assert_eq!(w.layer3_weights[0].len(), 8);
        assert_eq!(w.layer3_biases.len(), 2);
    }

    #[test]
    fn test_weights_in_q16_range() {
        let w = load_classifier_weights(&weights_path()).unwrap();

        // All weights should be within Q16.16 range
        for row in &w.layer1_weights {
            for &val in row {
                assert!(val.abs() < 32767.0, "Weight out of Q16.16 range: {}", val);
            }
        }
        for row in &w.layer2_weights {
            for &val in row {
                assert!(val.abs() < 32767.0, "Weight out of Q16.16 range: {}", val);
            }
        }
    }

    // =========================================================================
    // TEST GROUP 2: Build classifier from trained weights
    // =========================================================================

    #[test]
    fn test_build_classifier_from_trained_weights() {
        let w = load_classifier_weights(&weights_path()).unwrap();

        let l1_refs: Vec<&[f64]> = w.layer1_weights.iter().map(|r| r.as_slice()).collect();
        let l2_refs: Vec<&[f64]> = w.layer2_weights.iter().map(|r| r.as_slice()).collect();
        let l3_refs: Vec<&[f64]> = w.layer3_weights.iter().map(|r| r.as_slice()).collect();

        let classifier = SentimentClassifier {
            layer1: quantize_linear(&l1_refs, &w.layer1_biases),
            layer2: quantize_linear(&l2_refs, &w.layer2_biases),
            layer3: quantize_linear(&l3_refs, &w.layer3_biases),
        };

        // Just verify it doesn't panic — we test correctness in the full pipeline
        let zero_input = [0i32; 16];
        let logits = classifier.forward(&zero_input);
        assert_eq!(logits.len(), 2);
    }

    // =========================================================================
    // TEST GROUP 2b: Verify generated circuit weights match JSON weights
    // =========================================================================
    //
    // The Arcis circuit uses hardcoded const arrays (generated by
    // scripts/generate_circuit_weights.py). These tests verify the
    // generated constants match what from_f64() produces from the
    // JSON weights within ±1 (floating-point rounding tolerance).
    // This proves the circuit uses equivalent weights to arcinfer-core's
    // reference implementation. A ±1 difference in Q16.16 is <0.00002
    // in real value — completely negligible for classification.

    fn circuit_weights_path() -> String {
        format!("{}/../../encrypted-ixs/src/weights.rs", env!("CARGO_MANIFEST_DIR"))
    }

    /// Parse a 2D const array from the generated Rust source.
    /// Extracts i32 values from lines like "    [19805, 4902, -33754, ...],"
    fn parse_const_matrix(source: &str, name: &str) -> Vec<Vec<i32>> {
        let start_marker = format!("pub const {}: [[i32;", name);
        let mut in_array = false;
        let mut rows = Vec::new();

        for line in source.lines() {
            if line.contains(&start_marker) {
                in_array = true;
                continue;
            }
            if in_array {
                let trimmed = line.trim();
                if trimmed == "];" {
                    break;
                }
                if trimmed.starts_with('[') {
                    let inner = trimmed.trim_start_matches('[').trim_end_matches("],").trim_end_matches(']');
                    let vals: Vec<i32> = inner.split(',')
                        .map(|s| s.trim().parse::<i32>().unwrap())
                        .collect();
                    rows.push(vals);
                }
            }
        }
        rows
    }

    /// Parse a 1D const array from the generated Rust source.
    fn parse_const_vector(source: &str, name: &str) -> Vec<i32> {
        let start_marker = format!("pub const {}: [i32;", name);
        for line in source.lines() {
            if line.contains(&start_marker) {
                let bracket_start = line.find('[').unwrap();
                // Find the inner brackets (skip the type brackets)
                let inner_start = line[bracket_start + 1..].find('[').unwrap() + bracket_start + 2;
                let inner_end = line.rfind(']').unwrap();
                let inner = &line[inner_start..inner_end];
                return inner.split(',')
                    .map(|s| s.trim().parse::<i32>().unwrap())
                    .collect();
            }
        }
        panic!("Constant {} not found", name);
    }

    #[test]
    fn test_circuit_weights_match_json_layer1() {
        let w = load_classifier_weights(&weights_path()).unwrap();
        let source = std::fs::read_to_string(circuit_weights_path()).unwrap();

        let circuit_l1_w = parse_const_matrix(&source, "L1_WEIGHTS");
        let circuit_l1_b = parse_const_vector(&source, "L1_BIASES");

        // Verify dimensions
        assert_eq!(circuit_l1_w.len(), 16);
        assert_eq!(circuit_l1_w[0].len(), 16);
        assert_eq!(circuit_l1_b.len(), 16);

        // Verify every weight matches from_f64() quantization
        for (row_idx, (circuit_row, json_row)) in
            circuit_l1_w.iter().zip(w.layer1_weights.iter()).enumerate()
        {
            for (col_idx, (&circuit_val, &json_val)) in
                circuit_row.iter().zip(json_row.iter()).enumerate()
            {
                let expected = from_f64(json_val);
                assert!(
                    (circuit_val - expected).abs() <= 1,
                    "L1 weight mismatch at [{row_idx}][{col_idx}]: circuit={circuit_val}, expected={expected}"
                );
            }
        }

        for (idx, (&circuit_val, &json_val)) in
            circuit_l1_b.iter().zip(w.layer1_biases.iter()).enumerate()
        {
            let expected = from_f64(json_val);
            assert!(
                (circuit_val - expected).abs() <= 1,
                "L1 bias mismatch at [{idx}]: circuit={circuit_val}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_circuit_weights_match_json_layer2() {
        let w = load_classifier_weights(&weights_path()).unwrap();
        let source = std::fs::read_to_string(circuit_weights_path()).unwrap();

        let circuit_l2_w = parse_const_matrix(&source, "L2_WEIGHTS");
        let circuit_l2_b = parse_const_vector(&source, "L2_BIASES");

        assert_eq!(circuit_l2_w.len(), 8);
        assert_eq!(circuit_l2_w[0].len(), 16);
        assert_eq!(circuit_l2_b.len(), 8);

        for (row_idx, (circuit_row, json_row)) in
            circuit_l2_w.iter().zip(w.layer2_weights.iter()).enumerate()
        {
            for (col_idx, (&circuit_val, &json_val)) in
                circuit_row.iter().zip(json_row.iter()).enumerate()
            {
                let expected = from_f64(json_val);
                assert!(
                    (circuit_val - expected).abs() <= 1,
                    "L2 weight mismatch at [{row_idx}][{col_idx}]: circuit={circuit_val}, expected={expected}"
                );
            }
        }

        for (idx, (&circuit_val, &json_val)) in
            circuit_l2_b.iter().zip(w.layer2_biases.iter()).enumerate()
        {
            let expected = from_f64(json_val);
            assert!(
                (circuit_val - expected).abs() <= 1,
                "L2 bias mismatch at [{idx}]: circuit={circuit_val}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_circuit_weights_match_json_layer3() {
        let w = load_classifier_weights(&weights_path()).unwrap();
        let source = std::fs::read_to_string(circuit_weights_path()).unwrap();

        let circuit_l3_w = parse_const_matrix(&source, "L3_WEIGHTS");
        let circuit_l3_b = parse_const_vector(&source, "L3_BIASES");

        assert_eq!(circuit_l3_w.len(), 2);
        assert_eq!(circuit_l3_w[0].len(), 8);
        assert_eq!(circuit_l3_b.len(), 2);

        for (row_idx, (circuit_row, json_row)) in
            circuit_l3_w.iter().zip(w.layer3_weights.iter()).enumerate()
        {
            for (col_idx, (&circuit_val, &json_val)) in
                circuit_row.iter().zip(json_row.iter()).enumerate()
            {
                let expected = from_f64(json_val);
                assert!(
                    (circuit_val - expected).abs() <= 1,
                    "L3 weight mismatch at [{row_idx}][{col_idx}]: circuit={circuit_val}, expected={expected}"
                );
            }
        }

        for (idx, (&circuit_val, &json_val)) in
            circuit_l3_b.iter().zip(w.layer3_biases.iter()).enumerate()
        {
            let expected = from_f64(json_val);
            assert!(
                (circuit_val - expected).abs() <= 1,
                "L3 bias mismatch at [{idx}]: circuit={circuit_val}, expected={expected}"
            );
        }
    }

    // =========================================================================
    // TEST GROUP 3: Full end-to-end pipeline
    // =========================================================================
    //
    // Text → tokenize → embed → PCA → quantize → classify
    // This is the ultimate integration test.

    #[test]
    fn test_end_to_end_positive_sentiment() {
        let pipeline = InferencePipeline::load(&tokenizer_path(), &model_path()).unwrap();
        let pca = load_pca_params(&pca_path()).unwrap();
        let w = load_classifier_weights(&weights_path()).unwrap();

        // Build PCA arrays
        let mean: [f64; 384] = pca.mean.try_into().expect("mean should be 384-dim");
        let mut projection = [[0.0f64; 384]; 16];
        for (i, row) in pca.components.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                projection[i][j] = val;
            }
        }

        // Build classifier
        let l1_refs: Vec<&[f64]> = w.layer1_weights.iter().map(|r| r.as_slice()).collect();
        let l2_refs: Vec<&[f64]> = w.layer2_weights.iter().map(|r| r.as_slice()).collect();
        let l3_refs: Vec<&[f64]> = w.layer3_weights.iter().map(|r| r.as_slice()).collect();
        let classifier = SentimentClassifier {
            layer1: quantize_linear(&l1_refs, &w.layer1_biases),
            layer2: quantize_linear(&l2_refs, &w.layer2_biases),
            layer3: quantize_linear(&l3_refs, &w.layer3_biases),
        };

        let input = pipeline.embed_quantized(
            "This movie is absolutely wonderful and I loved every moment",
            &mean,
            &projection,
        );
        let class = classifier.classify(&input);
        assert_eq!(class, 1, "Expected positive sentiment (1)");
    }

    #[test]
    fn test_end_to_end_negative_sentiment() {
        let pipeline = InferencePipeline::load(&tokenizer_path(), &model_path()).unwrap();
        let pca = load_pca_params(&pca_path()).unwrap();
        let w = load_classifier_weights(&weights_path()).unwrap();

        let mean: [f64; 384] = pca.mean.try_into().expect("mean should be 384-dim");
        let mut projection = [[0.0f64; 384]; 16];
        for (i, row) in pca.components.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                projection[i][j] = val;
            }
        }

        let l1_refs: Vec<&[f64]> = w.layer1_weights.iter().map(|r| r.as_slice()).collect();
        let l2_refs: Vec<&[f64]> = w.layer2_weights.iter().map(|r| r.as_slice()).collect();
        let l3_refs: Vec<&[f64]> = w.layer3_weights.iter().map(|r| r.as_slice()).collect();
        let classifier = SentimentClassifier {
            layer1: quantize_linear(&l1_refs, &w.layer1_biases),
            layer2: quantize_linear(&l2_refs, &w.layer2_biases),
            layer3: quantize_linear(&l3_refs, &w.layer3_biases),
        };

        let input = pipeline.embed_quantized(
            "Terrible film. Boring and a complete waste of time.",
            &mean,
            &projection,
        );
        let class = classifier.classify(&input);
        assert_eq!(class, 0, "Expected negative sentiment (0)");
    }

    #[test]
    fn test_end_to_end_batch_accuracy() {
        let pipeline = InferencePipeline::load(&tokenizer_path(), &model_path()).unwrap();
        let pca = load_pca_params(&pca_path()).unwrap();
        let w = load_classifier_weights(&weights_path()).unwrap();

        let mean: [f64; 384] = pca.mean.try_into().expect("mean should be 384-dim");
        let mut projection = [[0.0f64; 384]; 16];
        for (i, row) in pca.components.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                projection[i][j] = val;
            }
        }

        let l1_refs: Vec<&[f64]> = w.layer1_weights.iter().map(|r| r.as_slice()).collect();
        let l2_refs: Vec<&[f64]> = w.layer2_weights.iter().map(|r| r.as_slice()).collect();
        let l3_refs: Vec<&[f64]> = w.layer3_weights.iter().map(|r| r.as_slice()).collect();
        let classifier = SentimentClassifier {
            layer1: quantize_linear(&l1_refs, &w.layer1_biases),
            layer2: quantize_linear(&l2_refs, &w.layer2_biases),
            layer3: quantize_linear(&l3_refs, &w.layer3_biases),
        };

        // Test batch — mix of positive and negative
        let test_cases: &[(&str, usize)] = &[
            ("A beautiful and moving film that touched my heart", 1),
            ("The acting was phenomenal and the story was gripping", 1),
            ("Worst movie I have ever seen, absolutely dreadful", 0),
            ("Dull, uninspired, and painfully slow", 0),
            ("An instant classic that everyone should watch", 1),
            ("I walked out halfway through, it was that bad", 0),
        ];

        let mut correct = 0;
        for (text, expected) in test_cases {
            let input = pipeline.embed_quantized(text, &mean, &projection);
            let predicted = classifier.classify(&input);
            if predicted == *expected {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / test_cases.len() as f64;
        assert!(
            accuracy >= 0.66,
            "End-to-end accuracy too low: {}/{} ({:.0}%). Expected >=66%",
            correct, test_cases.len(), accuracy * 100.0
        );
    }
}
