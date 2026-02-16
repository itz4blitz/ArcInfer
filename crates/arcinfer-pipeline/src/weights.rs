/// Load trained model weights from JSON files exported by the Python training script.
///
/// The training script (training/train.py) exports two JSON files:
/// - pca.json: PCA mean vector (384-dim) and projection matrix (16×384)
/// - classifier_weights.json: Layer weights and biases for the 16→16→8→2 network
///
/// This module deserializes those files into Rust structs that can be
/// fed into arcinfer-core's quantize_linear and SentimentClassifier.

use serde::Deserialize;

/// PCA parameters: mean vector and projection matrix.
#[derive(Debug, Deserialize)]
pub struct PcaParams {
    pub input_dim: usize,
    pub output_dim: usize,
    pub mean: Vec<f64>,
    pub components: Vec<Vec<f64>>,
    pub explained_variance_ratio: Vec<f64>,
}

/// Raw classifier weights from the JSON export.
///
/// The JSON keys are PyTorch-style: "net.0.weight", "net.0.bias", etc.
/// We parse them into a flat struct with named layers.
#[derive(Debug)]
pub struct ClassifierWeights {
    pub layer1_weights: Vec<Vec<f64>>,
    pub layer1_biases: Vec<f64>,
    pub layer2_weights: Vec<Vec<f64>>,
    pub layer2_biases: Vec<f64>,
    pub layer3_weights: Vec<Vec<f64>>,
    pub layer3_biases: Vec<f64>,
}

/// Load PCA parameters from a JSON file.
pub fn load_pca_params(path: &str) -> Result<PcaParams, Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read PCA file {}: {}", path, e))?;
    let pca: PcaParams = serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse PCA JSON: {}", e))?;
    Ok(pca)
}

/// Load classifier weights from a JSON file.
///
/// The JSON format uses PyTorch naming:
/// - "net.0.weight" / "net.0.bias" → layer 1 (16→16)
/// - "net.2.weight" / "net.2.bias" → layer 2 (16→8)
/// - "net.4.weight" / "net.4.bias" → layer 3 (8→2)
///
/// (Indices 1 and 3 are the SquareActivation layers, which have no parameters.)
pub fn load_classifier_weights(path: &str) -> Result<ClassifierWeights, Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read weights file {}: {}", path, e))?;

    let raw: std::collections::HashMap<String, Vec<serde_json::Value>> =
        serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse weights JSON: {}", e))?;

    let parse_matrix = |key: &str| -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let rows = raw.get(key)
            .ok_or_else(|| format!("Missing key: {}", key))?;
        rows.iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| format!("Expected array for {}", key).into())
                    .and_then(|arr| {
                        arr.iter()
                            .map(|v| v.as_f64().ok_or_else(|| format!("Expected f64 in {}", key).into()))
                            .collect()
                    })
            })
            .collect()
    };

    let parse_vector = |key: &str| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let values = raw.get(key)
            .ok_or_else(|| format!("Missing key: {}", key))?;
        values.iter()
            .map(|v| v.as_f64().ok_or_else(|| format!("Expected f64 in {}", key).into()))
            .collect()
    };

    Ok(ClassifierWeights {
        layer1_weights: parse_matrix("net.0.weight")?,
        layer1_biases: parse_vector("net.0.bias")?,
        layer2_weights: parse_matrix("net.2.weight")?,
        layer2_biases: parse_vector("net.2.bias")?,
        layer3_weights: parse_matrix("net.4.weight")?,
        layer3_biases: parse_vector("net.4.bias")?,
    })
}
