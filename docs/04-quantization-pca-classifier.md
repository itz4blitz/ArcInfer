# Quantization, PCA, and the Classifier

The three modules that sit on top of the math foundation. Each runs at a different stage of the pipeline, and understanding *where* each one executes is just as important as understanding *what* it does.

## The Three Execution Contexts

Building for MPC forces you to think carefully about when and where code runs:

| Module | When | Where | Inputs | Why it matters |
|--------|------|-------|--------|---------------|
| `quantize.rs` | Once, offline | Developer's machine | PyTorch f64 weights | Weights are PUBLIC constants, not secrets |
| `pca.rs` | Every request | Client device | 384-dim plaintext embedding | Reduces MPC cost by 6x |
| `classifier.rs` | Every request | MPC cluster | 16-dim encrypted input | The only code that touches secrets |

Getting this wrong is the most common mistake I can imagine someone making with MPC. If you accidentally put PCA inside the circuit, you're doing 384-dim matrix multiplies on encrypted data for no reason. If you try to quantize weights inside the circuit, you're wasting MPC rounds on a conversion that only needs to happen once.

## Weight Quantization: The PyTorch Bridge

### The Problem

I train the classifier in Python with PyTorch. PyTorch uses f32/f64 floating point. The Arcis circuit uses Q16.16 fixed-point i32. I need a way to convert entire weight matrices while catching values that would overflow.

### The Solution

Three functions that build on each other:

```rust
// Convert a single weight vector
pub fn quantize_vec<const N: usize>(values: &[f64]) -> [i32; N]

// Convert a weight matrix (OUT rows × IN cols)
pub fn quantize_matrix<const IN: usize, const OUT: usize>(rows: &[&[f64]]) -> [[i32; IN]; OUT]

// Convert directly into a ready-to-use Linear layer
pub fn quantize_linear<const IN: usize, const OUT: usize>(
    weight_rows: &[&[f64]],
    biases: &[f64],
) -> Linear<IN, OUT>
```

### Range Validation

Q16.16 can represent values in [-32768, 32767]. Neural net weights are typically in [-2, 2], so this is plenty of headroom. But I added explicit range checks that panic if any value is outside the safe range:

```rust
assert!(
    values[i] >= -32768.0 && values[i] <= 32767.0,
    "value {} ({}) out of Q16.16 range", i, values[i]
);
```

Why panic instead of clipping? Because wrong weights in an MPC circuit are nearly impossible to debug. The circuit runs on encrypted data — you can't inspect intermediate values during execution. If a weight overflows and wraps around, the classifier will produce garbage results, and you'll have no idea why. Better to catch it loudly during the offline quantization step.

### Quantization Error

The roundtrip error for typical neural net weights is less than 0.0001:

```
Original:  0.3427  →  Quantized: 22462  →  Recovered: 0.34271...  →  Error: ~0.00001
Original: -0.8912  →  Quantized: -58408  →  Recovered: -0.89120...  →  Error: ~0.00001
```

Q16.16 gives us ~15 bits of fractional precision (1/65536 ≈ 0.000015). For a classifier that just needs to tell "positive" from "negative" sentiment, this is more than enough. The model's accuracy is limited by the training data and architecture, not by 5 digits of weight precision.

## PCA: Client-Side Dimensionality Reduction

### Why PCA?

The sentence transformer (all-MiniLM-L6-v2) outputs 384-dimensional embeddings. Running the classifier directly on 384-dim input means the first layer, Linear(384→16), does:

```
384 multiplications × 16 output neurons = 6,144 multiplications
```

Each multiplication costs 2 MPC rounds (Beaver triple + truncation). That's ~12,288 rounds just for layer 1.

PCA reduces the input to 16 dimensions first:

```
16 multiplications × 16 output neurons = 256 multiplications
```

That's **24x fewer multiplications** in the most expensive layer. And PCA runs client-side on plaintext, so it costs zero MPC rounds.

### The Math

Standard PCA transform:

```
output = W × (x - μ)
```

Where:
- `x` is the 384-dim input embedding
- `μ` is the training-set mean (384-dim), computed offline from the training data
- `W` is the projection matrix (16 rows × 384 cols), the top 16 principal components

Both `μ` and `W` are PUBLIC. They're part of the model, computed from the training set. The only secret is the input embedding `x`.

### The Implementation

I split it into three functions for testability:

1. **`subtract_mean`**: Element-wise `input[i] - mean[i]`. Trivial but mathematically necessary — without centering, the projection doesn't work correctly. In fixed-point, this is just integer subtraction.

2. **`project`**: Matrix-vector multiply. Each output dimension is the dot product of one projection row with the centered input. Structurally identical to `Linear::forward` without the bias term.

3. **`pca_transform`**: The pipeline — center first, then project.

### Why Not Reuse Linear::forward?

I could have implemented PCA as a `Linear<384, 16>` with zero biases. I kept it separate for two reasons:

1. **Conceptual clarity**: PCA is preprocessing, not part of the neural network. In the Arcis circuit, the classifier is 16→16→8→2. PCA never touches encrypted data.

2. **Different lifecycle**: The PCA matrix comes from sklearn/numpy on the training set. The classifier weights come from PyTorch training. They're computed by different tools at different times.

When I explain this code in an interview, I want the separation to be obvious.

## The Production Classifier

### Architecture

```
Input [16] → Linear(16→16) → x² → Linear(16→8) → x² → Linear(8→2) → Output [2]
```

This is the exact computation that runs inside the Arcis `#[instruction]` function on secret-shared data.

### Parameter Count

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Linear(16→16) | 16 × 16 = 256 | 16 | 272 |
| Linear(16→8) | 16 × 8 = 128 | 8 | 136 |
| Linear(8→2) | 8 × 2 = 16 | 2 | 18 |
| **Total** | | | **426** |

426 parameters. For comparison, DistilBERT has 67 million. The entire classifier fits in under 2KB of Q16.16 values.

### MPC Cost Analysis

Each operation type has a different cost in MPC rounds:

| Operation | Count | Rounds per op | Total rounds |
|-----------|-------|--------------|--------------|
| Linear(16→16) | 256 muls + 16 adds | 2 | 2 |
| Square activation (16) | 16 muls | 2 | 2 |
| Linear(16→8) | 128 muls + 8 adds | 2 | 2 |
| Square activation (8) | 8 muls | 2 | 2 |
| Linear(8→2) | 16 muls + 2 adds | 2 | 2 |
| **Total** | | | **~10 rounds** |

The multiplications within each layer happen in parallel (they don't depend on each other). The sequential depth is just the number of layers × 2 rounds per layer. 10 rounds at ~100ms per round ≈ 1 second for the entire classification.

### The Forward Pass

```rust
pub fn forward(&self, input: &[i32; 16]) -> [i32; 2] {
    let h1 = square_activate(&self.layer1.forward(input));
    let h2 = square_activate(&self.layer2.forward(&h1));
    self.layer3.forward(&h2)
}
```

Three lines. Each line does exactly one thing. No hidden control flow, no dynamic dispatch, no allocations. This is exactly what Arcis needs — a straightforward sequence of fixed-size array operations.

Note that the last layer has no activation. This is standard in classification networks — the raw logits go directly to argmax. And since `argmax(softmax(x)) == argmax(x)`, we skip softmax entirely. That saves ~50 MPC rounds of expensive exp/division operations.

### Output Constraints

The output is `[i32; 2]` = 8 bytes. Solana transaction callbacks can carry about 1,232 bytes of return data. We're using 0.6% of the available space. Even if I added confidence scores or multi-class output, there's plenty of room.

## The End-to-End Test

The most important test in the classifier module is `test_build_from_quantized_weights`. It exercises the complete deployment pipeline:

1. Start with f64 weights (simulating PyTorch export)
2. Quantize them via `quantize_linear`
3. Construct a `SentimentClassifier`
4. Run classification on a test input
5. Verify the result

```rust
let classifier = SentimentClassifier {
    layer1: quantize_linear(&w1_refs, &b1_f64),
    layer2: quantize_linear(&w2_refs, &b2_f64),
    layer3: quantize_linear(&w3_refs, &b3_f64),
};
assert_eq!(classifier.classify(&input), 1);
```

This test proves that the entire path from PyTorch weights to MPC classification works. If it passes, I know the quantization doesn't corrupt the weights in a way that changes the classification result.

## What This Module Map Means for Deployment

When I deploy to Arcium, the code maps directly:

- **`quantize.rs`** → Used once in a build script to convert PyTorch weights to `[i32; N]` constants
- **`pca.rs`** → Runs in the TypeScript/Rust client before calling the Solana program
- **`classifier.rs`** → Becomes the Arcis `#[instruction]` function, running on MXE nodes

The test suite proves each piece works in isolation AND together. When the Arcis circuit produces a result, I can compare it against the client-side reference implementation running the exact same math on plaintext. If they disagree, I know there's a bug in the Arcis translation — not in the math itself.
