# The Client-Side Inference Pipeline

Everything in this doc runs on the user's machine, before encryption. It's the bridge between human-readable text and the encrypted 16-dimensional input that enters the MPC cluster.

## Architecture

```
 "I love this movie"
        │
        ▼
 ┌──────────────┐
 │  Tokenizer   │  WordPiece: text → [101, 1045, 2293, 2023, 3185, 102]
 │  (tokenizers) │  + attention mask: [1, 1, 1, 1, 1, 1]
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  ONNX Model  │  all-MiniLM-L6-v2: tokens → [1, 6, 384] hidden states
 │  (tract-onnx) │  + mean pooling → [384] sentence embedding
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │     PCA      │  384-dim → 16-dim (24x MPC cost reduction)
 │  (f64 math)  │  projection matrix + mean from Python training
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  Quantize    │  f64 → Q16.16 i32
 │  (from_f64)  │  [i32; 16] ready for MPC encryption
 └──────────────┘
```

## The Tokenizer

### What It Does

Takes a string and produces token IDs that the transformer model understands.

all-MiniLM-L6-v2 uses **WordPiece** tokenization (same as BERT):
- Common words stay intact: "love" → `[2293]`
- Uncommon words split: "unconstitutionally" → `["un", "##con", "##stitution", "##ally"]`
- Special tokens added: `[CLS]` (101) at start, `[SEP]` (102) at end

### Implementation Choice: Disable Padding

The tokenizer.json from sentence-transformers ships with padding enabled. This means every input gets padded to a fixed length with `[PAD]` tokens. I disabled this because:

1. **Wasted computation**: Padding a 5-token sentence to 128 tokens means 123 useless tokens flowing through the transformer. At ~0.7s per inference, that's pure overhead.
2. **tract handles variable shapes**: I re-optimize the tract model for each input's actual sequence length, so padding provides no benefit.
3. **Cleaner API**: Without padding, the attention mask is always all-1s, and token count accurately reflects the input text.

```rust
let mut inner = Tokenizer::from_file(path)?;
inner.with_padding(None);     // No padding
inner.with_truncation(None);  // No truncation
```

## The ONNX Model

### all-MiniLM-L6-v2

A 22M-parameter sentence transformer from Microsoft. It takes tokenized text and produces 384-dimensional embeddings where similar sentences are close in vector space.

**ONNX inputs** (discovered by inspecting the model, not from docs):
| Input | Type | Shape |
|-------|------|-------|
| `input_ids` | i64 | [1, seq_len] |
| `attention_mask` | i64 | [1, seq_len] |
| `token_type_ids` | i64 | [1, seq_len] |

`token_type_ids` is all zeros for single-sentence input. I initially tried passing only 2 inputs (as the documentation suggested), but the model requires all 3.

**Output**: `[1, seq_len, 384]` — per-token hidden states.

### Mean Pooling

The transformer gives per-token embeddings, not a sentence embedding. To get a single vector:

```
sentence_embedding = sum(hidden_states[t] for t where mask[t]==1) / count(mask==1)
```

This averages all real token embeddings, ignoring padding. It's the standard approach for sentence-transformers models.

I implemented mean pooling as a standalone function so it can be tested with synthetic data (no model needed):

```rust
pub fn mean_pool(hidden_states: &[f32], seq_len: usize, embed_dim: usize, mask: &[i64]) -> Vec<f32>
```

### tract and Dynamic Shapes

tract needs concrete tensor shapes to optimize the computation graph. Since different sentences produce different token counts, I re-optimize per call:

```rust
let model = tract_onnx::onnx()
    .model_for_read(&mut Cursor::new(&self.model_bytes))?
    .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_len as i64)))?
    .with_input_fact(1, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_len as i64)))?
    .with_input_fact(2, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_len as i64)))?
    .into_optimized()?
    .into_runnable()?;
```

This adds ~0.5s overhead per call. For a production system, I'd either:
- Pre-optimize for common sequence lengths (8, 16, 32, 64, 128) and pick the closest
- Use `tract_onnx::onnx().model_for_path()` with symbolic dimensions

For a demo, re-optimization per call is acceptable and simpler.

## The Pipeline Struct

The `InferencePipeline` is the public API. It holds the tokenizer + model and exposes three methods at different stages:

```rust
impl InferencePipeline {
    // Raw embedding — useful for generating PCA training data
    fn embed_f32(&self, text: &str) -> Vec<f32>        // 384-dim

    // After PCA — useful for debugging
    fn embed_pca(&self, text: &str, ...) -> [f64; 16]  // 16-dim

    // Final output — what gets encrypted for MPC
    fn embed_quantized(&self, text: &str, ...) -> [i32; 16]  // Q16.16
}
```

### Why f64 for PCA?

PCA is done in f64 precision even though the embedding is f32. Two reasons:

1. The PCA matrix comes from Python (numpy/sklearn), which uses f64 by default
2. We want maximum precision before quantization — quantization to Q16.16 is the lossy step, not the projection

### PCA Parameters as Arguments

The pipeline accepts the PCA projection matrix and mean vector as parameters rather than bundling them. This is deliberate — the PCA matrix comes from running sklearn on the training set in Python. Tests use an identity projection that picks the first 16 dimensions.

In the final system, these parameters would be loaded from a file exported by the Python training script.

## What the Tests Prove

### Semantic Quality

The `test_similar_sentences_closer_than_different` test verifies that the embedding model produces meaningful representations:

```rust
let emb_a = model.embed(&tok.encode("The food was delicious")).unwrap();
let emb_b = model.embed(&tok.encode("The meal was tasty")).unwrap();
let emb_c = model.embed(&tok.encode("The car broke down on the highway")).unwrap();

assert!(cosine_similarity(&emb_a, &emb_b) > cosine_similarity(&emb_a, &emb_c));
```

This isn't just a smoke test — it proves the model carries real semantic information that a downstream classifier can work with.

### Quantization Roundtrip

The `test_pipeline_full_quantized` test verifies that the f32→Q16.16 conversion doesn't destroy the signal:

```rust
let raw = pipeline.embed_f32("Hello world");
let quantized = pipeline.embed_quantized("Hello world", &mean, &projection);
for i in 0..16 {
    let recovered = to_f64(quantized[i]);
    let error = (recovered - raw[i] as f64).abs();
    assert!(error < 0.001);
}
```

If the quantization error were too large, the MPC classifier would receive garbage input. This test proves the signal survives the conversion.

## What's Still Missing

1. **Performance optimization**: Re-optimizing tract per call is slow. A production version would cache optimized models for common sequence lengths.

The PCA matrix and classifier weights are trained and exported — the pipeline loads them from `models/pca.json` and `models/classifier_weights.json`.
