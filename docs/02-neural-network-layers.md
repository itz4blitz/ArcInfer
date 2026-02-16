# Neural Network Layers: From Linear Algebra to MPC Circuits

## The Goal

With fixed-point arithmetic proven and tested, I needed the next layer up: actual neural network components. A `Linear` layer that does matrix-vector multiplication, a square activation function that works on arrays, and an argmax function for classification. Everything built with one constraint in mind: **it must be expressible in Arcis**, Arcium's MPC DSL.

## What Arcis Demands

Before writing any code, I listed the constraints from Arcium's documentation:

1. **Fixed-size arrays only**. No `Vec`, no `String`, no `HashMap`. Every array dimension must be a compile-time constant. In an MPC circuit, the computation graph is fixed — you can't have variable-length operations because every node in the cluster must execute the exact same sequence of steps.

2. **No branching on secret values**. Both branches of an `if/else` always execute when the condition depends on encrypted data. The MPC protocol evaluates both paths and selects the result based on the secret condition. This means the cost is always the sum of both branches, never one or the other.

3. **No ReLU, no max, no comparisons on secret data** (without paying a huge cost). Comparisons require bit-decomposition — converting secret-shared integers into individual secret-shared bits, running a Boolean circuit, converting back. Prohibitively expensive for an activation function applied to every neuron.

4. **Multiplication is the unit of cost**. Every `fp_mul` costs ~2 communication rounds between Arx nodes. Additions are free. The total inference time is dominated by the multiplicative depth of the circuit.

These constraints shaped every design decision.

## The Linear Layer: Const Generics for Compile-Time Safety

I used Rust's const generics to make the layer dimensions part of the type:

```rust
pub struct Linear<const IN: usize, const OUT: usize> {
    pub weights: [[i32; IN]; OUT],
    pub biases: [i32; OUT],
}
```

This means `Linear<64, 32>` is a different type from `Linear<32, 16>`. If I accidentally try to feed a 64-dimensional vector into a layer expecting 32, the compiler catches it. In the Arcis circuit, this becomes `[i32; 64]` and `[i32; 32]` — same fixed-size array pattern.

The forward pass is straightforward:

```rust
pub fn forward(&self, input: &[i32; IN]) -> [i32; OUT] {
    let mut output = [0i32; OUT];
    for j in 0..OUT {
        output[j] = fp_add(fp_dot(&self.weights[j], input.as_slice()), self.biases[j]);
    }
    output
}
```

For each output neuron: dot product of its weight vector with the input, plus bias. The `fp_dot` function handles the accumulate-then-truncate pattern I described in the fixed-point doc.

**MPC cost analysis**: All `OUT` dot products are independent — they can execute in parallel across the cluster. The multiplicative depth is just 1 (one round of multiplications, one round of truncation). So whether the layer has 2 outputs or 2000 outputs, the MPC depth is the same. Width is "free" in terms of latency; only depth matters.

### Testing the Linear Layer

I wrote tests for the simplest cases first:

**Identity passthrough**: A 2→2 layer with identity matrix weights and zero bias should return the input unchanged. This validates the basic wiring — indices aren't swapped, dot products are computed correctly.

**Bias addition**: A 2→1 layer computing `0.5*x0 + 0.5*x1 + 1.0`. Tests that biases are actually added (not multiplied, not ignored).

**Negative weights**: A layer with `-1.0` and `2.0` weights. Ensures signed arithmetic works through the entire pipeline — fixed-point encoding of negatives, signed multiplication, signed dot product accumulation.

All three passed on the first implementation. Moving on.

## Square Activation: The Test That Taught Me Something

The implementation is trivial:

```rust
pub fn square_activate<const N: usize>(input: &[i32; N]) -> [i32; N] {
    let mut output = [0i32; N];
    for i in 0..N {
        output[i] = fp_square(input[i]);
    }
    output
}
```

Element-wise `x²`. Each element costs 1 `fp_mul`. In MPC, all elements are independent and parallel — so the activation layer has depth 1 regardless of width.

The unit tests for `square_activate` itself passed fine: `[2.0, -3.0, 0.5]` → `[4.0, 9.0, 0.25]`. Nothing surprising.

### The Failure

The interesting part came when I wrote the end-to-end test. I built a tiny 2→2→2 classifier and tried to make it distinguish two classes:

```rust
// Layer 1: amplifies difference between inputs
let w1 = [
    [from_f64(1.0), from_f64(-1.0)],   // neuron 0: x0 - x1
    [from_f64(-1.0), from_f64(1.0)],    // neuron 1: x1 - x0
];
```

My idea: layer 1 computes the difference between dimensions, squaring amplifies it, layer 2 classifies based on which difference is larger.

The test failed. Here's why:

For input `[0.8, 0.2]`:
- neuron 0: `0.8 - 0.2 = 0.6`
- neuron 1: `-0.8 + 0.2 = -0.6`
- After squaring: **`[0.36, 0.36]`** — identical!

For input `[0.2, 0.8]`:
- neuron 0: `0.2 - 0.8 = -0.6`
- neuron 1: `-0.2 + 0.8 = 0.6`
- After squaring: **`[0.36, 0.36]`** — also identical!

**Square activation destroys sign information.** `(-0.6)² == (0.6)²`. Both inputs produce the same hidden representation. The classifier literally cannot distinguish them.

### The Fix and the Lesson

The fix was to redesign the network architecture. Instead of computing differences in layer 1 (which creates equal-magnitude opposite-sign values), I route each input dimension to its own neuron:

```rust
// Layer 1: pass-through
let w1 = [
    [from_f64(1.0), from_f64(0.0)],   // neuron 0 = x0
    [from_f64(0.0), from_f64(1.0)],   // neuron 1 = x1
];
```

Now after squaring:
- `[0.8, 0.2]` → `[0.64, 0.04]` (x0² is much larger)
- `[0.2, 0.8]` → `[0.04, 0.64]` (x1² is much larger)

The squared values are different! Layer 2 can now compare the squared magnitudes to classify.

**The lesson**: With square activations, useful signal must live in **magnitude differences**, not **sign differences**. You can't design the network the way you would with ReLU. The model must be trained with square activations from the start — you cannot take a ReLU-trained model and just swap the activation function.

This is something I'd absolutely bring up if asked about MPC-friendly model design in an interview. It's not in any textbook. I found it by writing a test and watching it fail.

## Argmax: Why We Skip Softmax Entirely

In standard ML inference, the last step is usually softmax followed by argmax:

```
softmax(logits) → probabilities → argmax → class label
```

Softmax computes `exp(x_i) / sum(exp(x_j))`. In MPC:
- `exp()` requires polynomial approximation (Taylor series or similar), costing many multiplications
- Division requires multiplicative inverse protocols
- Together: ~50+ MPC rounds just for the final classification step

Here's the key insight that saves all of this: **`argmax(softmax(x)) == argmax(x)`**. Softmax is a monotonically increasing function applied element-wise (after the normalization). It preserves the ordering of its inputs. The largest logit before softmax is still the largest after softmax.

So I skip softmax entirely. The Arcis circuit outputs raw logits as `Enc<Shared, [i32; 2]>`. The client decrypts them and runs argmax locally:

```rust
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
```

This runs client-side. No MPC cost. The comparison happens on decrypted plaintext. Zero communication rounds.

**What we lose**: Confidence scores. Without softmax, we don't get calibrated probabilities. We just get "class 0" or "class 1". For a sentiment analysis demo, this is perfectly fine. If a use case required confidence scores, you could approximate softmax with a low-degree polynomial at the cost of a few extra MPC rounds.

### Testing Argmax

I verified:
- Basic positive/negative cases
- **Close values**: `[1.001, 1.002]` should return index 1. In Q16.16, these differ by about 0.001 × 65536 ≈ 66. So the fixed-point representation preserves the distinction easily.

## Putting It All Together: The Full Forward Pass

The test `test_classifier_two_layers` chains everything:

```
Input [2.0, 3.0]
  → Layer 1 (2→3): [2.0, 3.0, 2.5]
  → Square: [4.0, 9.0, 6.25]
  → Layer 2 (3→2): [4.0, 9.0]
```

And `test_end_to_end_classify` proves binary classification works end-to-end with the architecture lesson baked in.

### MPC Cost of the Full Classifier

My production architecture is:
```
Linear(64→32) → Square → Linear(32→16) → Square → Linear(16→2)
```

MPC depth per component:
| Component | Depth |
|-----------|-------|
| Linear(64→32) | 1 mul + 1 trunc = 2 rounds |
| Square | 1 mul + 1 trunc = 2 rounds |
| Linear(32→16) | 1 mul + 1 trunc = 2 rounds |
| Square | 1 mul + 1 trunc = 2 rounds |
| Linear(16→2) | 1 mul + 1 trunc = 2 rounds |
| **Total** | **10 rounds** |

Compare this to running DistilBERT in MPC: 6 transformer layers × ~30 rounds per layer = ~180+ rounds. My architecture is 18x faster in terms of MPC communication.

The tradeoff: I need the client to run the embedding model locally. But that's a single forward pass of all-MiniLM-L6-v2 on a CPU — takes ~50ms. Totally acceptable.

## What's Next

With `fixed_point` and `nn` proven, the next modules are:

1. **Quantization**: Converting PyTorch f32 weights to our Q16.16 format in batch
2. **PCA**: Reducing 384-dim embeddings to 64-dim — another pure-math module
3. **The Arcis circuit**: Translating the `Linear` + `square_activate` pattern into Arcium's `#[encrypted]` DSL
4. **The Solana program**: Anchor integration with `queue_computation` and callbacks

Each will follow the same TDD pattern: tests first, implementation second.

## Files

- `crates/arcinfer-core/src/nn.rs` — All implementation and tests
- 31 total tests passing across both modules (20 fixed-point + 11 neural network)
- Zero external dependencies
