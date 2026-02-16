# Fixed-Point Arithmetic: The Foundation of MPC Inference

## The Problem

I need to run a neural network inside Arcium's MPC cluster. There's a catch: MPC protocols operate on integers in finite rings. There are no floats. Arcium's Arcis DSL does have `f64`, but under the hood it's fixed-point with 52 fractional bits over the Curve25519 base field. To truly understand what's happening — and to test my math client-side before deploying anything — I built my own fixed-point library from scratch.

## What I Read

Before writing any code, I studied:

- The Arcis DSL reference on supported types. Floats are described as "fixed-point with 52 fractional bits" with valid range `[-2^75, 2^75)`. Values outside this range get silently clamped. That "silently" part is terrifying — you wouldn't know your math was wrong until your model started producing garbage.
- Research papers on MPC-friendly neural networks: PolyMPCNet (arXiv:2209.09424), the Ditto paper on quantization-aware secure inference (arXiv:2405.05525), and the BOLT paper on privacy-preserving transformers.
- The SecureQ8 approach: weights and activations in 8-bit, but ciphertexts in a larger ring (64-bit).

I chose **Q16.16** as my fixed-point format: 16 integer bits, 16 fractional bits. This is simpler than Arcis's native 52-bit fractional format, but it maps cleanly to i32 values and gives me explicit control over precision and overflow.

## The Format: Q16.16

The idea is dead simple. Take a floating-point number, multiply by 65536 (that's 2^16), round to the nearest integer. Now you have a fixed-point representation.

```
3.14159  →  round(3.14159 × 65536)  →  205887
-0.5     →  round(-0.5 × 65536)     →  -32768
0.001    →  round(0.001 × 65536)    →  66
```

Precision is limited to 1/65536 ≈ 0.0000153. For neural network weights that typically live in the range [-2, 2], this gives us roughly 4.8 decimal digits of precision. More than enough.

The implementation is two functions:

```rust
pub fn from_f64(value: f64) -> i32 {
    (value * SCALE as f64).round() as i32
}

pub fn to_f64(fp: i32) -> f64 {
    fp as f64 / SCALE as f64
}
```

I tested the roundtrip — convert 3.14159 to fixed-point and back, the error is less than 0.0001. Convert -7.777 and back, same thing. Small fractions like 0.001 round to 66/65536 ≈ 0.001007, which is fine for our purposes.

## Operations and Their MPC Cost

This is where it gets interesting. Every arithmetic operation has a different cost profile in MPC. Understanding this drives every architectural decision.

### Addition: Free

```rust
pub fn fp_add(a: i32, b: i32) -> i32 {
    a + b
}
```

Yes, it's literally just integer addition. The scale factors are the same, so they cancel. In MPC, this is a **local operation** — each Arx node adds its own share. No communication between nodes. Zero MPC rounds. Free.

This is why bias addition in neural networks costs nothing in MPC. It's also why skip connections (like in ResNets) are essentially free.

### Multiplication: The Expensive One

```rust
pub fn fp_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> FRAC_BITS) as i32
}
```

Three things happen here:

1. **Widen to i64**: Two 32-bit numbers multiplied can produce a 64-bit result. Without widening, we'd overflow and get garbage.
2. **Multiply**: The actual multiplication.
3. **Right-shift by 16**: This is the truncation. When you multiply two Q16.16 numbers, the result has 32 fractional bits (16 + 16). We shift right by 16 to get back to Q16.16.

In MPC, this is the **expensive** operation. It requires:

1. **Beaver triple consumption**: During a preprocessing phase, the MPC cluster generates correlated random triples `(a, b, c)` where `c = a × b`. During the actual computation, each multiplication consumes one triple to mask the operands.
2. **One communication round**: The masked values are broadcast to all nodes. Each node combines its share with the broadcasted values to get its share of the product.
3. **Truncation protocol**: Another round to right-shift the shared value by 16 bits while maintaining the secret sharing.

Total: ~2 communication rounds per multiplication. When I say "minimizing multiplications is the #1 optimization for MPC performance," this is why.

### Square Activation: The ReLU Replacement

```rust
pub fn fp_square(x: i32) -> i32 {
    fp_mul(x, x)
}
```

One function call. One multiplication. Two MPC rounds.

Compare this to ReLU (`max(0, x)`), which requires a comparison. In MPC, comparisons need bit-decomposition: convert the secret-shared 32-bit integer into 32 individual secret-shared bits, run a Boolean circuit to evaluate the comparison, convert back. That's 20-40 MPC communication rounds for a single ReLU.

**Square activation costs 2 rounds. ReLU costs 20-40 rounds. That's a 10-20x speedup per activation.**

The tradeoff is that square activations behave differently from ReLU during training (more on this in the neural network doc). But the MPC performance gain is so large that it's worth the architectural change.

### Dot Product: The Accumulate-Then-Truncate Trick

```rust
pub fn fp_dot(a: &[i32], b: &[i32]) -> i32 {
    let sum: i64 = a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai as i64 * bi as i64)
        .sum();
    (sum >> FRAC_BITS) as i32
}
```

This is the core operation of every neural network layer. A single neuron computes `output = sum(input[i] * weight[i]) + bias`.

The naive approach would be: multiply each pair, truncate each product, then add them up. But I do something different — I accumulate all the products in i64 **without truncating**, then truncate once at the end.

Why?

1. **Fewer truncations = fewer MPC rounds**. If a dot product has 64 elements, the naive approach does 64 truncations (64 extra communication rounds). My approach does 1 truncation. The multiplications themselves still happen, but in MPC, all 64 multiplications in a dot product are independent — they can execute **in parallel** across the cluster. Only the final truncation is sequential.

2. **Less rounding error**. Each truncation introduces a rounding error of up to 1/65536. With 64 elements, naive truncation accumulates up to 64 × 0.0000153 ≈ 0.001 error. Single truncation: at most 0.0000153.

3. **i64 accumulation prevents overflow**. 64 products of Q16.16 values (each up to ~2 billion) could overflow i64 in theory, but for our use case with neural net weights in [-10, 10] and 64 dimensions, the intermediate sum stays well within i64 range.

## The TDD Process

I wrote every test before writing any implementation. Here's what the test suite covers and why each test matters:

### Conversion Tests
- `test_from_f64_positive`: 3.5 → 229376 (exact)
- `test_from_f64_negative`: -2.25 → -147456 (exact)
- `test_from_f64_zero`: 0.0 → 0
- `test_from_f64_small_fraction`: 0.001 → ~66 (tests rounding behavior)
- `test_to_f64_roundtrip`: 3.14159 → fixed → float, error < 0.0001
- `test_to_f64_negative_roundtrip`: -7.777 roundtrip

These prove the conversion is correct and quantify the precision loss. If I'm ever asked "how much accuracy do you lose from quantization?", I can point to these tests and say "less than 0.0001 for typical values."

### Arithmetic Tests
- `test_add_positive/negative`: Validates addition preserves signs
- `test_mul_simple/fractions/negative`: Validates multiplication across value ranges
- `test_mul_by_weight`: Simulates an actual neural net weight multiplication (0.7832 × 0.3145), verifies error < 0.001

The weight multiplication test is the most important one. If this test fails, the entire MPC inference pipeline is broken. It passes with error well under 0.001.

### Dot Product Tests
- `test_dot_product_simple`: [1,2,3] · [4,5,6] = 32 (exact)
- `test_dot_product_with_bias`: Simulates a full neuron computation with realistic fractional weights
- `test_dot_product_mismatched_lengths`: Verifies panic on dimension mismatch

### Forward Pass Test
- `test_tiny_forward_pass`: A complete 2→2→1 network with square activations, hand-computed to verify the exact output matches expectations (0.656, error < 0.01)

This last test is the most important in the entire fixed-point module. It proves that the fixed-point math can correctly execute a multi-layer neural network with square activations. When I write the Arcis circuit, it will do the exact same computation on secret-shared values.

## What I'd Do Differently

If I were building this for production rather than a demo, I'd consider:

- **Q8.24**: More fractional bits for better precision, but narrower integer range. Works if you can guarantee weights stay in [-128, 127].
- **Dynamic quantization**: Different precision for different layers based on value ranges.
- **Overflow detection**: Currently I'd silently wrap on overflow. A production system should detect and handle this.

But for a demo on Arcium's testnet, Q16.16 is the right choice. Simple, understandable, debuggable, and maps directly to Arcis types.

## Files

- `crates/arcinfer-core/src/fixed_point.rs` — All implementation and tests
- Zero external dependencies
