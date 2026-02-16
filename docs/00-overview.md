# Building ArcInfer: Confidential AI Inference on Arcium's MPC Network

## How This Started

I found Arcium through their CEO's interview on Tucker Carlson. ~20-person team, $14M raised, building a decentralized Multi-Party Computation network on Solana. Their "Request For Products" blog post explicitly lists **"Encrypted AI Inferencing on SLMs"** as a priority — run inference on encrypted inputs using small language models so organizations can deploy AI without ever seeing user data.

That's the product I decided to build. Not as a hypothetical proposal. As working code deployed to their testnet. 

## What I'm Building

**ArcInfer** is a confidential sentiment analysis demo. You type a sentence, it gets encrypted client-side, sent to Arcium's MPC cluster where multiple nodes collaboratively classify it without any single node ever seeing the plaintext, then the encrypted result comes back and only your browser can decrypt it.

The architecture is two-stage:

1. **Client-side**: Tokenize the text, run it through a sentence embedding model (all-MiniLM-L6-v2), reduce dimensions via PCA (384 → 16), quantize to fixed-point integers, encrypt with Arcium's RescueCipher
2. **MPC-side**: A tiny feedforward classifier (Linear → Square → Linear → Square → Linear) runs on the encrypted embedding inside Arcium's MXE. The output is two encrypted logits.

The client decrypts the logits and picks the class via argmax. Done. The MPC cluster never sees the input text or the classification result.

## Why This Specific Architecture

I spent a solid week researching before writing any code. Here's what drove the design decisions:

### Research Phase

I read through Arcium's full documentation — the Arcis DSL reference, the MPC protocol specs (Cerberus and Manticore), the computation lifecycle, every official example on their GitHub (coinflip, voting, sealed-bid auction, blackjack, medical records, rock-paper-scissors, ed25519). I studied their Anchor integration patterns — the three-instruction pattern (init_comp_def, queue_computation, callback), the ArgBuilder API, the auto-generated output types.

I also researched the competitive landscape: ZAMA's Fully Homomorphic Encryption (Concrete ML), Nillion's hybrid MPC/FHE/TEE approach, Phala Network's GPU TEE-based inference. Each makes different tradeoffs:

| Approach | Speed | Model Size | Trust Model |
|----------|-------|-----------|-------------|
| **Arcium (MPC)** | Fast for small models | Small-medium | Cryptographic (1 honest node) |
| **ZAMA (FHE)** | ~5 TPS, very slow | Small only | Cryptographic (no trust) |
| **Nillion (Hybrid)** | Varies | Large (via TEE) | Mixed |
| **Phala (TEE)** | Near-native | Large (LLMs) | Hardware trust |

Arcium's sweet spot is clear: small, MPC-friendly models where you need strong cryptographic guarantees without trusting any hardware manufacturer. That's exactly sentiment classification.

### Why Two Stages Instead of Running the Full Model in MPC

A transformer like DistilBERT has ~67M parameters and requires softmax, layer normalization, GELU activations — all operations that are brutally expensive in MPC. Softmax alone requires `exp()` and division, costing ~50+ MPC communication rounds per layer.

Instead, I split the work:
- **Client-side embedding** (the expensive part): Run the full transformer locally. This is the user's own machine — no privacy concern.
- **MPC classification** (the sensitive part): A 3-layer feedforward network with 426 parameters. This is what runs on encrypted data inside Arcium's cluster.

The client never sends raw text over the network. It sends an encrypted 16-dimensional fixed-point vector. The MPC cluster classifies it without knowing what it represents.

### Why Q16.16 Fixed-Point

MPC protocols operate on integer rings, not floating-point. Arcium's Arcis DSL does support `f64`, but it's actually fixed-point under the hood (52 fractional bits). I chose Q16.16 because:

- 16 fractional bits gives precision to ~0.000015 — more than enough for neural net weights
- 16 integer bits gives range [-32768, 32767] — neural net values typically live in [-10, 10]
- Fits in a 32-bit integer, with i64 for intermediates during multiplication
- Straightforward to reason about truncation behavior

### Why Square Activations Instead of ReLU

This was the most important architectural decision. ReLU computes `max(0, x)`, which requires a comparison on secret-shared values. In MPC, comparisons need bit-decomposition — converting a shared integer into individual shared bits, running a Boolean circuit, converting back. One ReLU costs 10-100x more than a multiplication.

Square activation (`f(x) = x²`) is just one multiplication. I train the neural network with square activations from the start, so the model learns to work with them. Research from PolyMPCNet and CrypTen shows less than 2% accuracy loss compared to ReLU.

## How I'm Building It

Test-Driven Development. I write the tests first, watch them fail, then write the minimum code to make them pass. Every module follows this cycle:

1. **RED**: Write tests that define the API and expected behavior
2. **GREEN**: Implement just enough code to pass
3. **REFACTOR**: Clean up without changing behavior

This approach means every piece of code is provably correct before I move on. When I eventually deploy to Arcium's testnet, I'll have high confidence that the math is right because I proved it incrementally.

## Project Structure

```
ArcInfer/
  crates/
    arcinfer-core/        ← Fixed-point math, neural net layers (ZERO dependencies)
    arcinfer-inference/   ← Tokenization + ONNX embedding (tract, tokenizers)
    arcinfer-pipeline/    ← Glues core + inference into the full pipeline
  encrypted-ixs/          ← Arcis MPC circuit (the on-chain classifier)
  programs/arcinfer/      ← Anchor/Solana program
  tests/                  ← TypeScript integration tests
  docs/                   ← What you're reading now
```

The key architectural choice: `arcinfer-core` has **zero external dependencies**. It's pure Rust math. Everything in it must be expressible in Arcis (Arcium's MPC DSL). This is intentional — it's the reference implementation of what runs inside the MPC circuit, and it's fully auditable without understanding any framework.

## What's in These Docs

- [01-fixed-point-arithmetic.md](./01-fixed-point-arithmetic.md) — How and why I built Q16.16 fixed-point from scratch, what each operation costs in MPC, and why I chose accumulate-then-truncate for dot products
- [02-neural-network-layers.md](./02-neural-network-layers.md) — Building Linear layers with const generics, the square activation lesson that broke my first test, and why argmax replaces softmax
- More docs will follow as I build each subsequent module
