# ArcInfer

Private sentiment classification powered by multi-party computation on Solana.

ArcInfer classifies text as positive or negative without ever exposing the input. The entire inference pipeline — from embedding to classification — runs on encrypted data using [Arcium](https://arcium.com)'s MPC network.

## How It Works

```
Text → Tokenize → ONNX Embed (384d) → PCA (16d) → Quantize (Q16.16) → Encrypt → MPC Classify → Result
       ──────── browser ──────────────────────────────────────────────   ─ solana ─   ─ arcium ─
```

1. **Browser** — Text is tokenized and embedded into a 384-dimensional vector using all-MiniLM-L6-v2 (runs entirely in-browser via ONNX/WASM)
2. **Browser** — PCA reduces to 16 dimensions, then values are quantized to Q16.16 fixed-point
3. **Browser** — Each value is encrypted with x25519 + RescueCipher using the MPC cluster's public key
4. **Solana** — A transaction carries the 16 ciphertexts on-chain and queues MPC computation
5. **Arcium** — MPC nodes secret-share the data and evaluate a neural network collaboratively — no single node ever sees the input
6. **Solana** — The classification result (positive / negative) is emitted via a callback transaction event

## Architecture

```
crates/
  arcinfer-core/         Zero-dep Rust library: fixed-point math, NN layers, classifier
  arcinfer-inference/    Client-side pipeline: tokenizer, ONNX embedding, PCA, quantization
  arcinfer-pipeline/     Integration tests and weight loading utilities

programs/
  arcinfer/              Solana/Anchor program: on-chain orchestration and MPC callbacks

encrypted-ixs/           Arcis MPC circuits: encrypted neural network evaluation

app/                     Next.js frontend: wallet connection, encryption, progress tracking

training/                Python training pipeline (PyTorch → ONNX → PCA → Q16.16 weights)
```

### Neural Network

| Property | Value |
|----------|-------|
| Architecture | 16 → 16 → 8 → 2 (fully connected) |
| Activation | x² (MPC-friendly, no comparisons needed) |
| Parameters | 426 |
| Precision | Q16.16 fixed-point (32-bit integers, 16 fractional bits) |
| Accuracy | 80.2% on SST-2 sentiment benchmark |

## Prerequisites

- [Rust](https://rustup.rs/) (nightly, managed via `rust-toolchain.toml`)
- [Solana CLI](https://docs.solana.com/cli/install-solana-cli-tools) v1.18+
- [Anchor](https://www.anchor-lang.com/docs/installation) v0.32+
- [Arcium CLI](https://docs.arcium.com/)
- [Node.js](https://nodejs.org/) v18+
- [Yarn](https://yarnpkg.com/) v1.22+

## Build & Test

### Rust crates

```bash
cargo test --workspace --exclude arcinfer --exclude arcis-arcinfer
```

### Solana program + MPC circuits

```bash
arcium build
arcium test
```

### Frontend

```bash
cd app
yarn install
yarn dev
```

The frontend expects model files in `app/public/models/` and ONNX runtime WASM binaries in `app/public/onnx/`. These are large binaries not tracked in git — download them from the [all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) model page and the [ONNX Runtime releases](https://github.com/nicktehrany/onnxruntime-web-bundler).

## Devnet

| Resource | Value |
|----------|-------|
| Program ID | `2UEesrBiknFE3BoAh5BtZwbr5y2AFvWe2wksVi3MqeX9` |
| Network | Solana Devnet |
| Circuits | Hosted on IPFS, fetched by Arcium nodes at runtime |

## Tech Stack

- **Rust** — Core math, NN inference, fixed-point arithmetic
- **Solana / Anchor** — On-chain program, transaction orchestration
- **Arcium / Arcis** — MPC circuit definition, encrypted computation
- **TypeScript / Next.js** — Frontend with browser-side ONNX inference
- **ONNX Runtime (WASM)** — In-browser sentence embedding (all-MiniLM-L6-v2)
- **Tailwind CSS** — UI styling

## License

[MIT](LICENSE)
