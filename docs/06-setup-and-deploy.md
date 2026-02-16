# Setup, Deploy, and Operate

## Quick Start (Rust-only, no Arcium needed)

These work with just Rust 1.89.0:

```bash
# Run all 96 unit tests (exclude Arcium crates that need arcium build)
cargo test --workspace --exclude arcinfer --exclude encrypted-ixs -- --test-threads=1

# Run coverage report
cargo llvm-cov --workspace --exclude arcinfer --exclude encrypted-ixs --summary-only -- --test-threads=1

# Regenerate circuit weights (if you retrain)
python3 scripts/generate_circuit_weights.py
```

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust | 1.89.0 | `rustup install 1.89.0` (auto via rust-toolchain.toml) |
| Arcium CLI | 0.8.3 | `curl --proto '=https' --tlsv1.2 -sSfL https://install.arcium.com/ \| bash && arcup install` |
| Docker | latest | docker.com (required for local MPC nodes) |
| Node.js | 18+ | nodejs.org |
| Yarn | 1.x | `npm i -g yarn` |

Arcium CLI installs: Solana CLI 2.3.0, Anchor CLI 0.32.1, arcis compiler.

### PATH setup

`arcium test` needs these in PATH:

```bash
export PATH="$HOME/.local/share/solana/install/active_release/bin:$HOME/.cargo/bin:$PATH"
```

### Solana keypair

```bash
solana-keygen new   # if you don't have one
```

### Node dependencies

```bash
yarn install
```

---

## Localnet Dev Loop

### 1. Kill zombie validators

```bash
pkill -f solana-test-validator || true
```

`solana-test-validator` persists from previous runs on port 8899.
Always kill before starting a new test run.

### 2. Build

```bash
arcium build
# Produces: build/classify.arcis, build/classify_reveal.arcis, target/deploy/arcinfer.so
```

### 3. Run tests (full cycle)

```bash
arcium test
```

This starts a local validator, boots 2 MPC nodes (Docker), deploys the program,
initializes comp defs, uploads circuits, runs all 4 integration tests, and tears down.

Expected output: **4 passing**, ~20 seconds total.

### 4. Run tests with persistent validator (faster iteration)

```bash
arcium test --detach
# Validator + MPC nodes stay running after tests complete.
# Re-run faster:
arcium test --detach --skip-local-circuit
```

### 5. Run the frontend against localnet

```bash
cd app && yarn dev
# Open http://localhost:3000
# Select "Localnet" from network dropdown
# Burner wallet auto-connects and auto-airdrops SOL
```

### 6. Verify end-to-end

1. Type "This movie was great" and click Classify
2. Watch the pipeline: Embedding -> PCA -> Encrypting -> Submitting -> MPC -> Done
3. Expect result within ~15 seconds: "POSITIVE" or "NEGATIVE"
4. During MPC stage, sub-status shows: "Queued..." -> "Executing..." -> "Finalized..."

### 7. Verify resume (optional)

1. During MPC stage, refresh the browser
2. "Pending MPC result found" card appears
3. Click "Resume" -- picks up where it left off without resubmitting

### 8. Cleanup

```bash
pkill -f solana-test-validator
```

### Localnet failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Port 8899 already in use" | Zombie validator | `pkill -f solana-test-validator` |
| "Cannot connect to Docker daemon" | Docker not running | Start Docker Desktop |
| Test hangs at "waiting for MXE key" | Nodes not ready | Wait 60s; MXE key gen is async |
| `cargo clean` then tests fail | New program keypair generated | Rebuild with `arcium build` |

---

## Devnet Deployment

### 1. Ensure wallet is funded

```bash
solana config set --url devnet
solana balance
# Need >= 2 SOL. Faucet: solana airdrop 2 (rate-limited to 2 requests/8h)
```

### 2. Set circuit URLs (required for devnet build)

The program uses offchain circuit storage (IPFS via Pinata) to avoid paying
~23 SOL each in on-chain rent for 3.3MB circuits.

```bash
export CLASSIFY_CIRCUIT_URL="https://salmon-worthy-whale-287.mypinata.cloud/ipfs/bafybeigjlp3cywmyd6xufivziaeedzfvlig6u2z6m4afsslaphmbblenzi"
export CLASSIFY_REVEAL_CIRCUIT_URL="https://salmon-worthy-whale-287.mypinata.cloud/ipfs/bafybeidgrpe3wvpr3b3p46ipc4nmsjguwq3afe47o5nxqvvcxwqe23bwuu"
```

Without these env vars, the program falls back to `CircuitSource::OnChain` which
requires uploading the full circuit data on-chain (not viable for large circuits).

### 3. Build with offchain circuit sources

```bash
arcium build
# Program binary contains hardcoded IPFS URLs + circuit hashes from circuit_hash!() macro
```

### 4. Deploy and test

```bash
arcium test --cluster devnet
# Expected: 4 passing
# classify: ~13s, classify_reveal: ~5s
```

### 5. Run the frontend against devnet

```bash
cd app && yarn dev
# Select "Devnet" in the network dropdown
# Connect Phantom wallet (must have devnet SOL)
# Cluster offset: 456 (devnet v0.8.3 default)
```

### Devnet failure modes

#### MPC nodes not finalizing (devnet outage)

**Symptoms:** Tests hang after submitting computation. Frontend shows
"Queued in Arcium mempool..." for >60s and eventually times out.

**Root cause:** Arcium devnet MPC nodes occasionally stop processing computations.
This is a known issue confirmed by the Arcium team on Discord.

**What to do:**
1. Check Arcium Discord `#devnet-status` for announcements
2. If tests passed earlier the same day but now hang -> node outage, not your bug
3. The frontend handles this gracefully: times out after 90s, saves the pending
   computation to localStorage, and shows "Resume" when you return later
4. Try again in 30-60 minutes
5. "Resume" button will poll the result PDA when nodes come back

#### "Program account does not exist"

```bash
# Program was never deployed to devnet, or keypair changed.
solana program show 2UEesrBiknFE3BoAh5BtZwbr5y2AFvWe2wksVi3MqeX9 --url devnet
```

#### "Insufficient funds"

```bash
solana airdrop 2 --url devnet
# If rate-limited, wait 8 hours or use a different wallet.
```

#### "Custom program error: 0xbc2" (signer privilege escalated)

The signer PDA bump was not set before `queue_computation`. Check that
`ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;` is called
before `queue_computation` in every classify/classify_reveal instruction.

---

## Architecture Reference

```
Input text
    |
    v
+---------------------------------------------+
| CLIENT (Browser)                             |
|                                              |
|  text -> tokenize -> ONNX embed (384-dim)    |
|              |                               |
|              v                               |
|  PCA reduce (384->16) -> Q16.16 quantize     |
|              |                               |
|              v                               |
|  x25519 encrypt -> 16 x [u8; 32] ciphertext |
+----------------------+-----------------------+
                       | Solana tx (classify_reveal)
                       v
+---------------------------------------------+
| SOLANA PROGRAM (Anchor)                      |
|                                              |
|  Init result PDA (is_set=false)              |
|  Queue MPC via ArgBuilder + CPI              |
|  Callback sets result PDA (is_set=true)      |
+----------------------+-----------------------+
                       | MPC
                       v
+---------------------------------------------+
| ARCIUM MPC CLUSTER                           |
|                                              |
|  Linear(16->16) + x^2 -> Linear(16->8) + x^2|
|  -> Linear(8->2) -> argmax -> reveal         |
|  426 params, ~10 MPC rounds                  |
+----------------------+-----------------------+
                       | Callback tx
                       v
+---------------------------------------------+
| CLIENT                                       |
|                                              |
|  Poll result PDA (getAccountInfo)            |
|  Display: "positive" or "negative"           |
+---------------------------------------------+
```

## Critical Implementation Notes

### Result PDA as source of truth

The frontend polls the `ClassificationResult` PDA (`getAccountInfo`) every second.
This is deterministic and works on both localnet and devnet, unlike:

- **Transaction history** (`getTransaction` returns null on localnet with tx history disabled)
- **PubSub/WebSocket** (flaky on localnet, unreliable on devnet public RPC)
- **Anchor events** (require tx history to parse)

### Callback account ordering (CRITICAL)

The `ClassifyRevealCallback` account struct MUST list custom accounts
(`result_account`) AFTER the 6 standard Arcium callback accounts:

1. `arcium_program`
2. `comp_def_account`
3. `mxe_account`
4. `computation_account`
5. `cluster_account`
6. `instructions_sysvar`
7. **`result_account`** (custom, writable)

If `result_account` appears before `instructions_sysvar`, Arcium will not include
it in the callback transaction. The callback will silently fail to write the
result PDA, and `callbackTransactionsSubmittedBm` will stay 0.

The integration test "classifies and reveals sentiment via MPC" in `tests/arcinfer.ts`
is the regression test for this ordering. If it breaks, the test times out with a
descriptive error.

### Signer PDA bump (CRITICAL)

Always set `ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;`
BEFORE calling `queue_computation`. Without this, the bump defaults to 0 and
the CPI fails with "signer privilege escalated" (error 0xbc2).

### Vendored anchor-attribute-program

The repo vendors `anchor-attribute-program@0.32.1` under `vendor/` with a patch
that conditionally compiles the utils module only for non-Solana targets:
`#[cfg(not(target_os = "solana"))]`. This fixes a BPF stack overflow in the
auto-generated `Account::try_from` method. The `[patch.crates-io]` entry in
the root `Cargo.toml` applies this patch.
