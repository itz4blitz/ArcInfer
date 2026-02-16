# Arcium Developers Docs - RAG Notes (Local)

Purpose: condensed, query-friendly notes from `https://docs.arcium.com/developers/*` for debugging ArcInfer.

This is not a copy of the docs; it's a structured extraction with links.

## Callback Tracking

Source: `https://docs.arcium.com/developers/js-client-library/callback.md`

- Confidential computations are multi-step: queue tx -> mempool -> MPC execution -> callback tx.
- You cannot `await` the queue tx and expect results.
- Use `awaitComputationFinalization(provider, computationOffset, program.programId, commitment?)`.
- This function works by listening for a finalize event over PubSub/websocket (Anchor event listener pattern).

## JS Client Library

Source: `https://docs.arcium.com/developers/js-client-library.md`

- `@arcium-hq/client` is for encryption + submitting computations + tracking callbacks.
- `@arcium-hq/reader` is for observing the network/state.

## Encrypting Inputs (Client)

Source: `https://docs.arcium.com/developers/js-client-library/encryption.md`

- Obtain MXE x25519 public key (`getMXEPublicKey` / retry helper).
- Generate ephemeral client keypair (`x25519.utils.randomSecretKey`, `x25519.getPublicKey`).
- Compute shared secret with MXE public key.
- Initialize `RescueCipher(sharedSecret)`.
- Encrypt plaintext BigInts with 16-byte nonce; ciphertext outputs are `[u8; 32]` field elements.

## Encryption (Concept)

Source: `https://docs.arcium.com/developers/encryption.md`

- Encrypted data is represented as `Enc<Owner, T>`.
- Owners:
  - `Shared`: client + MXE can decrypt (shared secret)
  - `Mxe`: MXE-only
- Internally: x25519 key exchange + Rescue cipher in CTR mode.
- Nonce handling: MXE increments nonce by 1 for outputs; new nonce required for subsequent interactions.

## Sealing (Re-encryption)

Source: `https://docs.arcium.com/developers/encryption/sealing.md`

- Arcium can re-encrypt results to a specific recipient public key ("sealing").
- Useful to share derived results without revealing underlying data.

## Computation Lifecycle

Source: `https://docs.arcium.com/developers/computation-lifecycle.md`

- Actors: Client, MXE Program, Arcium Program, MPC Cluster.
- Flow: client encrypts -> MXE queues via CPI -> cluster processes -> Arcium verifies -> MXE callback runs.

## Program Integration (Queue + Callback)

Source: `https://docs.arcium.com/developers/program.md`

- Queue instruction builds parameters via `ArgBuilder`.
- Callback instruction must be defined with `#[arcium_callback(encrypted_ix = "...")]`.
- Callback function signature must be `ctx: Context<...>, output: SignedComputationOutputs<T>`.
- Use `output.verify_output(&cluster_account, &computation_account)`.

## Callback Accounts (Persisting Results)

Source: `https://docs.arcium.com/developers/program/callback-accs.md`

- If you want to store results, pass custom accounts into `callback_ix(..., extra_accs)`.
- Recommended approach is to use the generated `callback_ix()` helper and pass `&[CallbackAccount { pubkey, is_writable }]`.
- Accounts must exist before callback executes; cannot create accounts during callback.
- Account ordering matters: 6 standard accounts first, then custom accounts.

## Computation Definitions

Source: `https://docs.arcium.com/developers/program/computation-def-accs.md`

- Each encrypted instruction has a computation definition account.
- `comp_def_offset` computed from `sha256(name)[0..4]` as little-endian u32.
- For large circuits, use offchain circuit storage (`CircuitSource::OffChain`) with a URL and `circuit_hash!()`.

## Callback Type Generation

Source: `https://docs.arcium.com/developers/program/callback-type-generation.md`

- Output types are auto-generated from circuit return types.
- Expect `PascalCaseInstructionName + Output` and numbered fields (`field_0`, ...).
- Encryption wrappers become `SharedEncryptedStruct<N>` / `MXEEncryptedStruct<N>`.

## Arcis: Input/Output

Source: `https://docs.arcium.com/developers/arcis/input-output.md`

- Use `to_arcis()` to convert encrypted input into secret-shared values.
- Use `owner.from_arcis(...)` to output encrypted values.
- Avoid plaintext parameters for sensitive data (ARX nodes can see them).
- For large arrays, consider `Pack<T>` and generated packers.

## Arcis: Mental Model

Source: `https://docs.arcium.com/developers/arcis/mental-model.md`

- Circuits are fixed at compile time.
- If conditions on secret/shared values: both branches execute; condition selects result.
- Loops must have compile-time bounds; no `while`, `break`, `continue`.
- `reveal()` / `from_arcis()` cannot be inside non-constant branches.

## Current Limitations

Source: `https://docs.arcium.com/developers/limitations.md`

- Output size must fit into one callback transaction; practical limit ~1232 bytes.
