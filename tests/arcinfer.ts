/**
 * ArcInfer Integration Tests
 *
 * Tests the full Arcium integration:
 * 1. Initialize computation definitions
 * 2. Encrypt a pre-computed embedding
 * 3. Submit for MPC classification
 * 4. Await and verify the result
 *
 * Run with: arcium test
 *
 * Prerequisites:
 * - Local Arcium cluster running (started by `arcium test`)
 * - Solana program deployed
 * - Encrypted instructions compiled
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { expect } from "chai";
import {
  getArciumEnv,
  getArciumProgram,
  getMXEAccAddress,
  getCompDefAccAddress,
  getCompDefAccOffset,
  getMempoolAccAddress,
  getExecutingPoolAccAddress,
  getComputationAccAddress,
  getClusterAccAddress,
  getLookupTableAddress,
  getMXEPublicKey,
  awaitComputationFinalization,
  uploadCircuit,
  RescueCipher,
  CURVE25519_SCALAR_FIELD_MODULUS,
} from "@arcium-hq/client";
import * as fs from "fs";
import { x25519 } from "@noble/curves/ed25519";
import { Arcinfer } from "../target/types/arcinfer";

// Feature dimensions (must match circuit)
const FEATURE_DIM = 16;

// Q16.16 scale
const SCALE = 65536;

// Quantize to Q16.16
function toQ16(value: number): number {
  return Math.round(value * SCALE);
}

// Convert the comp def offset from Uint8Array to number
// comp_def_offset("classify") in Rust produces a u32
function compDefOffsetToNumber(offsetBytes: Uint8Array): number {
  const view = new DataView(
    offsetBytes.buffer,
    offsetBytes.byteOffset,
    offsetBytes.byteLength
  );
  return view.getUint32(0, true); // little-endian
}

// Convert an i32 to a field element (bigint) for encryption
function i32ToFieldElement(value: number): bigint {
  // Sign-extend i32 to work in the field
  if (value < 0) {
    // Negative values: add the field modulus
    return CURVE25519_SCALAR_FIELD_MODULUS + BigInt(value);
  }
  return BigInt(value);
}

function getClassificationResultPda(
  programId: anchor.web3.PublicKey,
  computationAccount: anchor.web3.PublicKey
): anchor.web3.PublicKey {
  const seed = new TextEncoder().encode("classify_result");
  const [pda] = anchor.web3.PublicKey.findProgramAddressSync(
    [seed, computationAccount.toBuffer()],
    programId
  );
  return pda;
}

describe("arcinfer", () => {
  // Anchor's default confirm timeout can be too low when the localnet is booting.
  // Use a Connection with a longer confirmation timeout.
  const baseProvider = anchor.AnchorProvider.env();
  const connection = new anchor.web3.Connection(baseProvider.connection.rpcEndpoint, {
    commitment: "confirmed",
    confirmTransactionInitialTimeout: 120_000,
  });
  const provider = new anchor.AnchorProvider(connection, baseProvider.wallet, {
    commitment: "confirmed",
  });
  anchor.setProvider(provider);

  const program = anchor.workspace.Arcinfer as Program<Arcinfer>;
  const wallet = provider.wallet as anchor.Wallet;

  // Arcium environment (populated by arcium test)
  let clusterOffset: number;
  let mxeProgramId: anchor.web3.PublicKey;
  let mxeAccountAddress: anchor.web3.PublicKey;

  // x25519 encryption keys
  const clientPrivateKey = x25519.utils.randomPrivateKey();
  const clientPublicKey = x25519.getPublicKey(clientPrivateKey);

  before(async () => {
    // Get Arcium environment from test context
    const arciumEnv = getArciumEnv();
    clusterOffset = arciumEnv.arciumClusterOffset;
    // mxeProgramId is our program's ID (not the Arcium program ID).
    // The MXE account PDA is derived per-program.
    mxeProgramId = program.programId;
    mxeAccountAddress = getMXEAccAddress(mxeProgramId);
  });

  // =========================================================================
  // Initialization
  // =========================================================================

  it("initializes classify computation definition", async () => {
    const compDefOffsetBytes = getCompDefAccOffset("classify");
    const compDefOffset = compDefOffsetToNumber(compDefOffsetBytes);
    const compDefAddress = getCompDefAccAddress(mxeProgramId, compDefOffset);

    // Check if already initialized (arcium test pre-creates genesis accounts)
    const compDefInfo = await provider.connection.getAccountInfo(compDefAddress);
    if (compDefInfo) {
      console.log("  classify comp def already initialized (genesis)");
    } else {
      // Read MXE account to get LUT offset
      const arciumProgram = getArciumProgram(provider);
      const mxeAccount = await arciumProgram.account.mxeAccount.fetch(
        mxeAccountAddress
      );
      const lutAddress = getLookupTableAddress(
        mxeProgramId,
        mxeAccount.lutOffsetSlot
      );

      const tx = await program.methods
        .initClassifyCompDef()
        .accounts({
          payer: wallet.publicKey,
          mxeAccount: mxeAccountAddress,
          compDefAccount: compDefAddress,
          addressLookupTable: lutAddress,
        })
        .rpc();

      console.log("  init_classify_comp_def tx:", tx);
    }

    // Upload circuit bytecode (no-op for offchain circuits, uploads for onchain)
    const rawCircuit = fs.readFileSync("build/classify.arcis");
    await uploadCircuit(
      provider as anchor.AnchorProvider,
      "classify",
      mxeProgramId,
      rawCircuit,
      true
    );
    console.log("  classify comp def ready");
  });

  it("initializes classify_reveal computation definition", async () => {
    const compDefOffsetBytes = getCompDefAccOffset("classify_reveal");
    const compDefOffset = compDefOffsetToNumber(compDefOffsetBytes);
    const compDefAddress = getCompDefAccAddress(mxeProgramId, compDefOffset);

    // Check if already initialized (arcium test pre-creates genesis accounts)
    const compDefInfo = await provider.connection.getAccountInfo(compDefAddress);
    if (compDefInfo) {
      console.log("  classify_reveal comp def already initialized (genesis)");
    } else {
      const arciumProgram = getArciumProgram(provider);
      const mxeAccount = await arciumProgram.account.mxeAccount.fetch(
        mxeAccountAddress
      );
      const lutAddress = getLookupTableAddress(
        mxeProgramId,
        mxeAccount.lutOffsetSlot
      );

      const tx = await program.methods
        .initClassifyRevealCompDef()
        .accounts({
          payer: wallet.publicKey,
          mxeAccount: mxeAccountAddress,
          compDefAccount: compDefAddress,
          addressLookupTable: lutAddress,
        })
        .rpc();

      console.log("  init_classify_reveal_comp_def tx:", tx);
    }

    // Upload circuit bytecode (no-op for offchain circuits, uploads for onchain)
    const rawCircuit = fs.readFileSync("build/classify_reveal.arcis");
    await uploadCircuit(
      provider as anchor.AnchorProvider,
      "classify_reveal",
      mxeProgramId,
      rawCircuit,
      true
    );
    console.log("  classify_reveal circuit uploaded and finalized");
  });

  // =========================================================================
  // Classification Tests
  // =========================================================================

  it("classifies a positive sentence via MPC", async () => {
    // Synthetic 16-dim Q16.16 input for structure testing.
    // In production, features come from the embed → PCA → quantize pipeline.
    const syntheticFeatures = new Array(FEATURE_DIM)
      .fill(0)
      .map(() => toQ16((Math.random() - 0.5) * 2));

    // Wait for MXE public key (key generation runs asynchronously after node startup)
    let mxePublicKeyBytes: Uint8Array | null = null;
    for (let i = 0; i < 60; i++) {
      mxePublicKeyBytes = await getMXEPublicKey(provider, mxeProgramId);
      if (mxePublicKeyBytes) break;
      console.log(`  waiting for MXE key generation... (${i + 1}s)`);
      await new Promise((r) => setTimeout(r, 1000));
    }
    expect(mxePublicKeyBytes).to.not.be.null;

    // Derive shared secret for Rescue encryption
    const sharedSecret = x25519.getSharedSecret(
      clientPrivateKey,
      mxePublicKeyBytes!
    );
    const cipher = new RescueCipher(sharedSecret);

    // 16-byte nonce for Rescue CTR mode
    const nonce = new Uint8Array(16);
    const nonceView = new DataView(nonce.buffer);
    const nonceValue = BigInt(Date.now());
    nonceView.setBigUint64(0, nonceValue, true);

    // Encrypt each feature value as a field element
    const plaintextElements = syntheticFeatures.map(i32ToFieldElement);
    const encryptedAll = cipher.encrypt(plaintextElements, nonce);

    // Each encrypted element is a number[] (32 bytes)
    const encryptedFeatures: number[][] = encryptedAll;

    // Derive all required accounts
    const computationOffset = new anchor.BN(Date.now());
    const compDefOffsetBytes = getCompDefAccOffset("classify");
    const compDefOffset = compDefOffsetToNumber(compDefOffsetBytes);

    const tx = await program.methods
      .classify(
        computationOffset,
        encryptedFeatures as any,
        Array.from(clientPublicKey) as any,
        new anchor.BN(nonceValue.toString())
      )
      .accounts({
        payer: wallet.publicKey,
        mxeAccount: mxeAccountAddress,
        mempoolAccount: getMempoolAccAddress(clusterOffset),
        executingPool: getExecutingPoolAccAddress(clusterOffset),
        computationAccount: getComputationAccAddress(
          clusterOffset,
          computationOffset
        ),
        compDefAccount: getCompDefAccAddress(mxeProgramId, compDefOffset),
        clusterAccount: getClusterAccAddress(clusterOffset),
      })
      .rpc();

    console.log("  classify tx:", tx);

    // Wait for MPC computation to complete
    const result = await awaitComputationFinalization(
      provider,
      computationOffset,
      mxeProgramId
    );

    console.log("  MPC computation finalized:", result);
    expect(result).to.not.be.null;
  });

  // REGRESSION TEST: Callback Account Ordering
  //
  // This test validates that the ClassifyRevealCallback account struct in
  // programs/arcinfer/src/lib.rs has custom accounts (result_account) AFTER
  // the 6 standard Arcium callback accounts. If result_account appears before
  // instructions_sysvar, Arcium will not include it in the callback transaction.
  //
  // Symptoms of ordering regression:
  //   - Arcium computation reaches "finalized" status
  //   - callbackTransactionsSubmittedBm stays 0
  //   - result PDA never flips to is_set=true
  //   - This test times out with CALLBACK ORDERING REGRESSION error
  it("classifies and reveals sentiment via MPC", async () => {
    const syntheticFeatures = new Array(FEATURE_DIM)
      .fill(0)
      .map(() => toQ16((Math.random() - 0.5) * 2));

    let mxePublicKeyBytes: Uint8Array | null = null;
    for (let i = 0; i < 60; i++) {
      mxePublicKeyBytes = await getMXEPublicKey(provider, mxeProgramId);
      if (mxePublicKeyBytes) break;
      console.log(`  waiting for MXE key generation... (${i + 1}s)`);
      await new Promise((r) => setTimeout(r, 1000));
    }
    expect(mxePublicKeyBytes).to.not.be.null;

    const sharedSecret = x25519.getSharedSecret(
      clientPrivateKey,
      mxePublicKeyBytes!
    );
    const cipher = new RescueCipher(sharedSecret);

    const nonce = new Uint8Array(16);
    const nonceView = new DataView(nonce.buffer);
    const nonceValue = BigInt(Date.now() + 1);
    nonceView.setBigUint64(0, nonceValue, true);

    const plaintextElements = syntheticFeatures.map(i32ToFieldElement);
    const encryptedAll = cipher.encrypt(plaintextElements, nonce);
    const encryptedFeatures: number[][] = encryptedAll;

    const computationOffset = new anchor.BN(Date.now() + 1);
    const compDefOffsetBytes = getCompDefAccOffset("classify_reveal");
    const compDefOffset = compDefOffsetToNumber(compDefOffsetBytes);

    const computationAccount = getComputationAccAddress(
      clusterOffset,
      computationOffset
    );
    const resultAccount = getClassificationResultPda(
      program.programId,
      computationAccount
    );

    const tx = await program.methods
      .classifyReveal(
        computationOffset,
        encryptedFeatures as any,
        Array.from(clientPublicKey) as any,
        new anchor.BN(nonceValue.toString())
      )
      .accounts({
        payer: wallet.publicKey,
        mxeAccount: mxeAccountAddress,
        mempoolAccount: getMempoolAccAddress(clusterOffset),
        executingPool: getExecutingPoolAccAddress(clusterOffset),
        computationAccount,
        resultAccount,
        compDefAccount: getCompDefAccAddress(mxeProgramId, compDefOffset),
        clusterAccount: getClusterAccAddress(clusterOffset),
      })
      .rpc();

    console.log("  classify_reveal tx:", tx);

    // Poll the persistent result PDA written by the callback.
    // On localnet this completes in <20s. 60s timeout is generous.
    const POLL_TIMEOUT_S = 60;
    for (let i = 0; i < POLL_TIMEOUT_S; i++) {
      const acc: any = await program.account.classificationResult.fetchNullable(
        resultAccount
      );
      if (acc?.isSet) {
        console.log("  result PDA set:", {
          class: Number(acc.class),
          computationAccount: acc.computationAccount.toBase58(),
        });
        expect(Number(acc.class)).to.satisfy((n: number) => n === 0 || n === 1);

        // Verify the computation account reference matches what we submitted
        expect(acc.computationAccount.toBase58()).to.equal(
          computationAccount.toBase58()
        );
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }

    throw new Error(
      `CALLBACK ORDERING REGRESSION: Result PDA not set after ${POLL_TIMEOUT_S}s.\n` +
      `Check ClassifyRevealCallback account struct ordering in programs/arcinfer/src/lib.rs:\n` +
      `  result_account MUST come AFTER instructions_sysvar (position 7+).\n` +
      `If ordering is correct, check Arcium node logs for callback tx errors.\n` +
      `Expected PDA: ${resultAccount.toBase58()}`
    );
  });
});
