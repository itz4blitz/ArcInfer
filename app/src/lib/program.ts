/**
 * Anchor program interaction for browser clients.
 *
 * Handles account derivation, transaction construction, and
 * computation finalization polling for the ArcInfer program.
 */

import * as anchor from "@coral-xyz/anchor";
import type { Wallet as AnchorWallet } from "@coral-xyz/anchor/dist/esm/provider";
import { Connection, PublicKey } from "@solana/web3.js";
import idl from "./idl.json";
import { debugEvent } from "@/lib/debug";

type ClassifyRevealArgs = [
  anchor.BN,
  number[][],
  number[],
  anchor.BN
];

type ProgramMethods = {
  classifyReveal: (
    ...args: ClassifyRevealArgs
  ) => {
    accounts: (a: Record<string, unknown>) => { rpc: () => Promise<string> };
  };
};

export type ProgramConfig = {
  programId: PublicKey;
  clusterOffset: number;
};

/**
 * Arcium computation status surfaced during polling.
 * Derived from @arcium-hq/reader when available; falls back to "polling".
 */
export type MpcStatus =
  | "polling"          // Polling for callback event, no reader info yet
  | "queued"           // Arcium: computation is in mempool
  | "executing"        // Arcium: cluster is running MPC
  | "finalized"        // Arcium: finalized, waiting for callback tx
  | "failed"           // Arcium: computation failed
  | "unknown";         // Reader unavailable or unrecognized status

export type AwaitResultOpts = {
  timeoutMs?: number;
  pollIntervalMs?: number;
  /** Called with the current MPC status during polling so the UI can show sub-stages. */
  onStatus?: (status: MpcStatus, pollCount: number, elapsedMs: number) => void;
};

export type ClassificationRefs = {
  tx: string;
  computationOffset: anchor.BN;
  computationAccount: PublicKey;
};

/**
 * ClassificationRevealedEvent discriminator: sha256("event:ClassificationRevealedEvent")[..8].
 * Layout: 8-byte disc + 32-byte pubkey (computation_offset) + 1-byte u8 (class).
 */
const REVEALED_EVENT_DISC = new Uint8Array([135, 32, 254, 183, 20, 228, 43, 95]);

function normalizeArciumComputationStatus(status: unknown): string {
  if (!status || typeof status !== "object") return "unknown";
  const keys = Object.keys(status as Record<string, unknown>);
  return keys[0] ?? "unknown";
}

function mapArciumStatus(raw: string): MpcStatus {
  switch (raw) {
    case "queued": return "queued";
    case "executing": return "executing";
    case "finalized": return "finalized";
    case "failed": return "failed";
    default: return "unknown";
  }
}

function makeReadOnlyAnchorWallet(): AnchorWallet {
  // A wallet implementation for read-only Anchor Program fetches.
  // These methods should never be called in our read paths.
  const publicKey = new PublicKey("11111111111111111111111111111111");
  return {
    publicKey,
    async signTransaction() {
      throw new Error("Read-only wallet: signTransaction is not available");
    },
    async signAllTransactions() {
      throw new Error("Read-only wallet: signAllTransactions is not available");
    },
  };
}

/**
 * Create an Anchor program instance from a wallet adapter.
 */
export function getProgram(
  connection: Connection,
  wallet: AnchorWallet
): anchor.Program {
  const provider = new anchor.AnchorProvider(connection, wallet, {
    commitment: "confirmed",
    // Use "processed" for preflight so simulation sees the freshest state.
    // The default devnet RPC frequently returns stale data at "confirmed" level,
    // causing spurious AccountNotInitialized errors during preflight simulation.
    preflightCommitment: "processed",
  });
  return new anchor.Program(idl as unknown as anchor.Idl, provider);
}

/**
 * Submit encrypted features for MPC classification (revealed result).
 * Uses the classify_reveal instruction which returns the class as plaintext.
 */
export async function submitClassification(
  program: anchor.Program,
  wallet: AnchorWallet,
  encryptedFeatures: number[][],
  clientPublicKey: Uint8Array,
  nonceValue: bigint,
  config: ProgramConfig
): Promise<ClassificationRefs> {
  // Dynamic import for Arcium client SDK helpers
  const {
    getMXEAccAddress,
    getCompDefAccAddress,
    getCompDefAccOffset,
    getMempoolAccAddress,
    getExecutingPoolAccAddress,
    getComputationAccAddress,
    getClusterAccAddress,
  } = await import("@arcium-hq/client");

  const mxeAccountAddress = getMXEAccAddress(config.programId);

  const compDefOffsetBytes = getCompDefAccOffset("classify_reveal");
  const compDefOffset = new DataView(
    compDefOffsetBytes.buffer,
    compDefOffsetBytes.byteOffset,
    compDefOffsetBytes.byteLength
  ).getUint32(0, true);

  // Random 64-bit offset (docs recommend random 8 bytes)
  const rand = new Uint8Array(8);
  if (!globalThis.crypto?.getRandomValues) {
    throw new Error("WebCrypto not available (crypto.getRandomValues)");
  }
  globalThis.crypto.getRandomValues(rand);
  const hex = Array.from(rand)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  const computationOffset = new anchor.BN(hex, 16);

  const computationAccount = getComputationAccAddress(
    config.clusterOffset,
    computationOffset
  );

  debugEvent("solana", "submitClassification-derive", {
    programId: config.programId.toBase58(),
    clusterOffset: config.clusterOffset,
    computationOffset: computationOffset.toString(),
    mxeAccount: String(mxeAccountAddress),
    compDefOffset,
    computationAccount: computationAccount.toBase58(),
    featureCount: encryptedFeatures.length,
    ciphertextShape: encryptedFeatures.map((c) => c.length),
  });

  const methods = program.methods as unknown as ProgramMethods;
  const tx = await methods
    .classifyReveal(
      computationOffset,
      encryptedFeatures,
      Array.from(clientPublicKey),
      new anchor.BN(nonceValue.toString())
    )
    .accounts({
      payer: wallet.publicKey,
      mxeAccount: mxeAccountAddress,
      mempoolAccount: getMempoolAccAddress(config.clusterOffset),
      executingPool: getExecutingPoolAccAddress(config.clusterOffset),
      computationAccount,
      compDefAccount: getCompDefAccAddress(config.programId, compDefOffset),
      clusterAccount: getClusterAccAddress(config.clusterOffset),
    })
    .rpc();

  debugEvent("solana", "submitClassification-sent", {
    tx,
  }, "info");

  return { tx, computationOffset, computationAccount };
}

/**
 * Parse ClassificationRevealedEvent from a transaction's log messages.
 * Returns the class value if found, or undefined.
 */
function parseRevealedEventFromLogs(logMessages: string[]): number | undefined {
  for (const log of logMessages) {
    if (!log.startsWith("Program data: ")) continue;
    const b64 = log.slice("Program data: ".length);
    let data: Uint8Array;
    try {
      data = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    } catch {
      continue;
    }
    // ClassificationRevealedEvent: 8-byte discriminator + 32-byte pubkey + 1-byte u8
    if (data.length < 41) continue;
    let match = true;
    for (let i = 0; i < 8; i++) {
      if (data[i] !== REVEALED_EVENT_DISC[i]) {
        match = false;
        break;
      }
    }
    if (match) return data[40];
  }
  return undefined;
}

/**
 * Wait for MPC computation finalization and extract the classification result.
 *
 * Polls the computation account for new transactions. When the Arcium callback
 * executes, it emits a ClassificationRevealedEvent which we parse from the logs.
 */
export async function awaitResult(
  connection: Connection,
  computationOffset: anchor.BN,
  config: ProgramConfig,
  opts?: AwaitResultOpts
): Promise<{ classValue: number; callbackTx: string }> {
  const { getComputationAccAddress, getArciumProgram } = await import(
    "@arcium-hq/client"
  );

  const timeoutMs = opts?.timeoutMs ?? 90_000;
  const pollIntervalMs = opts?.pollIntervalMs ?? 2_000;
  const onStatus = opts?.onStatus;

  const computationAccount = getComputationAccAddress(
    config.clusterOffset,
    computationOffset
  );

  debugEvent("mpc", "awaitResult-start", {
    programId: config.programId.toBase58(),
    computationOffset: computationOffset.toString(),
    computationAccount: computationAccount.toBase58(),
    timeoutMs,
    pollIntervalMs,
  });

  const startedAt = Date.now();
  let pollCount = 0;
  let lastMpcStatus: MpcStatus = "polling";
  let readerAvailable = true;

  // Track signatures we've already checked so we don't re-fetch them.
  const checkedSignatures = new Set<string>();

  onStatus?.("polling", 0, 0);

  while (Date.now() - startedAt < timeoutMs) {
    pollCount += 1;
    const elapsedMs = Date.now() - startedAt;

    // Optional diagnostics via @arcium-hq/reader.
    if (readerAvailable && (pollCount === 1 || pollCount % 5 === 0)) {
      try {
        const { getComputationAccInfo } = await import("@arcium-hq/reader");
        const roProvider = new anchor.AnchorProvider(
          connection,
          makeReadOnlyAnchorWallet(),
          { commitment: "confirmed" }
        );
        const arciumProgram = getArciumProgram(roProvider);
        const comp = await getComputationAccInfo(
          arciumProgram,
          computationAccount,
          "confirmed"
        );
        const statusStr = normalizeArciumComputationStatus(
          (comp as unknown as Record<string, unknown>).status
        );
        debugEvent("mpc", "awaitResult-computationStatus",
          { pollCount, status: statusStr, computationAccount: computationAccount.toBase58() },
          "debug"
        );

        const mapped = mapArciumStatus(statusStr);
        if (mapped !== lastMpcStatus) {
          lastMpcStatus = mapped;
          onStatus?.(mapped, pollCount, elapsedMs);
        }

        if (statusStr === "failed") {
          onStatus?.("failed", pollCount, elapsedMs);
          throw new Error(
            "MPC computation failed. The Arcium cluster aborted or produced invalid output. Please try again with a new submission."
          );
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg.includes("computation failed")) throw e;
        if (msg.includes("Cannot find module") || msg.includes("Failed to resolve")) {
          readerAvailable = false;
        }
        debugEvent("mpc", "awaitResult-computationStatusUnavailable", { error: msg }, "debug");
      }
    }

    // Scan transactions on the computation account for the callback event.
    try {
      const sigs = await connection.getSignaturesForAddress(
        computationAccount,
        { limit: 10 },
        "confirmed"
      );

      for (const sigInfo of sigs) {
        if (checkedSignatures.has(sigInfo.signature)) continue;
        checkedSignatures.add(sigInfo.signature);
        if (sigInfo.err) continue;

        const tx = await connection.getTransaction(sigInfo.signature, {
          commitment: "confirmed",
          maxSupportedTransactionVersion: 0,
        });
        if (!tx?.meta?.logMessages) continue;

        const classValue = parseRevealedEventFromLogs(tx.meta.logMessages);
        if (classValue !== undefined) {
          debugEvent("mpc", "awaitResult-found", {
            pollCount,
            classValue,
            callbackTx: sigInfo.signature,
          }, "info");
          return { classValue, callbackTx: sigInfo.signature };
        }
      }
    } catch (e) {
      debugEvent("mpc", "awaitResult-sigsScanError",
        { error: e instanceof Error ? e.message : String(e) },
        "debug"
      );
    }

    if (pollCount === 1 || pollCount % 10 === 0) {
      debugEvent("mpc", "awaitResult-poll",
        { pollCount, checkedCount: checkedSignatures.size },
        "debug"
      );
    }

    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }

  const elapsed = Math.round(timeoutMs / 1000);
  throw new Error(
    `MPC computation timed out after ${elapsed}s.\n\nArcium's devnet MPC nodes may be temporarily offline — your computation is queued on-chain and will be processed when nodes recover.\n\nWhat you can do:\n  \u2022 Wait and click "Resume" later (no need to resubmit)\n  \u2022 Check Arcium Discord for devnet status updates\n\nYour submission is saved. You can close this tab and come back.`
  );
}

/**
 * Get the MXE public key for x25519 key exchange.
 */
export async function fetchMXEPublicKey(
  provider: anchor.AnchorProvider,
  programId: PublicKey
): Promise<Uint8Array> {
  const { getMXEPublicKey } = await import("@arcium-hq/client");

  // Retry for up to 30 seconds (MXE key generation is async)
  for (let i = 0; i < 30; i++) {
    const key = await getMXEPublicKey(provider, programId);
    if (key) return key;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("Failed to fetch MXE public key after 30 seconds");
}
