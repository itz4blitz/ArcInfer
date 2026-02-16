import { PublicKey } from "@solana/web3.js";

export const PROGRAM_ID = new PublicKey(
  "2UEesrBiknFE3BoAh5BtZwbr5y2AFvWe2wksVi3MqeX9"
);

// Q16.16 fixed-point scale factor
export const SCALE = 65536;

// Feature dimensions after PCA
export const FEATURE_DIM = 16;

// Backwards-compatible exports.
// The app now uses `app/src/lib/networks.ts` + `useArcInferNetwork()`.
// These remain so older imports (or cached builds) don't break.

export const RPC_ENDPOINT =
  process.env.NEXT_PUBLIC_RPC_URL || "https://api.devnet.solana.com";

export const CLUSTER_OFFSET = Number(
  process.env.NEXT_PUBLIC_CLUSTER_OFFSET || "456"
);
