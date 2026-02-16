/**
 * Arcium encryption helpers for browser usage.
 *
 * Handles x25519 key exchange and RescueCipher encryption of
 * quantized embedding features before submitting to the MPC network.
 */

import { x25519 } from "@noble/curves/ed25519";
import { debugEvent } from "@/lib/debug";
import type { RescueCipher } from "@arcium-hq/client";

// Field modulus for curve25519 scalar field
// Values must be < 2^252 for RescueCipher
const CURVE25519_SCALAR_FIELD_MODULUS = BigInt(
  "7237005577332262213973186563042994240857116359379907606001950938285454250989"
);

export interface EncryptionContext {
  clientPrivateKey: Uint8Array;
  clientPublicKey: Uint8Array;
  sharedSecret: Uint8Array;
  cipher: RescueCipher;
}

/**
 * Create an encryption context by performing x25519 key exchange with the MXE.
 */
export async function createEncryptionContext(
  mxePublicKey: Uint8Array
): Promise<EncryptionContext> {
  debugEvent("crypto", "createEncryptionContext-start", {
    mxePublicKeyLen: mxePublicKey.length,
  });
  const clientPrivateKey = x25519.utils.randomPrivateKey();
  const clientPublicKey = x25519.getPublicKey(clientPrivateKey);
  const sharedSecret = x25519.getSharedSecret(clientPrivateKey, mxePublicKey);

  // Dynamic import to avoid SSR issues
  const { RescueCipher } = await import("@arcium-hq/client");
  const cipher = new RescueCipher(sharedSecret);

  debugEvent("crypto", "createEncryptionContext-ready", {
    clientPublicKeyLen: clientPublicKey.length,
    sharedSecretLen: sharedSecret.length,
  });

  return { clientPrivateKey, clientPublicKey, sharedSecret, cipher };
}

/**
 * Convert a Q16.16 i32 value to a field element for encryption.
 * Negative values are mapped into the positive field via modular arithmetic.
 */
function i32ToFieldElement(value: number): bigint {
  if (value < 0) {
    return CURVE25519_SCALAR_FIELD_MODULUS + BigInt(value);
  }
  return BigInt(value);
}

/**
 * Encrypt quantized features for MPC submission.
 * Returns the ciphertext array and nonce needed for the Solana instruction.
 */
export function encryptFeatures(
  features: Int32Array,
  cipher: RescueCipher
): { ciphertexts: number[][]; nonce: Uint8Array; nonceValue: bigint } {
  debugEvent("crypto", "encryptFeatures-start", {
    featureCount: features.length,
  });
  // Generate a 16-byte nonce
  const nonce = new Uint8Array(16);
  crypto.getRandomValues(nonce);

  // Read nonce as little-endian u128
  const view = new DataView(nonce.buffer);
  const lo = view.getBigUint64(0, true);
  const hi = view.getBigUint64(8, true);
  const nonceValue = (hi << 64n) | lo;

  // Convert features to field elements
  const plaintextElements = Array.from(features).map(i32ToFieldElement);

  // Encrypt all features at once
  const ciphertexts: number[][] = cipher.encrypt(plaintextElements, nonce);

  debugEvent("crypto", "encryptFeatures-done", {
    ciphertextCount: ciphertexts.length,
    nonceLen: nonce.length,
  });

  return { ciphertexts, nonce, nonceValue };
}
