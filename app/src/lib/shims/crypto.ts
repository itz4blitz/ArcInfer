/**
 * Browser polyfill for Node's crypto module.
 * @arcium-hq/client imports randomBytes, createHash, createCipheriv,
 * createDecipheriv at top level. We provide working Web Crypto implementations.
 */

// -- randomBytes --
// Must return an object with .toString('hex') like Node's Buffer
export function randomBytes(size: number): Uint8Array & { toString(encoding?: string): string } {
  const buf = new Uint8Array(size);
  globalThis.crypto.getRandomValues(buf);
  const enhanced = buf as Uint8Array & { toString(encoding?: string): string };
  enhanced.toString = (encoding?: string) => {
    if (encoding === "hex") {
      return Array.from(buf)
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
    }
    return Uint8Array.prototype.toString.call(buf);
  };
  return enhanced;
}

// -- createHash --
// Synchronous hash using @noble/hashes (already a transitive dep)
interface HashLike {
  update(data: Uint8Array | string): HashLike;
  digest(): Uint8Array & { toString(encoding?: string): string };
}

function wrapDigest(raw: Uint8Array): Uint8Array & { toString(encoding?: string): string } {
  const out = raw as Uint8Array & { toString(encoding?: string): string };
  out.toString = (encoding?: string) => {
    if (encoding === "hex") {
      return Array.from(raw)
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
    }
    return Uint8Array.prototype.toString.call(raw);
  };
  return out;
}

function textToBytes(data: Uint8Array | string): Uint8Array {
  if (typeof data === "string") return new TextEncoder().encode(data);
  return data;
}

export function createHash(algorithm: string): HashLike {
  // Collect all update() data, hash on digest()
  const chunks: Uint8Array[] = [];

  const instance: HashLike = {
    update(data: Uint8Array | string) {
      chunks.push(textToBytes(data));
      return instance;
    },
    digest() {
      // Concatenate all chunks
      const total = chunks.reduce((a, c) => a + c.length, 0);
      const merged = new Uint8Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      const algo = algorithm.toLowerCase().replace("-", "");
      if (algo === "sha256") {
        // Use @noble/hashes
        const { sha256 } = require("@noble/hashes/sha256");
        return wrapDigest(sha256(merged));
      }
      if (algo === "sha3256") {
        const { sha3_256 } = require("@noble/hashes/sha3");
        return wrapDigest(sha3_256(merged));
      }
      if (algo === "sha512") {
        const { sha512 } = require("@noble/hashes/sha512");
        return wrapDigest(sha512(merged));
      }
      throw new Error(`Unsupported hash algorithm in browser shim: ${algorithm}`);
    },
  };
  return instance;
}

// -- createCipheriv / createDecipheriv --
// AES-CTR used by the SDK's AES helper class. Provide a sync-looking wrapper
// that buffers data and encrypts/decrypts in one shot.
interface CipherLike {
  update(data: Uint8Array): Uint8Array;
  final(): Uint8Array;
}

function parseCtrAlgo(algorithm: string): number {
  // e.g. "aes-128-ctr" or "aes-256-ctr"
  const match = algorithm.match(/aes-(\d+)-ctr/i);
  if (!match) throw new Error(`Unsupported cipher in browser shim: ${algorithm}`);
  return Number(match[1]);
}

function aesCtr(key: Uint8Array, iv: Uint8Array, data: Uint8Array): Uint8Array {
  // AES-CTR can be done synchronously using a pure-JS implementation
  // For the SDK's use case (small payloads), we use a simple CTR mode impl
  // via @noble/ciphers if available, or throw
  try {
    const { ctr } = require("@noble/ciphers/aes");
    const cipher = ctr(key, iv);
    return cipher.encrypt(data);
  } catch {
    throw new Error("AES-CTR not available in browser. Install @noble/ciphers.");
  }
}

export function createCipheriv(algorithm: string, key: Uint8Array, iv: Uint8Array): CipherLike {
  parseCtrAlgo(algorithm); // validate
  const chunks: Uint8Array[] = [];
  return {
    update(data: Uint8Array) {
      chunks.push(data);
      return new Uint8Array(0);
    },
    final() {
      const total = chunks.reduce((a, c) => a + c.length, 0);
      const merged = new Uint8Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      return aesCtr(key, iv, merged);
    },
  };
}

export function createDecipheriv(algorithm: string, key: Uint8Array, iv: Uint8Array): CipherLike {
  // AES-CTR encrypt === decrypt
  return createCipheriv(algorithm, key, iv);
}

export default { randomBytes, createHash, createCipheriv, createDecipheriv };
