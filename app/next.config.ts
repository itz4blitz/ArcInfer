import type { NextConfig } from "next";
import { resolve } from "path";

const nextConfig: NextConfig = {
  // Transpile Arcium client SDK for browser compatibility
  transpilePackages: ["@arcium-hq/client"],

  // Server-side: externalize native Node modules
  serverExternalPackages: ["onnxruntime-node"],

  // Turbopack (default bundler in Next.js 16)
  turbopack: {
    root: resolve(import.meta.dirname),
    resolveAlias: {
      // Stub out Node-only modules for client bundles
      // @arcium-hq/client imports these at top level
      fs: "./src/lib/shims/fs.ts",
      crypto: "./src/lib/shims/crypto.ts",
      // Stub out onnxruntime-node on client (we use onnxruntime-web)
      "onnxruntime-node": "./src/lib/shims/fs.ts",
    },
  },

  async headers() {
    // Required for SharedArrayBuffer, which onnxruntime-web's threaded WASM backend uses.
    // Without cross-origin isolation, model initialization can hang or fail in the browser.
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
      // Prevent HTTP cache confusion for large binary assets during dev.
      // (Transformers.js caching is handled separately via CacheStorage settings in code.)
      {
        source: "/models/:path*",
        headers: [{ key: "Cache-Control", value: "no-store" }],
      },
      {
        source: "/onnx/:path*",
        headers: [{ key: "Cache-Control", value: "no-store" }],
      },
    ];
  },
};

export default nextConfig;
