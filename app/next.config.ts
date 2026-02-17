import type { NextConfig } from "next";
import { resolve } from "path";

const isStaticExport = process.env.STATIC_EXPORT === "true";

const nextConfig: NextConfig = {
  // Static export for Cloudflare Pages deployment
  ...(isStaticExport ? { output: "export" } : {}),

  // Transpile Arcium client SDK for browser compatibility
  transpilePackages: ["@arcium-hq/client"],

  // Server-side: externalize native Node modules
  serverExternalPackages: ["onnxruntime-node"],

  // Turbopack (default bundler in Next.js 16, dev only)
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

  // Webpack aliases (production build uses webpack, not turbopack)
  webpack(config) {
    config.resolve.alias = {
      ...config.resolve.alias,
      fs: resolve(import.meta.dirname, "src/lib/shims/fs.ts"),
      crypto: resolve(import.meta.dirname, "src/lib/shims/crypto.ts"),
      "onnxruntime-node": resolve(import.meta.dirname, "src/lib/shims/fs.ts"),
    };
    return config;
  },

  // headers() is used by the dev server; static export uses _headers file instead.
  ...(!isStaticExport
    ? {
        async headers() {
          return [
            {
              source: "/:path*",
              headers: [
                { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
                { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
              ],
            },
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
      }
    : {}),
};

export default nextConfig;
