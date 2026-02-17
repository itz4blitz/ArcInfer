/**
 * Browser-side sentence embedding using @huggingface/transformers.
 *
 * Dev: loads all-MiniLM-L6-v2 from local /models/ assets (no CDN dependency).
 * Prod: fetches from HuggingFace Hub CDN and caches in browser.
 * Returns a 384-dimensional sentence embedding.
 */

import type { FeatureExtractionPipeline, ProgressInfo, Tensor } from "@huggingface/transformers";

type TransformersEnvironment = (typeof import("@huggingface/transformers"))["env"];

function tensorDataToFloat32Array(data: Tensor["data"]): Float32Array {
  if (data instanceof Float32Array) return data;

  if (
    data instanceof Int8Array ||
    data instanceof Uint8Array ||
    data instanceof Uint8ClampedArray ||
    data instanceof Int16Array ||
    data instanceof Uint16Array ||
    data instanceof Int32Array ||
    data instanceof Uint32Array ||
    data instanceof Float64Array
  ) {
    return new Float32Array(data);
  }

  if (typeof BigInt64Array !== "undefined" && data instanceof BigInt64Array) {
    throw new Error("Unexpected BigInt64Array embedding output");
  }
  if (typeof BigUint64Array !== "undefined" && data instanceof BigUint64Array) {
    throw new Error("Unexpected BigUint64Array embedding output");
  }

  if (Array.isArray(data)) {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (typeof v !== "number") throw new Error("Unexpected non-number embedding output");
      out[i] = v;
    }
    return out;
  }

  throw new Error("Unexpected embedding output type");
}

let pipelineInstance: FeatureExtractionPipeline | null = null;
let initPromise: Promise<void> | null = null;

export type EmbeddingProgress = {
  status: "loading" | "ready" | "error";
  progress?: number;
  message?: string;
};

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms);
  });

  return Promise.race([promise, timeout]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

const IS_PRODUCTION = process.env.NODE_ENV === "production";

function configureTransformersEnv(env: TransformersEnvironment) {
  if (IS_PRODUCTION) {
    // Production: fetch model from HuggingFace Hub CDN, cache in browser.
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;
  } else {
    // Dev: serve from /public/models/, no CDN dependency.
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = "/models/";
    env.useBrowserCache = false;
    env.useCustomCache = false;
  }

  // Turbopack aliases `fs` to a shim that exports real functions (existsSync, etc.).
  // This makes Transformers.js detect IS_FS_AVAILABLE=true, which causes getFile()
  // to use FileResponse (fs-based) instead of fetch(). The shim's existsSync always
  // returns false, so every file silently 404s. Force the HTTP fetch path instead.
  env.useFS = false;
  env.useFSCache = false;
}

// The default ORT entry point (ort.bundle.min.mjs) is the JSEP build.
// Its embedded Emscripten factory expects the `.jsep.wasm` binary at runtime.
const ORT_WASM_BINARY = "/onnx/ort-wasm-simd-threaded.jsep.wasm";

function configureOnnxBackend(env: TransformersEnvironment) {
  const wasm = env.backends.onnx.wasm;
  if (!wasm) {
    throw new Error("ORT WASM backend not available after Transformers.js import");
  }

  // CRITICAL: Use object-form wasmPaths — specify ONLY the .wasm binary path.
  //
  // String-form (e.g., "/onnx/") tells ORT to dynamically import() the .mjs
  // module factory from that prefix.  In Turbopack's dev server this dynamic
  // import() hangs because Turbopack intercepts the module request.
  //
  // Object-form lets ORT keep using its *embedded* (bundled-in) module factory
  // — no dynamic import — while fetching only the WASM binary from our local path.
  wasm.wasmPaths = { wasm: ORT_WASM_BINARY };

  // No worker proxy — avoids message-passing hangs in dev.
  wasm.proxy = false;

  // Single-threaded: works regardless of cross-origin isolation status.
  wasm.numThreads = 1;

  // Fail fast if WASM init stalls.
  wasm.initTimeout = 15_000;
}

async function preflightLocalAssets(): Promise<void> {
  if (typeof window === "undefined" || IS_PRODUCTION) return;

  const urls = [
    "/models/Xenova/all-MiniLM-L6-v2/config.json",
    "/models/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx",
    ORT_WASM_BINARY,
  ];

  await Promise.all(
    urls.map(async (url) => {
      const res = await withTimeout(
        fetch(url, { method: "HEAD", cache: "no-store" }),
        10_000,
        `Fetch ${url}`
      );
      if (!res.ok) throw new Error(`Asset not reachable: ${url} (${res.status})`);
    })
  );
}

/**
 * Initialize the embedding model. Call once on app load.
 */
export async function initEmbedding(
  onProgress?: (p: EmbeddingProgress) => void
): Promise<void> {
  if (pipelineInstance) {
    onProgress?.({ status: "ready", progress: 100, message: "Model ready" });
    return;
  }
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      onProgress?.({ status: "loading", progress: 0, message: "Preparing model loader..." });

      if (!IS_PRODUCTION) {
        onProgress?.({ status: "loading", progress: 2, message: "Checking local model assets..." });
        await preflightLocalAssets();
      }

      onProgress?.({ status: "loading", progress: 5, message: "Loading Transformers.js runtime..." });
      const { pipeline, env } = await withTimeout(
        import("@huggingface/transformers"),
        20_000,
        "Transformers import"
      );

      onProgress?.({ status: "loading", progress: 8, message: "Configuring ONNX backend..." });
      configureTransformersEnv(env);
      configureOnnxBackend(env);

      onProgress?.({ status: "loading", progress: 10, message: "Initializing model pipeline..." });
      const initTask = withTimeout(
        pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
          device: "wasm",
          local_files_only: !IS_PRODUCTION,
          dtype: "q8",
          subfolder: "onnx",
          progress_callback: (info: ProgressInfo) => {
            if (info.status === "progress") {
              const pct = info.progress <= 1 ? info.progress * 100 : info.progress;
              onProgress?.({
                status: "loading",
                progress: pct,
                message: `Loading model: ${Math.round(pct)}%`,
              });
            } else if (info.status === "initiate" || info.status === "download" || info.status === "done") {
              onProgress?.({
                status: "loading",
                message: `${info.status}: ${info.file}`,
              });
            }
          },
        }),
        45_000,
        "Model initialization"
      );

      pipelineInstance = await initTask;

      onProgress?.({ status: "ready", progress: 100, message: "Model ready" });
    } catch (error) {
      onProgress?.({ status: "error", message: `Failed to load model: ${error}` });
      throw error;
    } finally {
      if (!pipelineInstance) initPromise = null;
    }
  })();

  return initPromise;
}

export function resetEmbedding() {
  pipelineInstance = null;
  initPromise = null;
}

/**
 * Generate a 384-dimensional sentence embedding.
 * The model must be initialized first via initEmbedding().
 */
export async function embed(text: string): Promise<Float32Array> {
  if (!pipelineInstance) {
    throw new Error("Embedding model not initialized. Call initEmbedding() first.");
  }

  const output: Tensor = await pipelineInstance(text, {
    pooling: "mean",
    normalize: true,
  });

  return tensorDataToFloat32Array(output.data);
}
