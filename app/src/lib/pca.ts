/**
 * Client-side PCA dimensionality reduction and quantization.
 *
 * Takes a 384-dim embedding from the sentence transformer,
 * applies PCA to reduce to 16 dimensions, then quantizes to Q16.16 fixed-point.
 *
 * Both the PCA matrix and mean vector are PUBLIC model parameters.
 * Only the input embedding is private.
 */

import { SCALE, FEATURE_DIM } from "./constants";

export interface PcaParams {
  input_dim: number;   // 384
  output_dim: number;  // 16
  mean: number[];      // 384-dim training set mean
  components: number[][]; // 16 rows x 384 cols (principal components)
}

let pcaParams: PcaParams | null = null;

/** Load PCA parameters from the public model directory. */
export async function loadPcaParams(): Promise<PcaParams> {
  if (pcaParams) return pcaParams;

  const response = await fetch("/model/pca.json");
  if (!response.ok) throw new Error("Failed to load PCA parameters");

  pcaParams = await response.json();
  return pcaParams!;
}

/** Subtract training set mean from embedding (centering). */
function subtractMean(embedding: Float32Array, mean: number[]): number[] {
  const centered = Array.from({ length: embedding.length }, (_, i) =>
    embedding[i] - mean[i]
  );
  return centered;
}

/** Project centered embedding through PCA matrix (384 -> 16). */
function project(centered: number[], components: number[][]): number[] {
  const output = Array.from({ length: components.length }, (_, i) => {
    let dot = 0;
    for (let j = 0; j < centered.length; j++) {
      dot += components[i][j] * centered[j];
    }
    return dot;
  });
  return output;
}

/** Quantize a float to Q16.16 fixed-point integer. */
function toQ16(value: number): number {
  return Math.round(value * SCALE);
}

/**
 * Full PCA pipeline: center -> project -> quantize.
 * Takes a 384-dim f32 embedding, returns 16-dim Q16.16 i32 array.
 */
export function pcaTransform(
  embedding: Float32Array,
  params: PcaParams
): Int32Array {
  const centered = subtractMean(embedding, params.mean);
  const projected = project(centered, params.components);

  const quantized = new Int32Array(FEATURE_DIM);
  for (let i = 0; i < FEATURE_DIM; i++) {
    quantized[i] = toQ16(projected[i]);
  }
  return quantized;
}
