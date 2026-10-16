/*
 * src/lib/inference/toyModel.ts
 *
 * The tiny model the live demos run in the browser: the vendored
 * transformer (d_model 16, 2 heads, 2 blocks), character-level, 64-token
 * vocabulary. Since brief 41 its default weights are the trained ones the
 * Decoder site uses: Training's default run on Alice's Adventures in
 * Wonderland (public/data/trained-weights.json, vendored byte for byte from
 * transformer-explainer's public/training/default-run-weights.json), so its
 * greedy text starts to look like words. The random weights it started from
 * (seed 42) are one click away. Either way the computation is real, so the
 * KV cache can be checked against recomputation.
 * MDX:       /learn/01-generation-loop, /learn/02-kv-cache.
 */
import { initModelWeights } from "@/lib/transformer/init";
import {
  unpackWeights,
  type ModelConfig,
  type ModelWeights,
} from "@/lib/transformer/model";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";
import type { Matrix, Vector } from "@/lib/transformer/types";

/** d_model 16, 2 heads, 2 blocks: the explainer's widget model. */
export const TOY_CONFIG: ModelConfig = {
  seq_len: 32,
  d_model: 16,
  n_heads: 2,
  d_ff: 32,
  n_blocks: 2,
  vocab_size: 64,
  seed: 42,
};

/** Longest sequence the demos generate to. */
export const TOY_MAX_LEN = 32;

/** The trained model's context: it saw 8-character windows. */
export const TRAINED_CONTEXT = 8;

/** Which weights a demo runs. */
export type ToyKind = "trained" | "random";

let cached: ModelWeights | null = null;
let cachedTrained: ModelWeights | null = null;

/** The trained weights' file, served from public/. */
export const TRAINED_WEIGHTS_URL = "/data/trained-weights.json";
export type TrainedWeightsFile = {
  params: Record<string, Matrix | Vector>;
  steps: number;
};
let trainedFile: TrainedWeightsFile | null = null;

/**
 * Hand the trained weights to the library. They are fetched in the browser
 * rather than bundled (110 kB of doubles); server code and tests import
 * them through `trainedData.ts`.
 */
export function provideTrainedWeights(file: TrainedWeightsFile): void {
  if (!trainedFile) trainedFile = file;
}

/** Whether the trained weights have arrived. */
export function hasTrainedWeights(): boolean {
  return trainedFile !== null;
}

/** The toy model's weights (built once, deterministically). */
export function toyWeights(kind: ToyKind = "random"): ModelWeights {
  if (kind === "trained") {
    if (!trainedFile)
      throw new Error("toyModel: the trained weights are not loaded yet");
    cachedTrained ??= unpackWeights(trainedFile.params, TOY_CONFIG);
    return cachedTrained;
  }
  cached ??= initModelWeights(TOY_CONFIG);
  return cached;
}

/** Encode text exactly (no padding); unknown characters map to id 0 ('a'). */
export function toyEncode(text: string): number[] {
  const t = text.toLowerCase().slice(0, TOY_MAX_LEN);
  return encode(t, t.length);
}

/** One token id as a printable label. */
export function toyLabel(id: number): string {
  const ch = ALPHABET[id] ?? "?";
  if (ch === " ") return "␣";
  if (ch === "\n") return "↵";
  return ch;
}
