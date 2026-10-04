/*
 * src/lib/inference/toyModel.ts
 *
 * The tiny model the live demos run in the browser: the vendored
 * transformer with random weights (seed 42), character-level, 64-token
 * vocabulary. It is *untrained*: its text is gibberish. What matters is that
 * the computation is real, so the KV cache can be checked against
 * recomputation.
 * MDX:       /learn/01-generation-loop, /learn/02-kv-cache.
 */
import { initModelWeights } from "@/lib/transformer/init";
import type { ModelConfig, ModelWeights } from "@/lib/transformer/model";
import { ALPHABET, encode } from "@/lib/transformer/tokenizer";

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

let cached: ModelWeights | null = null;

/** The toy model's weights (built once, deterministically). */
export function toyWeights(): ModelWeights {
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
