/*
 * src/lib/inference/kvMemory.ts
 *
 * Operation: KV-cache size for a model shape and an attention variant.
 * Shapes:    kvValuesPerToken(shape, kind) → values per token (all layers)
 *            kvBytes(shape, kind, bytesPerValue, tokens) → bytes
 * Intuition: every layer keeps a key and a value vector per token per KV
 *            head: 2·layers·kv_heads·head_dim values. GQA and MQA shrink
 *            kv_heads; MLA caches one compressed latent per layer instead.
 * MDX:       /learn/02-kv-cache.
 *
 * Per-token element counts follow DeepSeek-V2 (arXiv:2405.04434) Table 1:
 * MHA 2·n_h·d_h·l, GQA 2·n_g·d_h·l, MQA 2·d_h·l, MLA (d_c + d_h^R)·l.
 */
import { LLAMA3_70B, LLAMA3_8B, headDim, type ModelSpec } from "./costModel";

export type AttentionKind = "mha" | "gqa" | "mqa" | "mla";

/** The attention shape that decides the cache size. */
export type KvShape = {
  name: string;
  layers: number;
  heads: number;
  headDim: number;
  /** KV heads under GQA (the model's own setting). */
  kvHeads: number;
  /** MLA latent sizes: d_c (KV compression) and d_h^R (decoupled RoPE key). */
  mla: { dC: number; dRope: number };
};

/** DeepSeek-V2's MLA dimensions (arXiv:2405.04434 §3.1.2). */
export const DEEPSEEK_V2_MLA = { dC: 512, dRope: 64 };

function fromSpec(m: ModelSpec): KvShape {
  return {
    name: m.name,
    layers: m.n_layers,
    heads: m.n_heads,
    headDim: headDim(m),
    kvHeads: m.n_kv_heads,
    mla: DEEPSEEK_V2_MLA,
  };
}

/**
 * Presets. The Llama shapes are the simulator's (hardware.py); DeepSeek-V2
 * is from its paper: 60 layers, 128 heads of 128 dims, MLA with d_c = 512
 * and d_h^R = 64. For a Llama shape, "MLA" applies DeepSeek-V2's latent
 * sizes hypothetically (no Llama model uses MLA).
 */
export const KV_PRESETS: Record<string, KvShape> = {
  "llama3-8b": fromSpec(LLAMA3_8B),
  "llama3-70b": fromSpec(LLAMA3_70B),
  "deepseek-v2": {
    name: "DeepSeek-V2",
    layers: 60,
    heads: 128,
    headDim: 128,
    kvHeads: 128,
    mla: DEEPSEEK_V2_MLA,
  },
};

/** KV heads each variant keeps: all heads, the GQA groups, or one. */
export function kvHeadsFor(shape: KvShape, kind: AttentionKind): number {
  if (kind === "mha") return shape.heads;
  if (kind === "gqa") return shape.kvHeads;
  return 1;
}

/** Values cached per token, summed over every layer. */
export function kvValuesPerToken(shape: KvShape, kind: AttentionKind): number {
  if (kind === "mla") return (shape.mla.dC + shape.mla.dRope) * shape.layers;
  return 2 * shape.layers * kvHeadsFor(shape, kind) * shape.headDim;
}

/** Cache bytes for `tokens` tokens (summed over the batch) at `bytesPerValue`. */
export function kvBytes(
  shape: KvShape,
  kind: AttentionKind,
  bytesPerValue: number,
  tokens: number,
): number {
  return kvValuesPerToken(shape, kind) * bytesPerValue * tokens;
}

/**
 * How many tokens of cache fit in `freeBytes` (e.g. HBM left after the
 * weights). Floor division, as the simulator's admission control does.
 */
export function tokensThatFit(
  shape: KvShape,
  kind: AttentionKind,
  bytesPerValue: number,
  freeBytes: number,
): number {
  if (freeBytes <= 0) return 0;
  return Math.floor(
    freeBytes / (kvValuesPerToken(shape, kind) * bytesPerValue),
  );
}
