/*
 * src/lib/transformer/trace.ts
 *
 * Operation: `Trace` types and helpers — the visualisation hook.
 * Shapes:    every op accepts an optional `Trace`-shaped argument and
 *            records its intermediates (Q/K/V, scores, weights, etc.).
 * Intuition: the visualisations on the site are NOT a separate model. They
 *            render the actual values produced by these ops, captured here.
 *            See CLAUDE.md §6 ("Trace objects").
 * MDX:       /learn/decoder/03-attention is the headline consumer.
 */
import type { Matrix, Tensor3D, Vector } from "./types";

/**
 * Per-head attention trace. Each `Tensor3D` is `[n_heads, seq_len, seq_len]`
 * for `scores` and `weights`; `Q`, `K`, `V` are flattened across heads as
 * `[seq_len, d_model]` to keep them easy to render in the UI.
 */
export type AttentionTrace = {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  scores: Tensor3D;
  weights: Tensor3D;
};

/** Per-block trace — captures both sub-layers' intermediates. */
export type BlockTrace = {
  ln1: Matrix;
  attn: AttentionTrace;
  attnOut: Matrix; // x + Attn(LN1(x))
  ln2: Matrix;
  ffnPre: Matrix; // h @ W1 + b1
  ffnAct: Matrix; // GELU(h @ W1 + b1)
  ffnOut: Matrix; // residual after FFN
};

/** Full forward-pass trace. */
export type ForwardTrace = {
  tokenIds: number[];
  tokEmb: Matrix;
  posEmb: Matrix;
  blocks: BlockTrace[];
  xFinal: Matrix;
  logits: Matrix;
};

/**
 * Build an empty `AttentionTrace`. Useful for callers that always want a
 * trace populated; the ops mutate it in place.
 */
export function emptyAttentionTrace(): AttentionTrace {
  return {
    Q: [],
    K: [],
    V: [],
    scores: [],
    weights: [],
  };
}

/**
 * Build an empty `BlockTrace`. Block ops fill the fields as they go.
 */
export function emptyBlockTrace(): BlockTrace {
  const m: Matrix = [];
  return {
    ln1: m,
    attn: emptyAttentionTrace(),
    attnOut: m,
    ln2: m,
    ffnPre: m,
    ffnAct: m,
    ffnOut: m,
  };
}

/**
 * Sample a single value from a vector for trace previews. Used only in the
 * UI; kept here so the type-only consumers can find it next to the trace
 * type.
 */
export function head<T extends Vector | number[]>(v: T, n = 4): number[] {
  return v.slice(0, n);
}
