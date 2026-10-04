/*
 * src/lib/transformer/embeddings.ts
 *
 * Operation: token embedding lookup + sinusoidal positional encoding.
 * Shapes:    tokenEmbedding(ids: [S], W: [V, D])   → [S, D]
 *            positionalEncoding(S, D)              → [S, D]
 * Intuition: token embedding is just a row gather from the embedding matrix
 *            (one row per id). Positional encoding is the Vaswani 2017
 *            sinusoidal scheme; PE_pos,2i = sin(pos / 10000^(2i/D)),
 *            PE_pos,2i+1 = cos(pos / 10000^(2i/D)).
 * MDX:       /learn/decoder/02-embeddings.
 */
import type { Matrix } from "./types";

/**
 * Look up a token embedding for each id in `ids`.
 *
 * @param ids        Array of token ids of length `S`.
 * @param embedding  Embedding matrix of shape `[V, D]`.
 * @returns          Matrix of shape `[S, D]`.
 *
 * Unknown ids (`< 0` or `>= V`) emit a zero row and a console warning so the
 * model doesn't silently learn to depend on garbage data.
 */
export function tokenEmbedding(ids: number[], embedding: Matrix): Matrix {
  const V = embedding.length;
  const D = V === 0 ? 0 : (embedding[0]?.length ?? 0);
  const out: Matrix = new Array<number[]>(ids.length);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i] ?? 0;
    if (id < 0 || id >= V) {
      out[i] = new Array<number>(D).fill(0);
      continue;
    }
    // Copy so callers can mutate without affecting the embedding table.
    out[i] = (embedding[id] ?? []).slice();
  }
  return out;
}

/**
 * Build the sinusoidal positional-encoding matrix.
 *
 * @param seqLen  Sequence length `S`.
 * @param dModel  Model dimension `D` (must be ≥ 2; even or odd both fine).
 * @returns       Matrix of shape `[S, D]`.
 *
 * Formula (Vaswani 2017 §3.5):
 *   PE[pos, 2i]   = sin(pos / 10000^(2i / D))
 *   PE[pos, 2i+1] = cos(pos / 10000^(2i / D))
 */
export function positionalEncoding(seqLen: number, dModel: number): Matrix {
  // Precompute the inverse of the geometric frequency div term:
  //   div[i] = exp(2i · (-log(10000) / D)) = 1 / 10000^(2i / D)
  // for i = 0, 1, ..., floor(D/2)-1
  const halfD = Math.floor(dModel / 2);
  const factor = -Math.log(10000) / dModel;
  const div = new Array<number>(halfD);
  for (let i = 0; i < halfD; i++) div[i] = Math.exp(2 * i * factor);

  const out: Matrix = new Array<number[]>(seqLen);
  for (let pos = 0; pos < seqLen; pos++) {
    const row = new Array<number>(dModel).fill(0);
    for (let i = 0; i < halfD; i++) {
      const angle = pos * (div[i] ?? 0);
      row[2 * i] = Math.sin(angle);
      const next = 2 * i + 1;
      if (next < dModel) row[next] = Math.cos(angle);
    }
    out[pos] = row;
  }
  return out;
}
