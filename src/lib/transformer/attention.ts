/*
 * src/lib/transformer/attention.ts
 *
 * Operation: causal multi-head self-attention.
 * Shapes:    multiHeadAttention(x: [S, D], { W_q, W_k, W_v, W_o }: [D, D] each,
 *                              n_heads) → [S, D]
 *            singleHeadAttention(x: [S, D], W_q, W_k, W_v: [D, d_head] each)
 *                              → [S, d_head]
 * Intuition: For each query position i, scores_ij = Q_i · K_j / √d_head.
 *            Mask the future (positions j > i) to −∞ so softmax assigns it
 *            zero probability. Then weights · V gives the output.
 *
 *            Multi-head splits Q/K/V along the model dimension into
 *            `n_heads` groups of `d_model / n_heads` columns each, runs
 *            attention independently per group, concats the outputs, and
 *            projects through W_o.
 *
 * MDX:       /learn/decoder/03-attention.
 */
import { matmul, transpose } from "./matmul";
import { softmaxRows } from "./softmax";
import type { AttentionTrace } from "./trace";
import type { Matrix, Tensor3D, Vector } from "./types";

/** Weights for one attention layer. All are `[d_model, d_model]`. */
export type AttentionWeights = {
  W_q: Matrix;
  W_k: Matrix;
  W_v: Matrix;
  W_o: Matrix;
};

/**
 * Build the additive causal mask: a `[seqLen, seqLen]` matrix of 0 on and
 * below the main diagonal, and `-Infinity` strictly above it.
 *
 * Adding this to a score row before softmax zeroes out future positions.
 */
export function causalMask(seqLen: number): Matrix {
  const out: Matrix = new Array<Vector>(seqLen);
  for (let i = 0; i < seqLen; i++) {
    const row = new Array<number>(seqLen);
    for (let j = 0; j < seqLen; j++) row[j] = j > i ? -Infinity : 0;
    out[i] = row;
  }
  return out;
}

/**
 * Slice a `[S, d_model]` matrix into a single head's `[S, d_head]` view.
 * `head` is in `[0, n_heads)`. d_head = d_model / n_heads.
 */
function sliceHead(m: Matrix, head: number, d_head: number): Matrix {
  const out: Matrix = new Array<Vector>(m.length);
  const start = head * d_head;
  for (let i = 0; i < m.length; i++) {
    out[i] = m[i]!.slice(start, start + d_head);
  }
  return out;
}

/**
 * Concatenate per-head outputs along the model dimension.
 * `heads[h]` has shape `[S, d_head]`; result has shape `[S, d_model]`.
 */
function concatHeads(heads: Matrix[]): Matrix {
  if (heads.length === 0) return [];
  const seqLen = heads[0]!.length;
  const out: Matrix = new Array<Vector>(seqLen);
  for (let i = 0; i < seqLen; i++) {
    const row: Vector = [];
    for (const h of heads) row.push(...h[i]!);
    out[i] = row;
  }
  return out;
}

/**
 * Run causal self-attention for one head.
 *
 * @returns  `{ scores, weights, output }` so the trace can record them.
 */
export function singleHeadAttention(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
): { scores: Matrix; weights: Matrix; output: Matrix } {
  const seqLen = Q.length;
  const d_head = seqLen === 0 ? 0 : (Q[0]?.length ?? 0);
  const scale = 1 / Math.sqrt(Math.max(1, d_head));

  // scores = Q · Kᵀ / √d_head
  const Kt = transpose(K);
  const raw = matmul(Q, Kt);
  const scores: Matrix = new Array<Vector>(seqLen);
  for (let i = 0; i < seqLen; i++) {
    const r = raw[i]!;
    const sr = new Array<number>(r.length);
    for (let j = 0; j < r.length; j++) sr[j] = (r[j] ?? 0) * scale;
    scores[i] = sr;
  }

  // Apply causal mask in the softmax (no need to materialise scores+mask).
  const mask = causalMask(seqLen);
  const weights = softmaxRows(scores, mask);

  // output = weights · V
  const output = matmul(weights, V);
  return { scores, weights, output };
}

/**
 * Run causal multi-head self-attention, optionally writing intermediates
 * into `trace`.
 *
 * @param x       Input matrix of shape `[seq_len, d_model]`.
 * @param w       `{ W_q, W_k, W_v, W_o }`, each `[d_model, d_model]`.
 * @param nHeads  Number of attention heads. Must divide `d_model`.
 * @param trace   Optional `AttentionTrace`; mutated in place.
 * @returns       Output matrix of shape `[seq_len, d_model]`.
 */
export function multiHeadAttention(
  x: Matrix,
  w: AttentionWeights,
  nHeads: number,
  trace?: AttentionTrace,
): Matrix {
  const seqLen = x.length;
  const d_model = seqLen === 0 ? 0 : (x[0]?.length ?? 0);
  if (nHeads <= 0 || d_model % nHeads !== 0) {
    throw new Error(
      `multiHeadAttention: n_heads (${nHeads}) must divide d_model (${d_model})`,
    );
  }
  const d_head = d_model / nHeads;

  // Project to Q/K/V (full d_model).
  const Q = matmul(x, w.W_q);
  const K = matmul(x, w.W_k);
  const V = matmul(x, w.W_v);

  // Per-head attention.
  const headOutputs: Matrix[] = [];
  const allScores: Tensor3D = [];
  const allWeights: Tensor3D = [];
  for (let h = 0; h < nHeads; h++) {
    const Qh = sliceHead(Q, h, d_head);
    const Kh = sliceHead(K, h, d_head);
    const Vh = sliceHead(V, h, d_head);
    const { scores, weights, output } = singleHeadAttention(Qh, Kh, Vh);
    headOutputs.push(output);
    allScores.push(scores);
    allWeights.push(weights);
  }

  // Concat heads → [S, d_model], then project through W_o.
  const concat = concatHeads(headOutputs);
  const out = matmul(concat, w.W_o);

  if (trace) {
    trace.Q = Q;
    trace.K = K;
    trace.V = V;
    trace.scores = allScores;
    trace.weights = allWeights;
  }
  return out;
}
