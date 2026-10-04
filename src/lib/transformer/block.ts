/*
 * src/lib/transformer/block.ts
 *
 * Operation: one pre-norm decoder block.
 * Shapes:    block(x: [S, D], BlockWeights) → [S, D]
 * Intuition: pre-norm means we LN *before* each sublayer rather than after,
 *            which trains more stably than the original post-norm:
 *                h = x + Attn(LN1(x))
 *                y = h + FFN(LN2(h))
 *            See SPEC §4 — we use pre-norm throughout.
 * MDX:       /learn/decoder/05-layernorm-residuals, 06-stacking.
 */
import { multiHeadAttention, type AttentionWeights } from "./attention";
import { ffn, type FFNWeights } from "./ffn";
import { addMat } from "./tensor";
import { layernormRows } from "./layernorm";
import type { BlockTrace } from "./trace";
import type { Matrix, Vector } from "./types";

/** All weights for one pre-norm decoder block. */
export type BlockWeights = {
  ln1: { gamma: Vector; beta: Vector };
  attn: AttentionWeights;
  ln2: { gamma: Vector; beta: Vector };
  ffn: FFNWeights;
};

/**
 * Run a pre-norm decoder block.
 *
 * @param x       Input matrix `[S, D]`.
 * @param w       Block weights — see `BlockWeights`.
 * @param nHeads  Number of attention heads (must divide `D`).
 * @param trace   Optional `BlockTrace`; mutated in place if provided.
 */
export function block(
  x: Matrix,
  w: BlockWeights,
  nHeads: number,
  trace?: BlockTrace,
): Matrix {
  // Sub-layer 1: residual + masked self-attention(LN).
  const ln1 = layernormRows(x, w.ln1.gamma, w.ln1.beta);
  const attnOut = multiHeadAttention(ln1, w.attn, nHeads, trace?.attn);
  const h = addMat(x, attnOut);

  // Sub-layer 2: residual + position-wise FFN(LN).
  const ln2 = layernormRows(h, w.ln2.gamma, w.ln2.beta);
  const ffnTrace = trace
    ? { pre: [] as Matrix, act: [] as Matrix, out: [] as Matrix }
    : undefined;
  const ffnOut = ffn(ln2, w.ffn, ffnTrace);
  const y = addMat(h, ffnOut);

  if (trace) {
    trace.ln1 = ln1;
    trace.attnOut = h;
    trace.ln2 = ln2;
    trace.ffnPre = ffnTrace!.pre;
    trace.ffnAct = ffnTrace!.act;
    trace.ffnOut = y;
  }
  return y;
}
