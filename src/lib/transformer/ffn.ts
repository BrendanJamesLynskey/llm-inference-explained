/*
 * src/lib/transformer/ffn.ts
 *
 * Operation: position-wise feed-forward network.
 * Shapes:    ffn(x: [S, D], W1: [D, D_ff], b1: [D_ff],
 *                W2: [D_ff, D], b2: [D])
 *               → [S, D]
 * Intuition: y = GELU(x W₁ + b₁) W₂ + b₂. Two linear layers wrapped around
 *            a non-linearity, applied independently at every sequence
 *            position. This is where most of the model's parameters sit.
 * MDX:       /learn/decoder/04-ffn.
 */
import { addRowBias } from "./tensor";
import { geluMat } from "./gelu";
import { matmul } from "./matmul";
import type { Matrix, Vector } from "./types";

export type FFNWeights = {
  W1: Matrix;
  b1: Vector;
  W2: Matrix;
  b2: Vector;
};

/** Trace fields recorded by `ffn` when a trace is provided. */
export type FFNTrace = {
  pre: Matrix; // x · W1 + b1
  act: Matrix; // GELU(pre)
  out: Matrix; // act · W2 + b2
};

/**
 * Run the position-wise FFN.
 *
 * @param x      Input matrix `[S, D]`.
 * @param w      `{ W1: [D, D_ff], b1: [D_ff], W2: [D_ff, D], b2: [D] }`.
 * @param trace  Optional record-keeping object; mutated in place.
 * @returns      Output matrix `[S, D]`.
 */
export function ffn(x: Matrix, w: FFNWeights, trace?: FFNTrace): Matrix {
  const pre = addRowBias(matmul(x, w.W1), w.b1);
  const act = geluMat(pre);
  const out = addRowBias(matmul(act, w.W2), w.b2);

  if (trace) {
    trace.pre = pre;
    trace.act = act;
    trace.out = out;
  }
  return out;
}
