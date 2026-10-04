/*
 * src/lib/transformer/gelu.ts
 *
 * Operation: GELU activation (tanh approximation) and ReLU for comparison.
 * Shapes:    scalar → scalar, with vector/matrix helpers.
 * Intuition: GELU smoothly gates a value by an approximate cumulative
 *            normal: y ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³))).
 *            We use the *tanh* approximation because PyTorch defaults to it
 *            for the GPT-style models we're emulating.
 * MDX:       /learn/decoder/04-ffn.
 */
import type { Matrix, Vector } from "./types";

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const GELU_COEF = 0.044715;

/**
 * GELU (tanh approximation). Single-scalar form so callers see the formula.
 */
export function gelu(x: number): number {
  // 0.5x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
  const x3 = x * x * x;
  const inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
  return 0.5 * x * (1 + Math.tanh(inner));
}

/** Element-wise GELU over a vector. */
export function geluVec(v: Vector): Vector {
  const out = new Array<number>(v.length);
  for (let i = 0; i < v.length; i++) out[i] = gelu(v[i] ?? 0);
  return out;
}

/** Element-wise GELU over a matrix. */
export function geluMat(m: Matrix): Matrix {
  const out: Matrix = new Array<Vector>(m.length);
  for (let i = 0; i < m.length; i++) out[i] = geluVec(m[i]!);
  return out;
}

/**
 * ReLU — kept here so the explainer can show the simpler activation
 * alongside GELU. Not used by the toy model itself.
 */
export function relu(x: number): number {
  return x > 0 ? x : 0;
}
