/*
 * src/lib/transformer/softmax.ts
 *
 * Operation: numerically-stable softmax over a 1-D vector.
 * Shapes:    softmax(x: [N]) → [N], with optional additive mask: [N].
 * Intuition: y_i = exp(x_i − max(x)) / Σ_j exp(x_j − max(x)). The shift by
 *            max(x) prevents overflow when x is large; the result is the
 *            same as plain exp / Σ exp by algebra.
 * MDX:       /learn/decoder/03-attention (turning scores → weights).
 */
import type { Matrix, Vector } from "./types";

/**
 * Apply softmax row-wise to a 1-D vector.
 *
 * If `mask` is supplied (same length as `x`), it is added to `x` before the
 * exp — pass `-Infinity` in masked positions to zero them out (a row that
 * is *entirely* masked returns all-zeros rather than NaNs, which keeps the
 * UI sane).
 *
 * @param x     Input vector of length `N`.
 * @param mask  Optional additive mask of length `N`. Use `-Infinity` to mask.
 * @returns     Probability vector of length `N`.
 */
export function softmax(x: Vector, mask?: Vector): Vector {
  const n = x.length;
  if (n === 0) return [];

  // 1) shift by the max (over the unmasked positions) for numerical stability.
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = (x[i] ?? 0) + (mask?.[i] ?? 0);
    if (v > max) max = v;
  }

  // 2) exp(shifted), accumulate sum.
  const out = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = (x[i] ?? 0) + (mask?.[i] ?? 0);
    // If max is -Infinity (whole row masked) every output should be 0.
    const e =
      Number.isFinite(max) && Number.isFinite(v) ? Math.exp(v - max) : 0;
    out[i] = e;
    sum += e;
  }

  // 3) normalise. Guard against the all-masked case (sum === 0) by emitting 0.
  if (sum === 0) return out;
  for (let i = 0; i < n; i++) out[i] = (out[i] ?? 0) / sum;
  return out;
}

/**
 * Convenience: row-wise softmax over a matrix. Each row is softmaxed
 * independently. `mask` (if given) must be the same shape as `x` and is
 * applied row-wise.
 */
export function softmaxRows(x: Matrix, mask?: Matrix): Matrix {
  const out: Matrix = new Array<Vector>(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = softmax(x[i]!, mask?.[i]);
  }
  return out;
}
