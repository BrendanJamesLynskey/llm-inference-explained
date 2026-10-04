/*
 * src/lib/transformer/layernorm.ts
 *
 * Operation: layer normalisation over the last dimension.
 * Shapes:    layernorm(x: [D], γ: [D], β: [D]) → [D]
 * Intuition: y_i = γ_i · (x_i − μ) / √(σ² + ε) + β_i, where μ and σ² are
 *            the mean and (population) variance of `x`. ε = 1e-5.
 *            Population variance (1/N), not sample (1/(N-1)) — matches
 *            PyTorch `var(unbiased=False)`.
 * MDX:       /learn/decoder/05-layernorm-residuals.
 */
import type { Matrix, Vector } from "./types";

/** Default epsilon for numerical stability. Matches PyTorch's default. */
export const LAYERNORM_EPS = 1e-5;

/**
 * Apply layer normalisation to a single 1-D vector.
 *
 * @param x     Input vector of length `D`.
 * @param gamma Scale of length `D`. Pass `Array(D).fill(1)` for identity.
 * @param beta  Shift of length `D`. Pass `Array(D).fill(0)` for identity.
 * @param eps   Numerical-stability constant (default 1e-5).
 */
export function layernorm(
  x: Vector,
  gamma: Vector,
  beta: Vector,
  eps: number = LAYERNORM_EPS,
): Vector {
  const n = x.length;
  if (gamma.length !== n || beta.length !== n) {
    throw new Error(
      `layernorm: length mismatch (x=${n}, γ=${gamma.length}, β=${beta.length})`,
    );
  }

  // μ = (1/N) Σ_i x_i
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i] ?? 0;
  mean /= n;

  // σ² = (1/N) Σ_i (x_i − μ)²  — population variance.
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = (x[i] ?? 0) - mean;
    varSum += d * d;
  }
  const variance = varSum / n;
  const invStd = 1 / Math.sqrt(variance + eps);

  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = ((x[i] ?? 0) - mean) * invStd * (gamma[i] ?? 0) + (beta[i] ?? 0);
  }
  return out;
}

/** Row-wise layernorm over a matrix `[N, D]`. */
export function layernormRows(
  x: Matrix,
  gamma: Vector,
  beta: Vector,
  eps: number = LAYERNORM_EPS,
): Matrix {
  const out: Matrix = new Array<Vector>(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = layernorm(x[i]!, gamma, beta, eps);
  }
  return out;
}
