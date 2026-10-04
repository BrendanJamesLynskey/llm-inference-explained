/*
 * src/lib/transformer/matmul.ts
 *
 * Operation: 2-D matrix multiplication, naive triple loop.
 * Shapes:    A [M, K] × B [K, N] → C [M, N].
 * Intuition: C[i,j] = Σ_k A[i,k] * B[k,j]. Visible, no BLAS, no tricks.
 *            Tiny inputs (seq_len ≤ 16, d_model ≤ 64) — readability wins.
 * MDX:       /learn/decoder/03-attention (Q = X·W_q etc.).
 */
import type { Matrix } from "./types";

/**
 * Multiply two matrices.
 *
 * @param A  Matrix of shape `[M, K]`.
 * @param B  Matrix of shape `[K, N]`.
 * @returns  Matrix of shape `[M, N]`.
 *
 * Throws on shape mismatch.
 */
export function matmul(A: Matrix, B: Matrix): Matrix {
  const M = A.length;
  if (M === 0) return [];
  const K = A[0]?.length ?? 0;
  const Kb = B.length;
  if (K !== Kb) {
    throw new Error(
      `matmul: inner dim mismatch (A is ${M}×${K}, B is ${Kb}×?)`,
    );
  }
  const N = B[0]?.length ?? 0;

  // C[i,j] = Σ_k A[i,k] · B[k,j]
  const C: Matrix = new Array<number[]>(M);
  for (let i = 0; i < M; i++) {
    const Ai = A[i]!;
    const Ci = new Array<number>(N).fill(0);
    for (let k = 0; k < K; k++) {
      const a = Ai[k] ?? 0;
      if (a === 0) continue;
      const Bk = B[k]!;
      for (let j = 0; j < N; j++) Ci[j] = (Ci[j] ?? 0) + a * (Bk[j] ?? 0);
    }
    C[i] = Ci;
  }
  return C;
}

/**
 * Transpose a matrix. `transpose(A)[j][i] === A[i][j]`.
 *
 * @param A  Matrix of shape `[M, N]`.
 * @returns  Matrix of shape `[N, M]`.
 */
export function transpose(A: Matrix): Matrix {
  const M = A.length;
  if (M === 0) return [];
  const N = A[0]?.length ?? 0;
  const T: Matrix = new Array<number[]>(N);
  for (let j = 0; j < N; j++) {
    const Tj = new Array<number>(M);
    for (let i = 0; i < M; i++) Tj[i] = A[i]?.[j] ?? 0;
    T[j] = Tj;
  }
  return T;
}
