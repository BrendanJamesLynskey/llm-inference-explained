/*
 * src/lib/transformer/tensor.ts
 *
 * Operation: small constructors and shape helpers for tensors.
 * Shapes:    create2D(rows, cols, fill)        -> [rows, cols]
 *            zeros(rows, cols), zerosVec(n)
 *            shape(m)                          -> [rows, cols]
 * Intuition: just enough to allocate matrices without obscuring the maths.
 *            We deliberately use `number[][]` rather than a Tensor class.
 * MDX:       /learn/decoder/01-overview (passes through everything).
 */
import type { Matrix, Vector } from "./types";

/** Allocate a vector of length `n` initialised to `fill` (default 0). */
export function zerosVec(n: number, fill = 0): Vector {
  const out: Vector = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = fill;
  return out;
}

/** Allocate a `rows × cols` matrix initialised to `fill` (default 0). */
export function zeros(rows: number, cols: number, fill = 0): Matrix {
  const out: Matrix = new Array<Vector>(rows);
  for (let i = 0; i < rows; i++) {
    out[i] = zerosVec(cols, fill);
  }
  return out;
}

/** Return `[rows, cols]`. Throws if `m` is empty (no canonical width). */
export function shape(m: Matrix): readonly [number, number] {
  const rows = m.length;
  const cols = rows === 0 ? 0 : (m[0]?.length ?? 0);
  return [rows, cols];
}

/** Element-wise add: c = a + b. `a` and `b` must have the same length. */
export function addVec(a: Vector, b: Vector): Vector {
  if (a.length !== b.length) {
    throw new Error(`addVec: length mismatch (${a.length} vs ${b.length})`);
  }
  const out: Vector = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] ?? 0) + (b[i] ?? 0);
  return out;
}

/** Row-wise matrix add: same shape required. */
export function addMat(a: Matrix, b: Matrix): Matrix {
  const [ra, ca] = shape(a);
  const [rb, cb] = shape(b);
  if (ra !== rb || ca !== cb) {
    throw new Error(`addMat: shape mismatch (${ra}×${ca} vs ${rb}×${cb})`);
  }
  const out: Matrix = new Array<Vector>(ra);
  for (let i = 0; i < ra; i++) out[i] = addVec(a[i]!, b[i]!);
  return out;
}

/** Add `bias` (length C) to every row of `m` (R×C). Returns a new matrix. */
export function addRowBias(m: Matrix, bias: Vector): Matrix {
  const [r, c] = shape(m);
  if (c !== bias.length) {
    throw new Error(`addRowBias: cols (${c}) ≠ bias length (${bias.length})`);
  }
  const out: Matrix = new Array<Vector>(r);
  for (let i = 0; i < r; i++) {
    const row = m[i]!;
    const o = new Array<number>(c);
    for (let j = 0; j < c; j++) o[j] = (row[j] ?? 0) + (bias[j] ?? 0);
    out[i] = o;
  }
  return out;
}
