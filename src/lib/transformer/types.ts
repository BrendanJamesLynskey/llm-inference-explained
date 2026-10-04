/*
 * src/lib/transformer/types.ts
 *
 * Operation: shared tensor type aliases used across the maths layer.
 * Shapes:    n/a (this file is type-only).
 * Intuition: deliberately plain `number[]` arrays — no class hierarchy.
 *            The point of this codebase is that the maths is *visible*.
 *            See CLAUDE.md §6 ("Tensor representation").
 * MDX:       referenced from every section of /learn/decoder/*.
 */

/** A 1-D tensor / row / vector. */
export type Vector = number[];

/** A 2-D tensor: row-major `[rows, cols]`. `m[i][j]`. */
export type Matrix = number[][];

/** A 3-D tensor: `[outer, rows, cols]`. Used for per-head attention scores. */
export type Tensor3D = number[][][];
