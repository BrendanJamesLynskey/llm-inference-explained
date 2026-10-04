import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { matmul, transpose } from "@/lib/transformer/matmul";

describe("matmul", () => {
  it("computes a 2×3 · 3×2 product by hand", () => {
    const A = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const B = [
      [7, 8],
      [9, 10],
      [11, 12],
    ];
    expect(matmul(A, B)).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });

  it("returns [] for an empty A", () => {
    expect(matmul([], [[1, 2]])).toEqual([]);
  });

  it("zero-skips on a zero element of A (correctness check)", () => {
    expect(
      matmul(
        [
          [0, 1],
          [2, 0],
        ],
        [
          [3, 4],
          [5, 6],
        ],
      ),
    ).toEqual([
      [5, 6],
      [6, 8],
    ]);
  });

  it("throws on inner-dim mismatch", () => {
    expect(() => matmul([[1, 2]], [[1, 2]])).toThrow(/inner dim mismatch/);
  });

  it("matches the PyTorch fixture", () => {
    const path = join("tests", "unit", "fixtures", "matmul.json");
    const f = JSON.parse(readFileSync(path, "utf-8")) as {
      A: number[][];
      B: number[][];
      output: number[][];
    };
    const got = matmul(f.A, f.B);
    let maxErr = 0;
    for (let i = 0; i < got.length; i++) {
      const gi = got[i]!;
      const oi = f.output[i]!;
      for (let j = 0; j < gi.length; j++) {
        const e = Math.abs((gi[j] ?? 0) - (oi[j] ?? 0));
        if (e > maxErr) maxErr = e;
      }
    }
    expect(maxErr).toBeLessThan(1e-5);
  });
});

describe("transpose", () => {
  it("transposes a 2×3 matrix to 3×2", () => {
    expect(
      transpose([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    ).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it("returns [] for an empty input", () => {
    expect(transpose([])).toEqual([]);
  });

  it("returns rows-first padding when a row is short", () => {
    // Defensive: if a row is shorter than expected, missing slots are 0.
    expect(transpose([[1, 2, 3], [4]])).toEqual([
      [1, 4],
      [2, 0],
      [3, 0],
    ]);
  });
});
