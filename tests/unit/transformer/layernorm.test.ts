import { describe, expect, it } from "vitest";

import {
  LAYERNORM_EPS,
  layernorm,
  layernormRows,
} from "@/lib/transformer/layernorm";

describe("layernorm", () => {
  it("normalises to mean 0, variance 1 with γ=1, β=0", () => {
    const x = [1, 2, 3, 4];
    const y = layernorm(x, [1, 1, 1, 1], [0, 0, 0, 0]);
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    const v = y.reduce((acc, b) => acc + (b - mean) ** 2, 0) / y.length;
    expect(Math.abs(mean)).toBeLessThan(1e-9);
    // Population variance ≈ 1 (only the ε in the denom keeps it under 1).
    expect(Math.abs(v - 1)).toBeLessThan(1e-3);
  });

  it("applies γ as a scale and β as a shift", () => {
    const x = [1, 2, 3];
    const baseline = layernorm(x, [1, 1, 1], [0, 0, 0]);
    const scaled = layernorm(x, [2, 2, 2], [0, 0, 0]);
    for (let i = 0; i < 3; i++) {
      expect(scaled[i]!).toBeCloseTo((baseline[i] ?? 0) * 2, 12);
    }
    const shifted = layernorm(x, [1, 1, 1], [10, 20, 30]);
    expect(shifted[0]!).toBeCloseTo((baseline[0] ?? 0) + 10, 12);
    expect(shifted[1]!).toBeCloseTo((baseline[1] ?? 0) + 20, 12);
    expect(shifted[2]!).toBeCloseTo((baseline[2] ?? 0) + 30, 12);
  });

  it("respects a non-default epsilon", () => {
    // With ε huge, invStd shrinks and the normalised output collapses
    // toward β (the (x-μ)·invStd·γ term goes to 0).
    const y = layernorm([1, 2, 3], [1, 1, 1], [10, 20, 30], 1e15);
    expect(y[0]!).toBeCloseTo(10, 5);
    expect(y[1]!).toBeCloseTo(20, 5);
    expect(y[2]!).toBeCloseTo(30, 5);
  });

  it("throws on length mismatch", () => {
    expect(() => layernorm([1, 2], [1], [0, 0])).toThrow(/length mismatch/);
    expect(() => layernorm([1, 2], [1, 1], [0])).toThrow(/length mismatch/);
  });

  it("LAYERNORM_EPS matches PyTorch's default", () => {
    expect(LAYERNORM_EPS).toBe(1e-5);
  });
});

describe("layernormRows", () => {
  it("normalises every row independently", () => {
    // Two identical rows must give identical outputs. (Two rows with the
    // same shape but different scales are *almost* equal because ε breaks
    // exact scale-invariance — that's covered by `layernorm` directly.)
    const rows = layernormRows(
      [
        [1, 2, 3],
        [1, 2, 3],
      ],
      [1, 1, 1],
      [0, 0, 0],
    );
    expect(rows[0]).toEqual(rows[1]);
  });
});
