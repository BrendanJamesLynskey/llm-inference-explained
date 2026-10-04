import { describe, expect, it } from "vitest";

import { gelu, geluMat, geluVec, relu } from "@/lib/transformer/gelu";

describe("gelu", () => {
  it("is zero at zero", () => {
    expect(gelu(0)).toBe(0);
  });

  it("approximately matches the reference at small values", () => {
    // PyTorch's tanh-approx GELU values, computed independently:
    expect(gelu(1)).toBeCloseTo(0.84119, 4);
    expect(gelu(-1)).toBeCloseTo(-0.15881, 4);
    expect(gelu(2)).toBeCloseTo(1.95459, 3);
  });

  it("has the expected sign behaviour: negative on x < 0, positive on x > 0", () => {
    // GELU is *not* monotonic — it has a small dip around x ≈ -0.75 — but its
    // sign matches x's outside that dip.
    expect(gelu(-3)).toBeLessThan(0);
    expect(gelu(-0.1)).toBeLessThan(0);
    expect(gelu(0)).toBe(0);
    expect(gelu(0.1)).toBeGreaterThan(0);
    expect(gelu(3)).toBeGreaterThan(0);
  });

  it("is monotonically increasing for x ≥ 0", () => {
    let prev = -Infinity;
    for (let x = 0; x <= 5; x += 0.1) {
      const y = gelu(x);
      expect(y).toBeGreaterThanOrEqual(prev - 1e-12);
      prev = y;
    }
  });
});

describe("geluVec / geluMat", () => {
  it("applies elementwise", () => {
    const v = geluVec([0, 1, 2]);
    expect(v[0]!).toBe(0);
    expect(v[1]!).toBeCloseTo(gelu(1), 12);
    expect(v[2]!).toBeCloseTo(gelu(2), 12);

    const m = geluMat([
      [0, 1],
      [-1, 2],
    ]);
    expect(m[0]![0]!).toBe(0);
    expect(m[1]![0]!).toBeCloseTo(gelu(-1), 12);
  });
});

describe("relu", () => {
  it("clamps negatives to zero", () => {
    expect(relu(-1)).toBe(0);
    expect(relu(0)).toBe(0);
    expect(relu(2)).toBe(2);
  });
});
