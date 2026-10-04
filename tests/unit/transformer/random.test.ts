import { describe, expect, it } from "vitest";

import { mulberry32, normalSampler } from "@/lib/transformer/random";

describe("mulberry32", () => {
  it("yields uniforms in [0, 1)", () => {
    const rng = mulberry32(42);
    for (let i = 0; i < 100; i++) {
      const u = rng();
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(1);
    }
  });

  it("is deterministic for a given seed", () => {
    const a = mulberry32(7);
    const b = mulberry32(7);
    for (let i = 0; i < 5; i++) expect(a()).toBe(b());
  });

  it("differs across seeds", () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    // Vanishingly unlikely that the first three samples all collide by chance.
    let same = 0;
    for (let i = 0; i < 3; i++) if (a() === b()) same++;
    expect(same).toBeLessThan(3);
  });
});

describe("normalSampler", () => {
  it("produces approximately N(0, 1) over many samples", () => {
    const sample = normalSampler(mulberry32(123));
    const n = 5_000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const x = sample();
      sum += x;
      sumSq += x * x;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.abs(variance - 1)).toBeLessThan(0.1);
  });
});
