import { describe, expect, it } from "vitest";

import { argmax, sample, sampleFromProbs } from "@/lib/transformer/sampling";

/** Tiny seeded RNG so sampling tests are deterministic. */
function seededRng(seed: number): () => number {
  // mulberry32
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("argmax", () => {
  it("returns the index of the largest value", () => {
    expect(argmax([1, 3, 2])).toBe(1);
  });

  it("breaks ties to the lower index", () => {
    expect(argmax([3, 3, 3])).toBe(0);
  });
});

describe("sampleFromProbs", () => {
  it("returns 0 if all probabilities are zero", () => {
    expect(sampleFromProbs([0, 0, 0])).toBe(0);
  });

  it("returns the only nonzero index", () => {
    expect(sampleFromProbs([0, 0, 1, 0])).toBe(2);
  });

  it("falls through to the last index when u rounds up", () => {
    // RNG returns 1 (max possible), so cumulative sum never strictly exceeds u.
    expect(sampleFromProbs([0.5, 0.5], () => 0.999_999)).toBe(1);
  });

  it("returns the last index when rng() === 1 (defensive)", () => {
    // Math.random() never returns 1, but a custom RNG might. The tail return
    // catches that — without it, we'd fall off the loop with no return value.
    expect(sampleFromProbs([0.5, 0.5], () => 1)).toBe(1);
  });
});

describe("sample", () => {
  const logits = [0.1, 1.5, 0.2, -0.5, 3.0];

  it("greedy returns argmax", () => {
    expect(sample(logits, { kind: "greedy" })).toBe(4);
  });

  it("temperature ≤ 0 collapses to greedy", () => {
    expect(
      sample(logits, { kind: "temperature", temperature: 0 }, seededRng(1)),
    ).toBe(4);
  });

  it("temperature > 0 produces a valid index", () => {
    const idx = sample(
      logits,
      { kind: "temperature", temperature: 1 },
      seededRng(7),
    );
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(logits.length);
  });

  it("top-k restricts to the top-k logits", () => {
    // With k=1, only the argmax should ever be sampled.
    for (let s = 1; s < 10; s++) {
      const idx = sample(logits, { kind: "top-k", k: 1 }, seededRng(s));
      expect(idx).toBe(4);
    }
  });

  it("top-k with k≥V returns the same support as plain temperature", () => {
    const big = sample(logits, { kind: "top-k", k: 99 }, seededRng(11));
    expect(big).toBeGreaterThanOrEqual(0);
    expect(big).toBeLessThan(logits.length);
  });

  it("top-k with k≤0 returns the input slice (unmasked)", () => {
    const idx = sample(logits, { kind: "top-k", k: 0 }, seededRng(11));
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(logits.length);
  });

  it("top-p restricts to the cumulative-mass core", () => {
    // For very small p, only the highest-prob token should be reachable.
    for (let s = 1; s < 10; s++) {
      const idx = sample(logits, { kind: "top-p", p: 0.5 }, seededRng(s));
      // index 4 is far above the others, so its probability mass alone should
      // exceed 0.5 once temperature=1 is applied.
      expect(idx).toBe(4);
    }
  });

  it("top-p with p≥1 is unrestricted", () => {
    const idx = sample(logits, { kind: "top-p", p: 1 }, seededRng(13));
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(logits.length);
  });

  it("top-p with p≤0 is unrestricted", () => {
    const idx = sample(logits, { kind: "top-p", p: 0 }, seededRng(13));
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(logits.length);
  });
});
