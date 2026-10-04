import { describe, expect, it } from "vitest";

import {
  acceptanceRate,
  bestGamma,
  emittedDistribution,
  expectedTokens,
  residual,
  simulateSpeculative,
  speedup,
  verify,
} from "@/lib/inference/speculative";
import { softmax } from "@/lib/transformer/softmax";

const p = softmax([2.0, 1.0, 0.5, 0.0, -1.0, 1.5]);
const q = softmax([1.5, 1.2, 0.0, 0.3, -0.5, 1.0]);

describe("closed forms", () => {
  it("expected tokens per pass: geometric series", () => {
    expect(expectedTokens(0, 4)).toBe(1);
    expect(expectedTokens(1, 4)).toBe(5);
    // α = 0.8, γ = 4: 1 + .8 + .64 + .512 + .4096 = 3.3616
    expect(expectedTokens(0.8, 4)).toBeCloseTo(3.3616, 12);
  });

  it("speed-up divides by the draft cost (Theorem 3.8)", () => {
    expect(speedup(0.8, 4, 0)).toBeCloseTo(3.3616, 12);
    expect(speedup(0.8, 4, 0.05)).toBeCloseTo(3.3616 / 1.2, 12);
    expect(speedup(0, 3, 0.1)).toBeLessThan(1);
  });

  it("best γ grows with α and shrinks with draft cost", () => {
    expect(bestGamma(0.9, 0.05)).toBeGreaterThan(bestGamma(0.6, 0.05));
    expect(bestGamma(0.8, 0.3)).toBeLessThan(bestGamma(0.8, 0.01));
  });
});

describe("the verify rule preserves the target distribution", () => {
  it("emitted distribution equals p exactly (to rounding)", () => {
    const e = emittedDistribution(p, q);
    e.forEach((v, i) => expect(v).toBeCloseTo(p[i]!, 14));
  });

  it("acceptance rate is Σ min(p, q), and 1 when the draft is the target", () => {
    expect(acceptanceRate(p, p)).toBeCloseTo(1, 14);
    expect(acceptanceRate(p, q)).toBeGreaterThan(0.5);
    expect(acceptanceRate(p, q)).toBeLessThan(1);
  });

  it("the residual is a distribution on the tokens p favours over q", () => {
    const r = residual(p, q);
    expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14);
    r.forEach((v, i) => {
      if (p[i]! <= q[i]!) expect(v).toBe(0);
    });
    expect(residual(p, p)).toEqual(p);
  });

  it("verify accepts below min(1, p/q) and otherwise resamples from the residual", () => {
    const x = 0; // p[0] > q[0], so always accepted
    expect(verify(p, q, x, 0.999, () => 0.5).accepted).toBe(true);
    const y = 3; // p[3] < q[3]
    const ratio = p[y]! / q[y]!;
    expect(verify(p, q, y, ratio - 1e-9, () => 0.5).accepted).toBe(true);
    const rej = verify(p, q, y, ratio + 1e-9, () => 0.5);
    expect(rej.accepted).toBe(false);
    expect(residual(p, q)[rej.token]).toBeGreaterThan(0);
  });

  it("Monte Carlo matches the closed forms and the target histogram", () => {
    const gamma = 4;
    const sim = simulateSpeculative(p, q, gamma, 40_000, 1);
    const alpha = acceptanceRate(p, q);
    expect(Math.abs(sim.acceptance - alpha)).toBeLessThan(0.01);
    expect(
      Math.abs(sim.tokensPerPass - expectedTokens(alpha, gamma)),
    ).toBeLessThan(0.03);
    sim.histogram.forEach((h, i) =>
      expect(Math.abs(h - p[i]!)).toBeLessThan(0.01),
    );
  });
});
