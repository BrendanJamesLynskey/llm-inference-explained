/*
 * src/lib/inference/speculative.ts
 *
 * Operation: speculative decoding — a cheap draft model proposes γ tokens,
 *            the target model checks them all in one forward pass, and a
 *            rejection rule keeps the output distribution exactly the
 *            target's.
 * Shapes:    distributions are probability vectors over the vocabulary.
 * Intuition: decode is memory-bound, so checking γ+1 positions costs about
 *            the same as producing one. If the draft is usually right, each
 *            target pass yields several tokens.
 * MDX:       /learn/07-speculative-decoding.
 *
 * After Leviathan, Kalman and Matias, "Fast Inference from Transformers via
 * Speculative Decoding" (arXiv:2211.17192), and Chen et al., "Accelerating
 * Large Language Model Decoding with Speculative Sampling"
 * (arXiv:2302.01318).
 */
import { mulberry32 } from "@/lib/transformer/random";
import { sampleFromProbs } from "@/lib/transformer/sampling";
import type { Vector } from "@/lib/transformer/types";

/**
 * Acceptance rate of one drafted token: β = Σₓ min(p(x), q(x)) for target p
 * and draft q (Leviathan et al. §3). α is its expectation over contexts.
 */
export function acceptanceRate(p: Vector, q: Vector): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += Math.min(p[i]!, q[i]!);
  return s;
}

/**
 * Expected tokens per target pass with γ drafted tokens and i.i.d.
 * acceptance α: (1 − α^{γ+1}) / (1 − α). (One more token always comes from
 * the target, accepted run or not.)
 */
export function expectedTokens(alpha: number, gamma: number): number {
  if (alpha >= 1) return gamma + 1;
  return (1 - alpha ** (gamma + 1)) / (1 - alpha);
}

/**
 * Expected wall-time speed-up when one draft step costs `c` of a target
 * step: (1 − α^{γ+1}) / ((1 − α)(γc + 1)) (Leviathan et al. Theorem 3.8).
 */
export function speedup(alpha: number, gamma: number, c: number): number {
  return expectedTokens(alpha, gamma) / (gamma * c + 1);
}

/** The γ in [1, maxGamma] with the best speed-up. */
export function bestGamma(alpha: number, c: number, maxGamma = 16): number {
  let best = 1;
  for (let g = 2; g <= maxGamma; g++) {
    if (speedup(alpha, g, c) > speedup(alpha, best, c)) best = g;
  }
  return best;
}

/** The residual distribution norm(max(0, p − q)) used after a rejection. */
export function residual(p: Vector, q: Vector): Vector {
  const r = p.map((pi, i) => Math.max(0, pi - q[i]!));
  const z = r.reduce((a, b) => a + b, 0);
  return z > 0 ? r.map((x) => x / z) : p.slice();
}

/**
 * The exact distribution of the token the verify rule emits for one drafted
 * position: accept x ~ q with probability min(1, p(x)/q(x)), otherwise draw
 * from the residual. Algebraically this is p; computing it shows that.
 */
export function emittedDistribution(p: Vector, q: Vector): Vector {
  const beta = acceptanceRate(p, q);
  const r = residual(p, q);
  return p.map((_, x) => Math.min(p[x]!, q[x]!) + (1 - beta) * r[x]!);
}

/**
 * One verify step: is drafted token `x` (drawn from q) accepted, given a
 * uniform draw `u`? If not, the replacement drawn from the residual.
 */
export function verify(
  p: Vector,
  q: Vector,
  x: number,
  u: number,
  rng: () => number,
): { accepted: boolean; token: number } {
  if (u < Math.min(1, p[x]! / q[x]!)) return { accepted: true, token: x };
  return { accepted: false, token: sampleFromProbs(residual(p, q), rng) };
}

/**
 * Monte Carlo: `rounds` target passes with γ drafted tokens each, using
 * fixed per-position target and draft distributions p and q (a stationary
 * toy, so acceptance is i.i.d. with α = Σ min(p, q)). Returns the measured
 * tokens per pass, acceptance rate and the histogram of emitted tokens.
 */
export function simulateSpeculative(
  p: Vector,
  q: Vector,
  gamma: number,
  rounds: number,
  seed: number,
): { tokensPerPass: number; acceptance: number; histogram: Vector } {
  const rng = mulberry32(seed);
  const hist = new Array<number>(p.length).fill(0);
  let tokens = 0;
  let accepted = 0;
  let drafted = 0;
  for (let r = 0; r < rounds; r++) {
    let k = 0;
    for (; k < gamma; k++) {
      const x = sampleFromProbs(q, rng);
      drafted++;
      const v = verify(p, q, x, rng(), rng);
      hist[v.token]!++;
      tokens++;
      if (!v.accepted) break;
      accepted++;
    }
    if (k === gamma) {
      // Every draft accepted: the target's pass also yields one bonus token.
      hist[sampleFromProbs(p, rng)]!++;
      tokens++;
    }
  }
  return {
    tokensPerPass: tokens / rounds,
    acceptance: accepted / drafted,
    histogram: hist.map((h) => h / tokens),
  };
}
