/*
 * src/lib/transformer/sampling.ts
 *
 * Operation: turn a logits vector into a sampled token id.
 * Shapes:    sample(logits: [V]) → number
 * Intuition: four modes, each used by the /learn/decoder/07-sampling page:
 *              greedy       — argmax(logits) — boring but deterministic.
 *              temperature  — softmax(logits / τ), then sample.
 *              top-k        — softmax over the top-k logits, then sample.
 *              top-p        — softmax over the smallest set of tokens whose
 *                             cumulative probability exceeds p.
 * MDX:       /learn/decoder/07-sampling.
 */
import { softmax } from "./softmax";
import type { Vector } from "./types";

export type SampleMode =
  | { kind: "greedy" }
  | { kind: "temperature"; temperature: number }
  | { kind: "top-k"; k: number; temperature?: number }
  | { kind: "top-p"; p: number; temperature?: number };

/**
 * Pluggable RNG. Defaults to `Math.random`, but tests / reproducible
 * playgrounds can pass a seeded function.
 */
export type RNG = () => number;

/** Argmax over a vector. Ties broken by the lower index. */
export function argmax(v: Vector): number {
  let best = -Infinity;
  let bestIdx = 0;
  for (let i = 0; i < v.length; i++) {
    const x = v[i] ?? 0;
    if (x > best) {
      best = x;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Sample an index from a probability vector using cumulative-sum + uniform.
 * `probs` does not need to sum exactly to 1 (we normalise on the fly).
 */
export function sampleFromProbs(probs: Vector, rng: RNG = Math.random): number {
  let total = 0;
  for (const p of probs) total += p;
  if (total <= 0) return 0;
  const u = rng() * total;
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i] ?? 0;
    if (u < acc) return i;
  }
  return probs.length - 1;
}

/** Apply temperature scaling: returns a fresh array. τ ≤ 0 falls back to argmax. */
function applyTemperature(logits: Vector, t: number): Vector {
  if (!Number.isFinite(t) || t <= 0) {
    // Caller should treat this as "greedy"; we degrade to a one-hot.
    const out = new Array<number>(logits.length).fill(-Infinity);
    out[argmax(logits)] = 0;
    return out;
  }
  const out = new Array<number>(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = (logits[i] ?? 0) / t;
  return out;
}

/** Top-k mask: keep the k largest logits, set the rest to −∞. */
function topKMask(logits: Vector, k: number): Vector {
  const n = logits.length;
  if (k <= 0) return logits.slice();
  if (k >= n) return logits.slice();
  // Find the k-th largest value (cheap for small n; we expect V ≤ 128).
  const sorted = logits.slice().sort((a, b) => b - a);
  const threshold = sorted[k - 1] ?? -Infinity;
  const out = new Array<number>(n);
  let kept = 0;
  for (let i = 0; i < n; i++) {
    const v = logits[i] ?? -Infinity;
    if (v > threshold || (v === threshold && kept < k)) {
      out[i] = v;
      kept++;
    } else {
      out[i] = -Infinity;
    }
  }
  return out;
}

/** Top-p (nucleus): keep the smallest set of indices whose cumulative
 * probability mass exceeds `p`, mask the rest. Operates on logits. */
function topPMask(logits: Vector, p: number): Vector {
  const n = logits.length;
  if (p <= 0 || p >= 1) return logits.slice();
  const probs = softmax(logits);
  const order: number[] = probs
    .map((_, i) => i)
    .sort((a, b) => (probs[b] ?? 0) - (probs[a] ?? 0));
  const keep = new Set<number>();
  let cum = 0;
  for (const i of order) {
    keep.add(i);
    cum += probs[i] ?? 0;
    if (cum >= p) break;
  }
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++)
    out[i] = keep.has(i) ? (logits[i] ?? -Infinity) : -Infinity;
  return out;
}

/**
 * Sample one token id from a logits vector under the given mode.
 *
 * @param logits  Logits over the vocabulary (length V).
 * @param mode    Sampling strategy.
 * @param rng     Optional pluggable uniform-[0,1) RNG (default Math.random).
 */
export function sample(
  logits: Vector,
  mode: SampleMode,
  rng: RNG = Math.random,
): number {
  switch (mode.kind) {
    case "greedy":
      return argmax(logits);
    case "temperature": {
      const probs = softmax(applyTemperature(logits, mode.temperature));
      return sampleFromProbs(probs, rng);
    }
    case "top-k": {
      const t = mode.temperature ?? 1;
      const masked = topKMask(applyTemperature(logits, t), mode.k);
      const probs = softmax(masked);
      return sampleFromProbs(probs, rng);
    }
    case "top-p": {
      const t = mode.temperature ?? 1;
      const masked = topPMask(applyTemperature(logits, t), mode.p);
      const probs = softmax(masked);
      return sampleFromProbs(probs, rng);
    }
  }
}
