/*
 * src/lib/transformer/random.ts
 *
 * Operation: seeded uniform + normal RNG so the API can produce
 *            deterministic weights without bundling fixture JSON.
 * Shapes:    n/a — scalar primitives.
 * Intuition: mulberry32 for uniform; Box-Muller for the standard normal.
 *            Same seed → same sequence; different seed → different model.
 *            We do **not** try to match PyTorch byte-for-byte here — the
 *            verify-maths fixtures still drive correctness against PyTorch.
 *            This RNG is for *visualisation* (the live playground/widgets).
 * MDX:       /learn/decoder/02-embeddings (seed control).
 */

/**
 * Mulberry32 — 32-bit PRNG. Tiny, deterministic, and good enough for our
 * "generate a small embedding table" use case.
 *
 * @param seed  Any 32-bit integer.
 * @returns     A function that yields a uniform float in [0, 1).
 */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Standard normal sampler from a uniform RNG. Box-Muller transform —
 * generates a Z ~ N(0, 1) per call (we discard one of the two samples).
 */
export function normalSampler(rng: () => number): () => number {
  return () => {
    // Avoid log(0) by clamping the lower end of the open interval.
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}
