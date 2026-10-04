import { describe, expect, it } from "vitest";

import {
  attentionNaive,
  attentionSplitK,
  attentionTiled,
  hbmFlash,
  hbmStandard,
  mergePartials,
} from "@/lib/inference/attentionKernels";
import { mulberry32, normalSampler } from "@/lib/transformer/random";
import type { Matrix } from "@/lib/transformer/types";

function randn(rows: number, cols: number, seed: number, scale = 1): Matrix {
  const n = normalSampler(mulberry32(seed));
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => n() * scale),
  );
}

function maxAbsDiff(a: Matrix, b: Matrix): number {
  let m = 0;
  a.forEach((row, i) =>
    row.forEach((v, j) => (m = Math.max(m, Math.abs(v - b[i]![j]!)))),
  );
  return m;
}

describe("tiled attention is exact", () => {
  for (const [N, d, B] of [
    [16, 8, 4],
    [37, 16, 5],
    [64, 32, 64],
    [50, 8, 1],
  ] as const) {
    it(`N=${N}, d=${d}, tile ${B}: matches the textbook output to rounding`, () => {
      // Large logits (scale 3) make the running-max rescaling matter.
      const Q = randn(N, d, 1, 3);
      const K = randn(N, d, 2, 3);
      const V = randn(N, d, 3);
      expect(
        maxAbsDiff(attentionTiled(Q, K, V, B), attentionNaive(Q, K, V)),
      ).toBeLessThan(1e-12);
    });
  }
});

describe("split-K decode is exact", () => {
  it("any number of splits gives the single-pass output", () => {
    const K = randn(1000, 16, 4, 2);
    const V = randn(1000, 16, 5);
    const q = randn(1, 16, 6, 2)[0]!;
    const ref = attentionNaive([q], K, V)[0]!;
    for (const s of [1, 2, 7, 16, 1000, 1500]) {
      const { out, partials } = attentionSplitK(q, K, V, s);
      expect(partials).toHaveLength(s);
      expect(maxAbsDiff([out], [ref])).toBeLessThan(1e-12);
    }
  });

  it("merging is commutative and empty states are identities", () => {
    const K = randn(20, 4, 7);
    const V = randn(20, 4, 8);
    const q = randn(1, 4, 9)[0]!;
    const { partials } = attentionSplitK(q, K, V, 2);
    const ab = mergePartials(partials[0]!, partials[1]!);
    const ba = mergePartials(partials[1]!, partials[0]!);
    expect(ab.l).toBeCloseTo(ba.l, 12);
    const e = { m: -Infinity, l: 0, o: [0, 0, 0, 0] };
    expect(mergePartials(e, partials[0]!)).toEqual(partials[0]);
  });
});

describe("HBM traffic", () => {
  it("standard attention moves 4Nd + 4N² elements", () => {
    expect(hbmStandard(1024, 64)).toBe(4 * 1024 * 64 + 4 * 1024 * 1024);
  });

  it("FlashAttention moves far less for realistic on-chip memory, and less still with more", () => {
    const N = 4096;
    const d = 64;
    const M = (100 * 1024) / 2; // ~100 KB of SRAM in 16-bit elements
    expect(hbmFlash(N, d, M)).toBeLessThan(hbmStandard(N, d) / 3);
    expect(hbmFlash(N, d, 4 * M)).toBeLessThan(hbmFlash(N, d, M));
    // With on-chip memory for the whole problem, one pass: ≈ 5Nd + 4N.
    expect(hbmFlash(N, d, 4 * d * N)).toBe(2 * N * d + (3 * N * d + 4 * N));
  });

  it("grows like N² for standard and N²d²/M for flash", () => {
    const d = 64;
    const M = 50_000;
    const r1 = hbmStandard(8192, d) / hbmStandard(4096, d);
    const r2 = hbmFlash(8192, d, M) / hbmFlash(4096, d, M);
    expect(r1).toBeGreaterThan(3.5);
    expect(r2).toBeGreaterThan(3.5);
  });
});
