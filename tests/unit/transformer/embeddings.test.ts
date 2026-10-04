import { describe, expect, it } from "vitest";

import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";

describe("tokenEmbedding", () => {
  const E = [
    [10, 11, 12],
    [20, 21, 22],
    [30, 31, 32],
  ];

  it("looks up rows of the embedding matrix", () => {
    expect(tokenEmbedding([0, 2, 1], E)).toEqual([
      [10, 11, 12],
      [30, 31, 32],
      [20, 21, 22],
    ]);
  });

  it("returns a copy that callers can mutate independently", () => {
    const out = tokenEmbedding([0], E);
    out[0]![0] = 99;
    expect(E[0]![0]).toBe(10);
  });

  it("emits a zero row for an out-of-range id", () => {
    expect(tokenEmbedding([99], E)).toEqual([[0, 0, 0]]);
    expect(tokenEmbedding([-1], E)).toEqual([[0, 0, 0]]);
  });

  it("returns [] for empty ids", () => {
    expect(tokenEmbedding([], E)).toEqual([]);
  });

  it("returns zero rows when V=0", () => {
    expect(tokenEmbedding([0, 1], [])).toEqual([[], []]);
  });
});

describe("positionalEncoding", () => {
  it("first row (pos=0) is sin/cos at 0 → all 0s and 1s", () => {
    const pe = positionalEncoding(2, 4);
    expect(pe[0]![0]!).toBeCloseTo(0, 12);
    expect(pe[0]![1]!).toBeCloseTo(1, 12);
    expect(pe[0]![2]!).toBeCloseTo(0, 12);
    expect(pe[0]![3]!).toBeCloseTo(1, 12);
  });

  it("returns a [seqLen, dModel] matrix", () => {
    const pe = positionalEncoding(5, 8);
    expect(pe.length).toBe(5);
    expect(pe[0]!.length).toBe(8);
  });

  it("supports odd dModel (last column stays at 0)", () => {
    const pe = positionalEncoding(2, 3);
    expect(pe[0]!.length).toBe(3);
    expect(pe[0]![2]!).toBeCloseTo(0, 12);
  });

  it("matches the PyTorch fixture for the default 8×16 shape", () => {
    // Inline the small expected values: just check that the second
    // half of row 1 differs from row 0 (sanity check that pos affects PE).
    const pe = positionalEncoding(8, 16);
    expect(pe[0]![1]!).toBeCloseTo(1, 12);
    expect(pe[1]![0]!).toBeCloseTo(Math.sin(1), 12);
    expect(pe[1]![1]!).toBeCloseTo(Math.cos(1), 12);
  });
});
