import { describe, expect, it } from "vitest";

import {
  causalMask,
  multiHeadAttention,
  singleHeadAttention,
} from "@/lib/transformer/attention";
import { emptyAttentionTrace } from "@/lib/transformer/trace";

describe("causalMask", () => {
  it("is 0 on and below the diagonal, -Infinity above it", () => {
    expect(causalMask(3)).toEqual([
      [0, -Infinity, -Infinity],
      [0, 0, -Infinity],
      [0, 0, 0],
    ]);
  });

  it("is empty for seqLen=0", () => {
    expect(causalMask(0)).toEqual([]);
  });
});

describe("singleHeadAttention", () => {
  it("each query attends only to itself or earlier keys", () => {
    // 3 positions, d_head=2. Set Q=K=V to a simple basis so weights are easy.
    const I = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const { weights } = singleHeadAttention(I, I, I);
    // Row i should have all zeros at j > i.
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        expect(weights[i]![j]!).toBe(0);
      }
    }
  });

  it("weights sum to 1 per row", () => {
    const Q = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const { weights } = singleHeadAttention(Q, Q, Q);
    for (const row of weights) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-12);
    }
  });
});

describe("multiHeadAttention", () => {
  it("throws when n_heads doesn't divide d_model", () => {
    const x = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const W = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    expect(() =>
      multiHeadAttention(x, { W_q: W, W_k: W, W_v: W, W_o: W }, 2),
    ).toThrow(/n_heads.*divide d_model/);
  });

  it("populates the trace when given one", () => {
    const x = [
      [1, 0],
      [0, 1],
    ];
    const I = [
      [1, 0],
      [0, 1],
    ];
    const trace = emptyAttentionTrace();
    multiHeadAttention(x, { W_q: I, W_k: I, W_v: I, W_o: I }, 2, trace);
    expect(trace.Q.length).toBe(2);
    expect(trace.scores.length).toBe(2); // one per head
    expect(trace.weights.length).toBe(2);
    // Each head's scores/weights are [seq_len, seq_len].
    expect(trace.weights[0]!.length).toBe(2);
    expect(trace.weights[0]![0]!.length).toBe(2);
  });
});
