import { describe, expect, it } from "vitest";

import { gelu } from "@/lib/transformer/gelu";
import { ffn } from "@/lib/transformer/ffn";

describe("ffn", () => {
  // 1-input, 2-hidden, 1-output network so we can hand-compute the result.
  const w = {
    W1: [[1, -1]],
    b1: [0.5, -0.5],
    W2: [[2], [-2]],
    b2: [0.25],
  };

  it("matches a hand-computed forward pass", () => {
    // x = [[3]]; pre = [[3*1+0.5, 3*-1-0.5]] = [[3.5, -3.5]]
    // act = [[gelu(3.5), gelu(-3.5)]]
    // out = [[2*gelu(3.5) + -2*gelu(-3.5) + 0.25]]
    const out = ffn([[3]], w);
    const expected = 2 * gelu(3.5) + -2 * gelu(-3.5) + 0.25;
    expect(out[0]![0]!).toBeCloseTo(expected, 10);
  });

  it("populates the trace when given one", () => {
    const trace = {
      pre: [] as number[][],
      act: [] as number[][],
      out: [] as number[][],
    };
    ffn([[1]], w, trace);
    expect(trace.pre.length).toBe(1);
    expect(trace.pre[0]!.length).toBe(2);
    expect(trace.act.length).toBe(1);
    expect(trace.out.length).toBe(1);
    expect(trace.out[0]!.length).toBe(1);
  });

  it("works for batched inputs (multiple rows)", () => {
    const out = ffn([[1], [2], [3]], w);
    expect(out.length).toBe(3);
    expect(out[0]!.length).toBe(1);
  });
});
