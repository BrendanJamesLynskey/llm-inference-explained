import { describe, expect, it } from "vitest";

import {
  emptyAttentionTrace,
  emptyBlockTrace,
  head,
} from "@/lib/transformer/trace";

describe("emptyAttentionTrace", () => {
  it("returns an object with all the expected keys", () => {
    const t = emptyAttentionTrace();
    expect(Object.keys(t).sort()).toEqual(["K", "Q", "V", "scores", "weights"]);
  });
});

describe("emptyBlockTrace", () => {
  it("nests an empty AttentionTrace inside", () => {
    const t = emptyBlockTrace();
    expect(t.attn.scores).toEqual([]);
  });
});

describe("head", () => {
  it("returns the first n elements (default 4)", () => {
    expect(head([1, 2, 3, 4, 5])).toEqual([1, 2, 3, 4]);
  });

  it("respects an explicit n", () => {
    expect(head([1, 2, 3, 4, 5], 2)).toEqual([1, 2]);
  });

  it("returns the full vector if n exceeds length", () => {
    expect(head([1, 2], 10)).toEqual([1, 2]);
  });
});
