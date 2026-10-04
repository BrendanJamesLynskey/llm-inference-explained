import { describe, expect, it } from "vitest";

import { block, type BlockWeights } from "@/lib/transformer/block";
import { emptyBlockTrace } from "@/lib/transformer/trace";

const D = 4;
const D_FF = 8;
const SEQ = 3;
const N_HEADS = 2;

function eye(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}

function zerosMat(r: number, c: number): number[][] {
  return Array.from({ length: r }, () => new Array<number>(c).fill(0));
}

const wIdentity: BlockWeights = {
  ln1: {
    gamma: new Array<number>(D).fill(1),
    beta: new Array<number>(D).fill(0),
  },
  attn: { W_q: eye(D), W_k: eye(D), W_v: eye(D), W_o: eye(D) },
  ln2: {
    gamma: new Array<number>(D).fill(1),
    beta: new Array<number>(D).fill(0),
  },
  ffn: {
    W1: zerosMat(D, D_FF),
    b1: new Array<number>(D_FF).fill(0),
    W2: zerosMat(D_FF, D),
    b2: new Array<number>(D).fill(0),
  },
};

describe("block", () => {
  it("returns a [S, D] matrix", () => {
    const x = Array.from({ length: SEQ }, (_, i) =>
      Array.from({ length: D }, (_, j) => i + j * 0.1),
    );
    const out = block(x, wIdentity, N_HEADS);
    expect(out.length).toBe(SEQ);
    expect(out[0]!.length).toBe(D);
  });

  it("produces no NaNs/Infinites", () => {
    const x = Array.from({ length: SEQ }, (_, i) =>
      Array.from({ length: D }, (_, j) => i - 1 + j),
    );
    const out = block(x, wIdentity, N_HEADS);
    for (const row of out)
      for (const v of row) expect(Number.isFinite(v)).toBe(true);
  });

  it("populates a BlockTrace when given one", () => {
    const x = Array.from({ length: SEQ }, (_, i) =>
      Array.from({ length: D }, (_, j) => i + j),
    );
    const trace = emptyBlockTrace();
    block(x, wIdentity, N_HEADS, trace);
    expect(trace.ln1.length).toBe(SEQ);
    expect(trace.attn.scores.length).toBe(N_HEADS);
    expect(trace.ln2.length).toBe(SEQ);
    expect(trace.ffnOut.length).toBe(SEQ);
  });
});
