import { describe, expect, it } from "vitest";

import { initModelWeights } from "@/lib/transformer/init";

const baseConfig = {
  seq_len: 8,
  d_model: 16,
  n_heads: 2,
  d_ff: 32,
  n_blocks: 2,
  vocab_size: 64,
  seed: 42,
};

describe("initModelWeights", () => {
  it("produces the right shapes", () => {
    const w = initModelWeights(baseConfig);
    expect(w.tok_emb.length).toBe(baseConfig.vocab_size);
    expect(w.tok_emb[0]?.length).toBe(baseConfig.d_model);
    expect(w.blocks.length).toBe(baseConfig.n_blocks);
    expect(w.ln_final.gamma.length).toBe(baseConfig.d_model);
    expect(w.ln_final.beta.length).toBe(baseConfig.d_model);

    for (const block of w.blocks) {
      expect(block.ln1.gamma).toEqual(new Array(baseConfig.d_model).fill(1));
      expect(block.ln1.beta).toEqual(new Array(baseConfig.d_model).fill(0));
      expect(block.attn.W_q.length).toBe(baseConfig.d_model);
      expect(block.attn.W_q[0]?.length).toBe(baseConfig.d_model);
      expect(block.ffn.W1.length).toBe(baseConfig.d_model);
      expect(block.ffn.W1[0]?.length).toBe(baseConfig.d_ff);
      expect(block.ffn.b1.length).toBe(baseConfig.d_ff);
      expect(block.ffn.W2.length).toBe(baseConfig.d_ff);
      expect(block.ffn.W2[0]?.length).toBe(baseConfig.d_model);
      expect(block.ffn.b2.length).toBe(baseConfig.d_model);
    }
  });

  it("is deterministic for a given seed", () => {
    const a = initModelWeights(baseConfig);
    const b = initModelWeights(baseConfig);
    expect(a.tok_emb).toEqual(b.tok_emb);
    expect(a.blocks[0]?.attn.W_q).toEqual(b.blocks[0]?.attn.W_q);
  });

  it("differs across seeds", () => {
    const a = initModelWeights(baseConfig);
    const b = initModelWeights({ ...baseConfig, seed: 99 });
    expect(a.tok_emb).not.toEqual(b.tok_emb);
  });

  it("supports zero blocks", () => {
    const w = initModelWeights({ ...baseConfig, n_blocks: 0 });
    expect(w.blocks).toEqual([]);
  });
});
