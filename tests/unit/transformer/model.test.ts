import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  forward,
  forwardTyped,
  unpackWeights,
  type ModelConfig,
} from "@/lib/transformer/model";
import { emptyBlockTrace, type ForwardTrace } from "@/lib/transformer/trace";

type FwdFix = {
  config: Record<string, number>;
  input_text: string;
  token_ids: number[];
  tok_emb: number[][];
  pos_emb: number[][];
  x_after_emb: number[][];
  x_final: number[][];
  logits: number[][];
  weights: Record<string, number[][] | number[]>;
};

const fixture = JSON.parse(
  readFileSync(join("tests", "unit", "fixtures", "forward.json"), "utf-8"),
) as FwdFix;

function maxAbsErr2D(a: number[][], b: number[][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    for (let j = 0; j < ai.length; j++) {
      const e = Math.abs((ai[j] ?? 0) - (bi[j] ?? 0));
      if (e > m) m = e;
    }
  }
  return m;
}

describe("forward", () => {
  it("matches the PyTorch fixture to ≤ 1e-5 on every output", () => {
    const got = forward(fixture.token_ids, fixture.config, fixture.weights);
    expect(maxAbsErr2D(got.tokEmb, fixture.tok_emb)).toBeLessThan(1e-5);
    expect(maxAbsErr2D(got.posEmb, fixture.pos_emb)).toBeLessThan(1e-5);
    expect(maxAbsErr2D(got.xFinal, fixture.x_final)).toBeLessThan(1e-5);
    expect(maxAbsErr2D(got.logits, fixture.logits)).toBeLessThan(1e-5);
  });

  it("populates a ForwardTrace identically to the bare forward", () => {
    const config: ModelConfig = {
      seq_len: fixture.config["seq_len"]!,
      d_model: fixture.config["d_model"]!,
      n_heads: fixture.config["n_heads"]!,
      d_ff: fixture.config["d_ff"]!,
      n_blocks: fixture.config["n_blocks"]!,
      vocab_size: fixture.config["vocab_size"]!,
      seed: fixture.config["seed"]!,
    };
    const w = unpackWeights(fixture.weights, config);
    const trace: ForwardTrace = {
      tokenIds: [],
      tokEmb: [],
      posEmb: [],
      blocks: [],
      xFinal: [],
      logits: [],
    };
    const got = forwardTyped(fixture.token_ids, config, w, trace);
    expect(trace.tokenIds).toEqual(fixture.token_ids);
    expect(trace.blocks.length).toBe(config.n_blocks);
    expect(trace.xFinal).toEqual(got.xFinal);
    expect(trace.logits).toEqual(got.logits);
    // Each block's trace has all sublayer fields.
    for (const bt of trace.blocks) {
      expect(bt.ln1.length).toBe(config.seq_len);
      expect(bt.ffnOut.length).toBe(config.seq_len);
    }
  });
});

describe("unpackWeights", () => {
  it("throws on a missing 2-D weight", () => {
    const cfg: ModelConfig = {
      seq_len: 1,
      d_model: 1,
      n_heads: 1,
      d_ff: 1,
      n_blocks: 1,
      vocab_size: 1,
      seed: 1,
    };
    const partial: Record<string, number[][] | number[]> = {};
    expect(() => unpackWeights(partial, cfg)).toThrow(/missing or non-(1|2)D/);
  });

  it("throws when a 2-D field is actually 1-D", () => {
    // Provide all 1-D vectors but pass a 1-D vector where a 2-D matrix is
    // expected (tok_emb).
    const cfg: ModelConfig = {
      seq_len: 1,
      d_model: 1,
      n_heads: 1,
      d_ff: 1,
      n_blocks: 0, // skip block-key checks; we only care about tok_emb here
      vocab_size: 1,
      seed: 1,
    };
    const flat: Record<string, number[][] | number[]> = {
      tok_emb: [1], // wrong: should be 2-D
      "ln_final.gamma": [1],
      "ln_final.beta": [1],
    };
    expect(() => unpackWeights(flat, cfg)).toThrow(/missing or non-2D/);
  });

  it("throws when a 1-D field is actually 2-D", () => {
    // Provide all 2-D matrices but pass a 2-D array where a 1-D vector is
    // expected (ln1.gamma).
    const cfg: ModelConfig = {
      seq_len: 1,
      d_model: 1,
      n_heads: 1,
      d_ff: 1,
      n_blocks: 1,
      vocab_size: 1,
      seed: 1,
    };
    const flat: Record<string, number[][] | number[]> = {
      tok_emb: [[1]],
      "block_0.attn.W_q": [[1]],
      "block_0.attn.W_k": [[1]],
      "block_0.attn.W_v": [[1]],
      "block_0.attn.W_o": [[1]],
      "block_0.ffn.W1": [[1]],
      "block_0.ffn.W2": [[1]],
      "block_0.ln1.gamma": [[1]], // wrong: should be 1-D
      "block_0.ln1.beta": [1],
      "block_0.ln2.gamma": [1],
      "block_0.ln2.beta": [1],
      "block_0.ffn.b1": [1],
      "block_0.ffn.b2": [1],
      "ln_final.gamma": [1],
      "ln_final.beta": [1],
    };
    expect(() => unpackWeights(flat, cfg)).toThrow(/missing or non-1D/);
  });
});

// emptyBlockTrace is exercised here so it ends up in coverage even though
// it's primarily consumed by model.ts internally.
describe("emptyBlockTrace", () => {
  it("returns an object with all the expected keys", () => {
    const t = emptyBlockTrace();
    expect(Object.keys(t).sort()).toEqual(
      ["attn", "attnOut", "ffnAct", "ffnOut", "ffnPre", "ln1", "ln2"].sort(),
    );
  });
});
