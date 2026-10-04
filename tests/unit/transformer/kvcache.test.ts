/**
 * The KV-cache equivalence proof (brief 07 §2): cached generation must give
 * the *identical* logits to recomputing the whole sequence at every step,
 * at a fraction of the FLOPs.
 */
import { describe, expect, it } from "vitest";

import { initModelWeights } from "@/lib/transformer/init";
import {
  cacheBytes,
  decodeStep,
  decodeStepFlops,
  emptyCache,
  prefill,
  prefillFlops,
  type FlopCounter,
} from "@/lib/transformer/kvcache";
import { forwardTyped, type ModelConfig } from "@/lib/transformer/model";
import { argmax } from "@/lib/transformer/sampling";
import { encode } from "@/lib/transformer/tokenizer";
import { emptyBlockTrace, type ForwardTrace } from "@/lib/transformer/trace";

const CONFIGS: ModelConfig[] = [
  // The site's live demo model.
  {
    seq_len: 24,
    d_model: 16,
    n_heads: 2,
    d_ff: 32,
    n_blocks: 2,
    vocab_size: 64,
    seed: 42,
  },
  // Wider, more heads and blocks, another seed.
  {
    seq_len: 20,
    d_model: 32,
    n_heads: 4,
    d_ff: 64,
    n_blocks: 3,
    vocab_size: 64,
    seed: 7,
  },
];

/** Uncached reference: run the full forward pass over `ids`; last row. */
function recompute(ids: number[], c: ModelConfig): number[] {
  const w = initModelWeights(c);
  const { logits } = forwardTyped(ids, { ...c, seq_len: ids.length }, w);
  return logits[ids.length - 1]!;
}

describe("KV cache", () => {
  for (const c of CONFIGS) {
    it(`cached greedy generation is bit-identical to recomputation (D=${c.d_model}, H=${c.n_heads}, L=${c.n_blocks})`, () => {
      const w = initModelWeights(c);
      const prompt = encode("the cat", 7);
      const { logits: pre, cache } = prefill(prompt, c, w);

      // Prefill's last row = recomputation over the prompt.
      expect(pre[prompt.length - 1]).toEqual(recompute(prompt, c));

      const ids = prompt.slice();
      let next = argmax(pre[prompt.length - 1]!);
      for (let step = 0; step < 12; step++) {
        ids.push(next);
        const cached = decodeStep(next, cache, c, w);
        const full = recompute(ids, c);
        // toEqual on number[] is exact (Object.is per element): no tolerance.
        expect(cached).toEqual(full);
        next = argmax(cached);
      }
      expect(cache.length).toBe(ids.length);
    });
  }

  it("prefill matches the vendored forward pass row for row and caches its K and V", () => {
    const c = CONFIGS[0]!;
    const w = initModelWeights(c);
    const ids = encode("hello", 5);
    const trace = {
      blocks: [] as ReturnType<typeof emptyBlockTrace>[],
    } as ForwardTrace;
    const ref = forwardTyped(ids, { ...c, seq_len: 5 }, w, trace);
    const { logits, cache } = prefill(ids, c, w);
    expect(logits).toEqual(ref.logits);
    for (let b = 0; b < c.n_blocks; b++) {
      expect(cache.layers[b]!.K).toEqual(trace.blocks[b]!.attn.K);
      expect(cache.layers[b]!.V).toEqual(trace.blocks[b]!.attn.V);
    }
  });

  it("decoding from an empty cache equals prefill, token by token", () => {
    const c = CONFIGS[1]!;
    const w = initModelWeights(c);
    const ids = encode("kv cache", 8);
    const cache = emptyCache(c.n_blocks);
    const rows = ids.map((id) => decodeStep(id, cache, c, w));
    expect(rows).toEqual(prefill(ids, c, w).logits);
  });

  it("counts FLOPs as the closed forms say", () => {
    const c = CONFIGS[0]!;
    const w = initModelWeights(c);
    const ids = encode("abc", 3);
    const pc: FlopCounter = { flops: 0 };
    const { cache } = prefill(ids, c, w, pc);
    expect(pc.flops).toBe(prefillFlops(3, c));
    for (let t = 3; t < 10; t++) {
      const dc: FlopCounter = { flops: 0 };
      decodeStep(5, cache, c, w, dc);
      // The instrumented matmuls agree with the closed form exactly.
      expect(dc.flops).toBe(decodeStepFlops(t, c));
    }
  });

  it("does a fraction of the uncached FLOPs over a generation", () => {
    const c = CONFIGS[0]!;
    const P = 4;
    const N = 20;
    let cached = prefillFlops(P, c);
    let uncached = prefillFlops(P, c);
    for (let t = P; t < P + N - 1; t++) {
      cached += decodeStepFlops(t, c);
      uncached += prefillFlops(t + 1, c);
    }
    // Cached work grows ~linearly with length, uncached ~quadratically.
    expect(cached / uncached).toBeLessThan(0.15);
    // Each later cached step is cheaper than recomputing by roughly the
    // sequence length.
    expect(prefillFlops(23, c) / decodeStepFlops(22, c)).toBeGreaterThan(20);
  });

  it("holds 2·L·S·D values", () => {
    const c = CONFIGS[1]!;
    const w = initModelWeights(c);
    const { cache } = prefill(encode("abcd", 4), c, w);
    expect(cacheBytes(cache, 2)).toBe(2 * c.n_blocks * 4 * c.d_model * 2);
  });

  it("rejects an empty prompt", () => {
    const c = CONFIGS[0]!;
    expect(() => prefill([], c, initModelWeights(c))).toThrow(/prompt/);
  });
});
