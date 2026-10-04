/*
 * src/lib/transformer/kvcache.ts
 *
 * Operation: incremental (cached) decoding — prefill a prompt once, then
 *            generate one token per step reusing every earlier position's
 *            keys and values instead of recomputing them.
 * Shapes:    prefill(tokenIds: [S], …) → { logits: [S, V], cache }
 *            decodeStep(tokenId, cache, …) → logits: [V]   (cache grows by 1)
 *            cache.layers[b].K / .V: [S_cached, D] (all heads side by side)
 * Intuition: in a causal decoder, position i's keys and values depend only
 *            on tokens 0..i. Appending a token never changes them, so they
 *            can be kept ("the KV cache") and only the new row computed.
 *            The new row's logits are *identical* to a full recomputation:
 *            tests/unit/transformer/kvcache.test.ts checks it bit for bit.
 * MDX:       /learn/01-generation-loop, /learn/02-kv-cache.
 *
 * Added in llm-inference-explained; everything else in this directory is
 * vendored unchanged from transformer-explainer (see README "Credits").
 */
import { block } from "./block";
import { positionalEncoding, tokenEmbedding } from "./embeddings";
import { ffn } from "./ffn";
import { layernormRows } from "./layernorm";
import type { ModelConfig, ModelWeights } from "./model";
import { matmul, transpose } from "./matmul";
import { softmax } from "./softmax";
import { addMat, addVec } from "./tensor";
import { emptyBlockTrace } from "./trace";
import type { Matrix, Vector } from "./types";

/** Keys and values of one block, one row per cached position. */
export type LayerKV = {
  K: Matrix; // [S_cached, D]
  V: Matrix; // [S_cached, D]
};

/** The KV cache: one `LayerKV` per decoder block. */
export type KVCache = {
  layers: LayerKV[];
  /** Number of positions cached (the same in every layer). */
  length: number;
};

/**
 * A running FLOP tally. Counts matrix-multiply FLOPs only (2 per
 * multiply-add, the usual convention); LayerNorm, softmax and GELU are
 * O(S·D) and ignored, as in most inference cost models.
 */
export type FlopCounter = { flops: number };

/** An empty cache for a model with `nBlocks` blocks. */
export function emptyCache(nBlocks: number): KVCache {
  const layers: LayerKV[] = [];
  for (let i = 0; i < nBlocks; i++) layers.push({ K: [], V: [] });
  return { layers, length: 0 };
}

/** Bytes the cache holds at `bytesPerValue` bytes per number: 2·L·S·D·b. */
export function cacheBytes(cache: KVCache, bytesPerValue: number): number {
  let values = 0;
  for (const l of cache.layers) {
    for (const row of l.K) values += row.length;
    for (const row of l.V) values += row.length;
  }
  return values * bytesPerValue;
}

/** `matmul` that also adds its 2·M·K·N FLOPs to `counter`. */
function mm(A: Matrix, B: Matrix, counter?: FlopCounter): Matrix {
  if (counter) {
    counter.flops += 2 * A.length * (A[0]?.length ?? 0) * (B[0]?.length ?? 0);
  }
  return matmul(A, B);
}

/**
 * Closed-form FLOPs of a full (uncached) forward pass over `S` tokens —
 * what recomputing everything costs at every generation step.
 *
 * Per block: Q, K, V and output projections 4·(2·S·D·D); scores Q·Kᵀ and
 * weights·V 2·(2·S·S·D) summed over heads; FFN 2·(2·S·D·D_ff). Plus the
 * tied output head 2·S·D·V. Dense counts: the masked half of the S×S
 * score matrix is still computed by the plain forward pass.
 */
export function prefillFlops(S: number, c: ModelConfig): number {
  const { d_model: D, d_ff: F, vocab_size: V, n_blocks: L } = c;
  const perBlock = 8 * S * D * D + 4 * S * S * D + 4 * S * D * F;
  return L * perBlock + 2 * S * D * V;
}

/**
 * Closed-form FLOPs of one cached decode step when the cache already holds
 * `ctx` positions (so the new token attends to `ctx + 1` positions).
 */
export function decodeStepFlops(ctx: number, c: ModelConfig): number {
  const { d_model: D, d_ff: F, vocab_size: V, n_blocks: L } = c;
  const n = ctx + 1;
  const perBlock = 8 * D * D + 4 * n * D + 4 * D * F;
  return L * perBlock + 2 * D * V;
}

/**
 * Prefill: run the whole prompt through the model in one pass (exactly the
 * vendored `forwardTyped` computation) and keep every block's K and V.
 *
 * @param tokenIds  Prompt token ids `[S]`, S ≥ 1.
 * @param config    Model hyperparameters; `seq_len` is ignored (the prompt
 *                  length is used).
 * @param weights   Model weights.
 * @param counter   Optional FLOP tally (closed form: `prefillFlops`).
 * @returns         Logits for every prompt position `[S, V]`, and the cache.
 */
export function prefill(
  tokenIds: number[],
  config: ModelConfig,
  weights: ModelWeights,
  counter?: FlopCounter,
): { logits: Matrix; cache: KVCache } {
  const S = tokenIds.length;
  if (S === 0) throw new Error("prefill: the prompt must hold a token");
  const cache = emptyCache(config.n_blocks);
  let x = addMat(
    tokenEmbedding(tokenIds, weights.tok_emb),
    positionalEncoding(S, config.d_model),
  );
  for (let b = 0; b < config.n_blocks; b++) {
    const bt = emptyBlockTrace();
    x = block(x, weights.blocks[b]!, config.n_heads, bt);
    cache.layers[b] = { K: bt.attn.K, V: bt.attn.V };
  }
  cache.length = S;
  const xFinal = layernormRows(
    x,
    weights.ln_final.gamma,
    weights.ln_final.beta,
  );
  const logits = matmul(xFinal, transpose(weights.tok_emb));
  if (counter) counter.flops += prefillFlops(S, config);
  return { logits, cache };
}

/**
 * One cached decode step: compute the new token's row only, append its K
 * and V to every layer, and attend over the whole cache.
 *
 * The operations and their order match the vendored full forward pass row
 * for row (same matmul loop order, same softmax; the masked future
 * positions the full pass carries contribute exact zeros), so the logits
 * are bit-identical, not merely close.
 *
 * @param tokenId  The token just generated (or the next prompt token).
 * @param cache    The cache; mutated (one row appended per layer).
 * @param config   Model hyperparameters.
 * @param weights  Model weights.
 * @param counter  Optional FLOP tally (closed form: `decodeStepFlops`).
 * @returns        Logits for the next token `[V]`.
 */
export function decodeStep(
  tokenId: number,
  cache: KVCache,
  config: ModelConfig,
  weights: ModelWeights,
  counter?: FlopCounter,
): Vector {
  const pos = cache.length;
  const D = config.d_model;
  const H = config.n_heads;
  const dHead = D / H;
  const scale = 1 / Math.sqrt(Math.max(1, dHead));

  // x = token embedding + sinusoidal position `pos`.
  let x: Vector = addVec(
    tokenEmbedding([tokenId], weights.tok_emb)[0]!,
    positionalEncoding(pos + 1, D)[pos]!,
  );

  for (let b = 0; b < config.n_blocks; b++) {
    const w = weights.blocks[b]!;
    const layer = cache.layers[b]!;

    // Sub-layer 1: x + Attn(LN1(x)), for the new row only.
    const ln1 = layernormRows([x], w.ln1.gamma, w.ln1.beta);
    const q = mm(ln1, w.attn.W_q, counter)[0]!;
    const k = mm(ln1, w.attn.W_k, counter)[0]!;
    const v = mm(ln1, w.attn.W_v, counter)[0]!;
    layer.K.push(k);
    layer.V.push(v);

    const concat: Vector = [];
    for (let h = 0; h < H; h++) {
      const lo = h * dHead;
      const qh = [q.slice(lo, lo + dHead)];
      const Kh = layer.K.map((row) => row.slice(lo, lo + dHead));
      const Vh = layer.V.map((row) => row.slice(lo, lo + dHead));
      // scores = q · Kᵀ / √d_head over every cached position (0..pos).
      const raw = mm(qh, transpose(Kh), counter)[0]!;
      const scores = raw.map((s) => s * scale);
      const weightsRow = softmax(scores);
      concat.push(...mm([weightsRow], Vh, counter)[0]!);
    }
    const attnOut = mm([concat], w.attn.W_o, counter)[0]!;
    const hRow = addVec(x, attnOut);

    // Sub-layer 2: h + FFN(LN2(h)).
    const ln2 = layernormRows([hRow], w.ln2.gamma, w.ln2.beta);
    if (counter) counter.flops += 4 * D * config.d_ff;
    const ffnOut = ffn(ln2, w.ffn)[0]!;
    x = addVec(hRow, ffnOut);
  }
  cache.length = pos + 1;

  const xFinal = layernormRows(
    [x],
    weights.ln_final.gamma,
    weights.ln_final.beta,
  );
  return mm(xFinal, transpose(weights.tok_emb), counter)[0]!;
}
