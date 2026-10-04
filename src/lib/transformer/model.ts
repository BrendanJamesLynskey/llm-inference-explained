/*
 * src/lib/transformer/model.ts
 *
 * Operation: full forward pass — embed → N decoder blocks → final LN →
 *            tied-head projection to logits.
 * Shapes:    forward(tokenIds: [S], config, weights) →
 *              { tokEmb: [S, D], posEmb: [S, D], xFinal: [S, D], logits: [S, V] }
 * Intuition: same as a tiny GPT. The output head is *tied* to the token
 *            embedding (logits = xFinal · Eᵀ) — common in small models and
 *            mirrors `scripts/reference.py`.
 * MDX:       /learn/decoder/06-stacking, /learn/decoder/07-sampling.
 */
import { block, type BlockWeights } from "./block";
import { positionalEncoding, tokenEmbedding } from "./embeddings";
import { layernormRows } from "./layernorm";
import { matmul, transpose } from "./matmul";
import { addMat } from "./tensor";
import { emptyBlockTrace, type ForwardTrace } from "./trace";
import type { Matrix, Vector } from "./types";

/** Toy-model hyperparameters. Matches `scripts/reference.py:DEFAULT_CONFIG`. */
export type ModelConfig = {
  seq_len: number;
  d_model: number;
  n_heads: number;
  d_ff: number;
  n_blocks: number;
  vocab_size: number;
  seed: number;
};

/** Typed model weights, ready to pass to `forwardTyped`. */
export type ModelWeights = {
  tok_emb: Matrix; // [V, D]
  blocks: BlockWeights[];
  ln_final: { gamma: Vector; beta: Vector };
};

/** Result of a forward pass — keeps the names the verify script expects. */
export type ForwardResult = {
  tokEmb: Matrix;
  posEmb: Matrix;
  xFinal: Matrix;
  logits: Matrix;
};

/**
 * Unpack the flat `{ "block_0.attn.W_q": [...], ... }` weight dict produced
 * by `scripts/reference.py` into a typed `ModelWeights`. Throws clearly if
 * a required key is missing.
 */
export function unpackWeights(
  flat: Record<string, Matrix | Vector>,
  config: ModelConfig,
): ModelWeights {
  function get2D(key: string): Matrix {
    const v = flat[key];
    if (!v || !Array.isArray(v[0])) {
      throw new Error(`unpackWeights: missing or non-2D ${key}`);
    }
    return v as Matrix;
  }
  function get1D(key: string): Vector {
    const v = flat[key];
    if (!v || Array.isArray(v[0])) {
      throw new Error(`unpackWeights: missing or non-1D ${key}`);
    }
    return v as Vector;
  }

  const blocks: BlockWeights[] = [];
  for (let i = 0; i < config.n_blocks; i++) {
    const p = `block_${i}`;
    blocks.push({
      ln1: { gamma: get1D(`${p}.ln1.gamma`), beta: get1D(`${p}.ln1.beta`) },
      attn: {
        W_q: get2D(`${p}.attn.W_q`),
        W_k: get2D(`${p}.attn.W_k`),
        W_v: get2D(`${p}.attn.W_v`),
        W_o: get2D(`${p}.attn.W_o`),
      },
      ln2: { gamma: get1D(`${p}.ln2.gamma`), beta: get1D(`${p}.ln2.beta`) },
      ffn: {
        W1: get2D(`${p}.ffn.W1`),
        b1: get1D(`${p}.ffn.b1`),
        W2: get2D(`${p}.ffn.W2`),
        b2: get1D(`${p}.ffn.b2`),
      },
    });
  }
  return {
    tok_emb: get2D("tok_emb"),
    blocks,
    ln_final: {
      gamma: get1D("ln_final.gamma"),
      beta: get1D("ln_final.beta"),
    },
  };
}

/**
 * Forward pass with already-typed weights and config. Optionally records
 * intermediates into `trace`.
 */
export function forwardTyped(
  tokenIds: number[],
  config: ModelConfig,
  weights: ModelWeights,
  trace?: ForwardTrace,
): ForwardResult {
  const tokEmb = tokenEmbedding(tokenIds, weights.tok_emb);
  const posEmb = positionalEncoding(config.seq_len, config.d_model);
  let x = addMat(tokEmb, posEmb);

  if (trace) {
    trace.tokenIds = tokenIds.slice();
    trace.tokEmb = tokEmb;
    trace.posEmb = posEmb;
    trace.blocks = [];
  }

  for (let i = 0; i < config.n_blocks; i++) {
    const bt = trace ? emptyBlockTrace() : undefined;
    x = block(x, weights.blocks[i]!, config.n_heads, bt);
    if (trace && bt) trace.blocks.push(bt);
  }

  const xFinal = layernormRows(
    x,
    weights.ln_final.gamma,
    weights.ln_final.beta,
  );

  // Tied output head: logits = xFinal · tok_embᵀ → shape [S, V].
  const logits = matmul(xFinal, transpose(weights.tok_emb));

  if (trace) {
    trace.xFinal = xFinal;
    trace.logits = logits;
  }
  return { tokEmb, posEmb, xFinal, logits };
}

/**
 * Forward pass that takes the *flat* weight dict produced by
 * `scripts/reference.py` (and the JSON fixtures). Used by `verify-maths.ts`.
 */
export function forward(
  tokenIds: number[],
  config: Record<string, number>,
  weights: Record<string, Matrix | Vector>,
): ForwardResult {
  const cfg: ModelConfig = {
    seq_len: config["seq_len"]!,
    d_model: config["d_model"]!,
    n_heads: config["n_heads"]!,
    d_ff: config["d_ff"]!,
    n_blocks: config["n_blocks"]!,
    vocab_size: config["vocab_size"]!,
    seed: config["seed"]!,
  };
  const w = unpackWeights(weights, cfg);
  return forwardTyped(tokenIds, cfg, w);
}
