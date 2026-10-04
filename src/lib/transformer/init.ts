/*
 * src/lib/transformer/init.ts
 *
 * Operation: deterministic weight initialisation from a seed.
 * Shapes:    initModelWeights(config) → ModelWeights with the shapes
 *            described in `model.ts`.
 * Intuition: GPT-style normal(0, 0.02) init for matrices; zero biases;
 *            γ=1, β=0 for layernorm. Order of draws matters: changing it
 *            invalidates any cached visualisations.
 * MDX:       /learn/decoder/06-stacking (seed-driven hyperparameter panel).
 */
import type { ModelConfig, ModelWeights } from "./model";
import { mulberry32, normalSampler } from "./random";
import type { Matrix, Vector } from "./types";

const INIT_STD = 0.02;

function makeNormal(rng: () => number): (rows: number, cols: number) => Matrix {
  const sample = normalSampler(rng);
  return (rows: number, cols: number) => {
    const out: Matrix = new Array<Vector>(rows);
    for (let i = 0; i < rows; i++) {
      const row = new Array<number>(cols);
      for (let j = 0; j < cols; j++) row[j] = sample() * INIT_STD;
      out[i] = row;
    }
    return out;
  };
}

/** Build a deterministic weight set for the given config + seed. */
export function initModelWeights(config: ModelConfig): ModelWeights {
  const rng = mulberry32(config.seed);
  const normal = makeNormal(rng);

  const tok_emb = normal(config.vocab_size, config.d_model);

  const blocks: ModelWeights["blocks"] = [];
  for (let i = 0; i < config.n_blocks; i++) {
    blocks.push({
      ln1: {
        gamma: ones(config.d_model),
        beta: zerosVec(config.d_model),
      },
      attn: {
        W_q: normal(config.d_model, config.d_model),
        W_k: normal(config.d_model, config.d_model),
        W_v: normal(config.d_model, config.d_model),
        W_o: normal(config.d_model, config.d_model),
      },
      ln2: {
        gamma: ones(config.d_model),
        beta: zerosVec(config.d_model),
      },
      ffn: {
        W1: normal(config.d_model, config.d_ff),
        b1: zerosVec(config.d_ff),
        W2: normal(config.d_ff, config.d_model),
        b2: zerosVec(config.d_model),
      },
    });
  }

  return {
    tok_emb,
    blocks,
    ln_final: {
      gamma: ones(config.d_model),
      beta: zerosVec(config.d_model),
    },
  };
}

function ones(n: number): Vector {
  return new Array<number>(n).fill(1);
}

function zerosVec(n: number): Vector {
  return new Array<number>(n).fill(0);
}
