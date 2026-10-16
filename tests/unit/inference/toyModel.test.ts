/**
 * The toy model's two sets of weights (brief 41): the trained ones arrive
 * by `provideTrainedWeights` (fetched in the browser, imported on the
 * server) and are the vendored file; random init needs nothing.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  TOY_CONFIG,
  TRAINED_WEIGHTS_URL,
  hasTrainedWeights,
  provideTrainedWeights,
  toyWeights,
  type TrainedWeightsFile,
} from "@/lib/inference/toyModel";

const file = JSON.parse(
  readFileSync(join(process.cwd(), "public", TRAINED_WEIGHTS_URL), "utf-8"),
) as TrainedWeightsFile & { config: Record<string, number> };

describe("toy weights", () => {
  it("refuse to run trained before the weights arrive", () => {
    expect(hasTrainedWeights()).toBe(false);
    expect(() => toyWeights("trained")).toThrow(/not loaded/);
    expect(toyWeights()).toBe(toyWeights("random"));
  });
  it("are the vendored trained file once provided (a second file is ignored)", () => {
    provideTrainedWeights(file);
    provideTrainedWeights({ params: {}, steps: 0 });
    expect(hasTrainedWeights()).toBe(true);
    const w = toyWeights("trained");
    expect(w).toBe(toyWeights("trained"));
    expect(w.tok_emb[0]).toEqual(file.params["tok_emb"]![0]);
    expect(w.tok_emb[0]).not.toEqual(toyWeights("random").tok_emb[0]);
    // the trained model has the toy model's shapes
    for (const k of [
      "d_model",
      "n_heads",
      "d_ff",
      "n_blocks",
      "vocab_size",
    ] as const)
      expect(file.config[k]).toBe(TOY_CONFIG[k]);
    expect(file.steps).toBe(2000);
  });
});
