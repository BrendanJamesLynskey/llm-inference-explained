/*
 * src/lib/inference/heroPreview.ts
 *
 * The landing page's figure, computed on the server by the same code the
 * KV-cache chapter runs in the browser: generate with the cache, recompute
 * every step without it, and record the FLOPs of both and the largest logit
 * difference (which is 0: the cache is exact).
 */
import {
  decodeStep,
  decodeStepFlops,
  prefill,
  prefillFlops,
} from "@/lib/transformer/kvcache";
import { forwardTyped } from "@/lib/transformer/model";
import { argmax } from "@/lib/transformer/sampling";

import { TOY_CONFIG, toyEncode, toyWeights } from "./toyModel";

export type HeroPreview = {
  /** Cumulative FLOPs after each generated token. */
  cached: number[];
  uncached: number[];
  maxDiff: number;
  promptLen: number;
};

export function heroPreview(prompt = "the", tokens = 24): HeroPreview {
  const w = toyWeights();
  const ids = toyEncode(prompt);
  const { logits, cache } = prefill(ids, TOY_CONFIG, w);
  let next = argmax(logits[ids.length - 1]!);
  let c = prefillFlops(ids.length, TOY_CONFIG);
  let u = c;
  const cached = [c];
  const uncached = [u];
  let maxDiff = 0;
  for (let i = 1; i < tokens; i++) {
    ids.push(next);
    const ctx = cache.length;
    const l = decodeStep(next, cache, TOY_CONFIG, w);
    const ref = forwardTyped(ids, { ...TOY_CONFIG, seq_len: ids.length }, w)
      .logits[ids.length - 1]!;
    for (let j = 0; j < l.length; j++)
      maxDiff = Math.max(maxDiff, Math.abs(l[j]! - ref[j]!));
    c += decodeStepFlops(ctx, TOY_CONFIG);
    u += prefillFlops(ids.length, TOY_CONFIG);
    cached.push(c);
    uncached.push(u);
    next = argmax(l);
  }
  return { cached, uncached, maxDiff, promptLen: toyEncode(prompt).length };
}
