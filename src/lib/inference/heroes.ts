/*
 * src/lib/inference/heroes.ts
 *
 * Operation: the states behind the chapter heroes of brief 41, each a pure
 *            function of the site's tested models:
 *              - chapter 1, the generation timeline: one request's prefill
 *                pass then its decode steps, priced by the cost model;
 *              - chapter 3, the roofline climb: a decode step's arithmetic
 *                intensity and attainable FLOP/s as the batch grows.
 *            (Chapter 2's cache hero lives in kvHero.ts and chapter 6's tile
 *            hero in attentionKernels.ts, next to the models they replay.)
 * Shapes:    generationTimeline(opts) → TimelineStep[];
 *            rooflineClimb(opts) → RoofClimb.
 * Intuition: the same numbers the roofline chapter quotes, replayed in time
 *            order. scripts/hero_reference.py computes every state with the
 *            simulator's Python cost model (hardware.py at the commit
 *            cost_model.json names) and tests/unit/inference/heroes.test.ts
 *            requires these to equal it.
 * MDX:       /learn/01-generation-loop, /learn/03-roofline.
 */
import {
  DEVICES,
  MODELS,
  costModel,
  type StepCost,
} from "@/lib/inference/costModel";

export type ModelKey = keyof typeof MODELS;
export type DeviceKey = keyof typeof DEVICES;

const ms = (s: number): string =>
  s >= 0.1 ? `${(s * 1e3).toFixed(0)} ms` : `${(s * 1e3).toFixed(1)} ms`;

// ─── chapter 1: the generation timeline ─────────────────────────────

/** One forward pass of the request: the prefill, or one decode step. */
export type TimelineStep = {
  /** 0 = the prefill pass; k ≥ 1 = the k-th decode step. */
  k: number;
  kind: "prefill" | "decode";
  /** Start and duration, seconds. */
  t0: number;
  dt: number;
  flops: number;
  bytes: number;
  bound: StepCost["bound"];
  /** The pass's arithmetic and memory time at the derated peaks, seconds. */
  tCompute: number;
  tMemory: number;
  /** Tokens whose KV the pass reads (decode) or writes (prefill). */
  ctx: number;
  /** Output tokens produced so far, after this pass. */
  out: number;
};

export type TimelineOpts = {
  model: ModelKey;
  device: DeviceKey;
  nDevices: number;
  prompt: number;
  output: number;
};

export const TIMELINE_DEFAULT: TimelineOpts = {
  model: "llama3-8b",
  device: "h100",
  nDevices: 1,
  prompt: 2048,
  output: 16,
};

/**
 * Prefill the prompt in one pass (it yields the first output token), then
 * one decode step per further token, each reading the whole cache so far.
 */
export function generationTimeline(
  o: TimelineOpts = TIMELINE_DEFAULT,
): TimelineStep[] {
  const cm = costModel(MODELS[o.model], DEVICES[o.device], {
    nDevices: o.nDevices,
  });
  const out: TimelineStep[] = [];
  const p = cm.prefill([o.prompt]);
  out.push({
    k: 0,
    kind: "prefill",
    t0: 0,
    dt: p.time,
    flops: p.flops,
    bytes: p.bytes,
    bound: p.bound,
    tCompute: p.flops / cm.flopsRate,
    tMemory: p.bytes / cm.byteRate,
    ctx: o.prompt,
    out: 1,
  });
  let t = p.time;
  for (let k = 1; k < o.output; k++) {
    // the k-th decode step feeds output token k and attends over the prompt
    // and the k − 1 tokens already cached after it
    const ctx = o.prompt + k - 1;
    const d = cm.decode(ctx, 1);
    out.push({
      k,
      kind: "decode",
      t0: t,
      dt: d.time,
      flops: d.flops,
      bytes: d.bytes,
      bound: d.bound,
      tCompute: d.flops / cm.flopsRate,
      tMemory: d.bytes / cm.byteRate,
      ctx,
      out: k + 1,
    });
    t += d.time;
  }
  return out;
}

/** The caption for step k of a timeline. */
export function timelineCaption(
  steps: readonly TimelineStep[],
  k: number,
): string {
  const s = steps[k]!;
  const r = s.flops / s.bytes;
  const fpb = r >= 10 ? Math.round(r).toLocaleString("en-GB") : r.toFixed(1);
  if (s.kind === "prefill")
    return `Prefill: all ${s.ctx.toLocaleString("en-GB")} prompt tokens in one pass, ${ms(s.dt)}, ${s.bound}-bound (${fpb} FLOP per byte). It writes their keys and values and yields the first token: time to first token ${ms(s.t0 + s.dt)}.`;
  const t = s.t0 + s.dt;
  return `Decode step ${s.k}: one new token, reading the weights and ${s.ctx.toLocaleString("en-GB")} cached tokens, ${ms(s.dt)}, ${s.bound}-bound (${fpb} FLOP per byte). ${s.out} tokens out at ${ms(t)}.`;
}

// ─── chapter 3: the decode point climbing the roofline ──────────────

export type RoofPoint = {
  batch: number;
  /** Each sequence's context; the step reads batch × ctx cached tokens. */
  ctx: number;
  flops: number;
  bytes: number;
  time: number;
  bound: StepCost["bound"];
  intensity: number;
  /** The roof at this intensity: min(peak, intensity × bandwidth), FLOP/s. */
  attainable: number;
  /** Output tokens per second of the step (batch / time). */
  tokensPerS: number;
};

export type RoofClimb = {
  /** Derated peak FLOP/s and bytes/s of the instance. */
  peak: number;
  bandwidth: number;
  /** FLOP per byte where the two roofs meet. */
  ridge: number;
  points: RoofPoint[];
  /** A prefill pass of `ctx` tokens, for contrast. */
  prefill: RoofPoint;
};

export type RoofOpts = {
  model: ModelKey;
  device: DeviceKey;
  nDevices: number;
  ctx: number;
  batches: readonly number[];
};

export const ROOF_DEFAULT: RoofOpts = {
  model: "llama3-8b",
  device: "h100",
  nDevices: 1,
  ctx: 2048,
  batches: [1, 2, 4, 8, 16, 32, 64, 128, 256],
};

export function rooflineClimb(o: RoofOpts = ROOF_DEFAULT): RoofClimb {
  const cm = costModel(MODELS[o.model], DEVICES[o.device], {
    nDevices: o.nDevices,
  });
  const point = (batch: number, s: StepCost, tokens: number): RoofPoint => ({
    batch,
    ctx: o.ctx,
    flops: s.flops,
    bytes: s.bytes,
    time: s.time,
    bound: s.bound,
    intensity: s.intensity,
    attainable: Math.min(cm.flopsRate, s.intensity * cm.byteRate),
    tokensPerS: tokens / s.time,
  });
  return {
    peak: cm.flopsRate,
    bandwidth: cm.byteRate,
    ridge: cm.flopsRate / cm.byteRate,
    points: o.batches.map((b) => point(b, cm.decode(o.ctx * b, b), b)),
    prefill: point(1, cm.prefill([o.ctx]), o.ctx),
  };
}

/** The caption for the climb's k-th point. */
export function roofCaption(c: RoofClimb, k: number): string {
  const p = c.points[k]!;
  const i = p.intensity.toFixed(p.intensity >= 10 ? 0 : 1);
  const tps = Math.round(p.tokensPerS).toLocaleString("en-GB");
  const under =
    p.intensity < c.ridge
      ? `below the ridge at ${c.ridge.toFixed(0)}: memory-bound`
      : `past the ridge at ${c.ridge.toFixed(0)}: compute-bound`;
  return `Batch ${p.batch}: ${i} FLOP per byte, ${under}. The step takes ${ms(p.time)} and yields ${tps} tokens/s.`;
}
