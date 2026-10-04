/*
 * src/lib/inference/batching.ts
 *
 * Operation: a small step-by-step scheduler simulation of one serving
 *            instance under four batching policies.
 * Shapes:    simulateBatching(requests, config) → { records, steps, horizon }
 * Intuition: the instance runs one forward pass ("step") at a time; the
 *            policy decides which requests share each step. Step times come
 *            from the roofline cost model (costModel.ts), so a step with
 *            more sequences costs only a little more than one with few, which
 *            is why batching pays.
 * MDX:       /learn/04-batching, /learn/10-serving-metrics.
 *
 * Policies (simplified, single instance, no memory limit beyond maxBatch):
 *   static     — wait for a full batch (or the last requests), prefill it,
 *                decode until its longest member finishes. Finished members
 *                keep their slot (padding) until then.
 *   dynamic    — as static, but start early once the oldest waiting request
 *                has waited `timeout` seconds.
 *   continuous — iteration-level scheduling (Orca): every step, admit
 *                waiting requests into free slots. A step with new prompts
 *                is a prefill step and the running decodes wait for it.
 *   chunked    — continuous plus chunked prefill (Sarathi): each step has a
 *                token budget; decodes go first and prompt chunks fill the
 *                rest, so decodes never stall behind a long prompt.
 * Real engines differ in many details (preemption, memory-aware admission,
 * priorities); this is a teaching model, not a reproduction of one.
 */
import { mulberry32, normalSampler } from "@/lib/transformer/random";

import type { CostModel } from "./costModel";
import type { RequestRecord } from "./metrics";

export type Policy = "static" | "dynamic" | "continuous" | "chunked";
export const POLICIES: Policy[] = [
  "static",
  "dynamic",
  "continuous",
  "chunked",
];

export type Request = {
  id: number;
  arrival: number;
  prompt: number;
  output: number;
};

export type WorkloadSpec = {
  /** Mean arrival rate, requests/s (Poisson). */
  rate: number;
  n: number;
  /** Mean prompt and output lengths, tokens. */
  prompt: number;
  output: number;
  /** Coefficient of variation of both lengths (lognormal). 0 = fixed. */
  cv: number;
  seed: number;
};

/** Seeded Poisson arrivals with lognormal lengths. */
export function makeWorkload(w: WorkloadSpec): Request[] {
  const rng = mulberry32(w.seed);
  const normal = normalSampler(rng);
  const sigma2 = Math.log(1 + w.cv * w.cv);
  const draw = (mean: number): number => {
    if (w.cv === 0) return Math.max(1, Math.round(mean));
    const mu = Math.log(mean) - sigma2 / 2;
    return Math.max(1, Math.round(Math.exp(mu + Math.sqrt(sigma2) * normal())));
  };
  const out: Request[] = [];
  let t = 0;
  for (let i = 0; i < w.n; i++) {
    if (w.rate > 0) t += -Math.log(1 - rng()) / w.rate;
    out.push({
      id: i,
      arrival: t,
      prompt: draw(w.prompt),
      output: draw(w.output),
    });
  }
  return out;
}

export type BatchingConfig = {
  policy: Policy;
  /** Most sequences in one step. */
  maxBatch: number;
  /** chunked: tokens per step (decodes + prompt chunk). */
  chunk: number;
  /** dynamic: longest wait before a partial batch starts, seconds. */
  timeout: number;
  cm: CostModel;
};

export type StepLog = {
  start: number;
  time: number;
  kind: "prefill" | "decode" | "mixed";
  batch: number;
};

type Seq = {
  req: Request;
  rec: RequestRecord;
  /** Prompt tokens prefilled so far (chunked). */
  done: number;
  out: number;
};

export type BatchingResult = {
  records: RequestRecord[];
  steps: StepLog[];
  horizon: number;
};

/** Run every request to completion under `cfg.policy`. */
export function simulateBatching(
  requests: Request[],
  cfg: BatchingConfig,
): BatchingResult {
  const reqs = requests.slice().sort((a, b) => a.arrival - b.arrival);
  const records: RequestRecord[] = [];
  const steps: StepLog[] = [];
  const waiting: Seq[] = [];
  let running: Seq[] = [];
  let next = 0;
  let t = 0;
  const { cm } = cfg;

  const arrive = (): void => {
    while (next < reqs.length && reqs[next]!.arrival <= t) {
      const r = reqs[next++]!;
      const rec: RequestRecord = { ...r, tokenTimes: [] };
      records.push(rec);
      waiting.push({ req: r, rec, done: 0, out: 0 });
    }
  };
  const log = (kind: StepLog["kind"], time: number, batch: number): void => {
    steps.push({ start: t, time, kind, batch });
    t += time;
  };
  const emit = (s: Seq): void => {
    s.out++;
    s.rec.tokenTimes.push(t);
  };
  const ctxOf = (s: Seq): number => s.req.prompt + s.out - 1;

  while (records.length < reqs.length || waiting.length || running.length) {
    arrive();
    if (!waiting.length && !running.length) {
      t = reqs[next]!.arrival;
      continue;
    }

    if (cfg.policy === "static" || cfg.policy === "dynamic") {
      const allArrived = next >= reqs.length;
      const full = waiting.length >= cfg.maxBatch;
      const timedOut =
        cfg.policy === "dynamic" &&
        waiting.length > 0 &&
        // Same expression as the wake-up time below, so rounding can't
        // leave the clock parked just short of the timeout.
        t >= waiting[0]!.req.arrival + cfg.timeout;
      if (!full && !allArrived && !timedOut) {
        // Wait for the next arrival (or the timeout, if sooner).
        let until = reqs[next]!.arrival;
        if (cfg.policy === "dynamic" && waiting.length > 0) {
          until = Math.min(until, waiting[0]!.req.arrival + cfg.timeout);
        }
        t = Math.max(t, until);
        continue;
      }
      const batch = waiting.splice(0, cfg.maxBatch);
      const pre = cm.prefill(batch.map((s) => s.req.prompt));
      log("prefill", pre.time, batch.length);
      for (const s of batch) {
        s.done = s.req.prompt;
        emit(s);
      }
      // Decode until the longest member finishes; finished members stay
      // in the batch as padding (their slots still compute).
      while (batch.some((s) => s.out < s.req.output)) {
        const ctx = batch.reduce((a, s) => a + ctxOf(s), 0);
        const d = cm.decode(ctx, batch.length);
        log("decode", d.time, batch.length);
        for (const s of batch) if (s.out < s.req.output) emit(s);
      }
      continue;
    }

    if (cfg.policy === "continuous") {
      const room = cfg.maxBatch - running.length;
      const admit = waiting.splice(0, Math.max(0, room));
      if (admit.length) {
        const pre = cm.prefill(admit.map((s) => s.req.prompt));
        log("prefill", pre.time, admit.length);
        for (const s of admit) {
          s.done = s.req.prompt;
          emit(s);
        }
        running.push(...admit);
      } else {
        const ctx = running.reduce((a, s) => a + ctxOf(s), 0);
        const d = cm.decode(ctx, running.length);
        log("decode", d.time, running.length);
        for (const s of running) emit(s);
      }
      running = running.filter((s) => s.out < s.req.output);
      continue;
    }

    // chunked: admit into free slots, then decodes first, prompt chunks after.
    const room = cfg.maxBatch - running.length;
    running.push(...waiting.splice(0, Math.max(0, room)));
    const decoding = running.filter((s) => s.done === s.req.prompt);
    let budget = Math.max(0, cfg.chunk - decoding.length);
    const chunks: [number, number][] = [];
    const prefilled: Seq[] = [];
    for (const s of running) {
      if (s.done === s.req.prompt || budget === 0) continue;
      const take = Math.min(budget, s.req.prompt - s.done);
      chunks.push([s.done, s.done + take]);
      s.done += take;
      budget -= take;
      if (s.done === s.req.prompt) prefilled.push(s);
    }
    const ctx = decoding.reduce((a, s) => a + ctxOf(s), 0);
    const step = cm.mixed(chunks, ctx, decoding.length);
    log(
      chunks.length ? "mixed" : "decode",
      step.time,
      decoding.length + chunks.length,
    );
    for (const s of decoding) emit(s);
    for (const s of prefilled) emit(s);
    running = running.filter((s) => s.out < s.req.output);
  }
  return { records, steps, horizon: t };
}
