/**
 * "Where the time went": the what-if's timeline animation, as pure
 * functions of the engine's request stamps. A request's life is cut at its
 * stamps into phases: waiting for prefill (arrival → prefill start), prefill
 * (→ first token), hand-off (first token → decode start, disaggregated
 * serving only: KV waiting for decode room and crossing the link), and
 * decode (→ finish). The animation shows the same requests under the
 * "before" and "after" configurations, with a clock sweeping simulated time;
 * a frame is a function of (lanes, step) only.
 */
import type { EngineRequest } from "./engine";

export type Phase = "queue" | "prefill" | "handoff" | "decode";
export const PHASES: readonly Phase[] = [
  "queue",
  "prefill",
  "handoff",
  "decode",
];

export type Segment = { phase: Phase; t0: number; t1: number };

/** One request's phases (empty when it never finished, e.g. rejected). */
export function segments(r: EngineRequest): Segment[] {
  if (r.prefillStart === null || r.firstToken === null || r.finish === null)
    return [];
  const out: Segment[] = [];
  const push = (phase: Phase, t0: number, t1: number) => {
    if (t1 > t0) out.push({ phase, t0, t1 });
  };
  push("queue", r.arrival, r.prefillStart);
  push("prefill", r.prefillStart, r.firstToken);
  const dec =
    r.decodeStart !== null && r.decodeStart > r.firstToken
      ? r.decodeStart
      : r.firstToken;
  push("handoff", r.firstToken, dec);
  push("decode", dec, r.finish);
  return out;
}

/** The phase a request is in at time t, or "waiting" (not arrived) / "done". */
export function phaseAt(
  segs: readonly Segment[],
  arrival: number,
  t: number,
): Phase | "waiting" | "done" {
  if (t < arrival) return "waiting";
  for (const s of segs) if (t >= s.t0 && t < s.t1) return s.phase;
  return segs.length && t >= segs[segs.length - 1]!.t1 ? "done" : "waiting";
}

export type Lanes = {
  /** The requests shown, by index in the workload. */
  ids: number[];
  before: Segment[][];
  after: Segment[][];
  arrival: number[];
  t0: number;
  t1: number;
};

/**
 * The requests to draw: `count` consecutive requests starting a tenth of the
 * way into the workload (past the ramp-up), and the time window from the
 * first one's arrival to the last finish under either configuration.
 */
export function lanes(
  before: readonly EngineRequest[],
  after: readonly EngineRequest[],
  count = 24,
): Lanes {
  const start = Math.floor(before.length / 10);
  const ids: number[] = [];
  for (let i = start; i < before.length && ids.length < count; i++) ids.push(i);
  const b = ids.map((i) => segments(before[i]!));
  const a = ids.map((i) => segments(after[i]!));
  const arrival = ids.map((i) => before[i]!.arrival);
  const t0 = arrival.length ? Math.min(...arrival) : 0;
  let t1 = t0;
  for (const segs of [...b, ...a])
    for (const s of segs) if (s.t1 > t1) t1 = s.t1;
  return { ids, before: b, after: a, arrival, t0, t1 };
}

/** The clock's time at a step of `n` (step 0 = t0, step n-1 = t1). */
export function timeAt(l: Lanes, step: number, n: number): number {
  return n <= 1 ? l.t1 : l.t0 + ((l.t1 - l.t0) * step) / (n - 1);
}

export type Counts = Record<Phase | "waiting" | "done", number>;

/** How many of the shown requests are in each phase at a time. */
export function counts(
  segs: readonly Segment[][],
  arrival: readonly number[],
  t: number,
): Counts {
  const c: Counts = {
    waiting: 0,
    queue: 0,
    prefill: 0,
    handoff: 0,
    decode: 0,
    done: 0,
  };
  segs.forEach((s, i) => {
    c[phaseAt(s, arrival[i]!, t)] += 1;
  });
  return c;
}

const fmtT = (t: number) => (t < 10 ? t.toFixed(2) : t.toFixed(1));

/** The frame's caption (also read out by screen readers). */
export function caption(l: Lanes, step: number, n: number): string {
  const t = timeAt(l, step, n);
  const b = counts(l.before, l.arrival, t);
  const a = counts(l.after, l.arrival, t);
  const part = (c: Counts) =>
    `${c.done} done, ${c.decode} decoding, ${c.prefill} in prefill, ${c.queue} queued` +
    (c.handoff ? `, ${c.handoff} in hand-off` : "");
  return `t = ${fmtT(t - l.t0)} s: before: ${part(b)}; after: ${part(a)}.`;
}
