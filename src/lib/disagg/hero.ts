/*
 * src/lib/disagg/hero.ts
 *
 * Operation: the landing page's hero: the same 200 requests served by the
 *            same eight H100s, colocated (two instances of four) and
 *            disaggregated (one prefill and one decode instance of four),
 *            simulated by the vendored engine; the first requests drawn as
 *            lanes (queue, prefill, KV hand-off, decode, with the decode
 *            stalls marked), with a clock sweeping across them.
 * Shapes:    heroLanes(run, n) → Lane[]; laneFrame(lanes, t) → LaneFrame.
 * Intuition: colocated, a new prompt's prefill pass holds up every decode
 *            on its instance, and those requests' tokens stall; split, the
 *            decode instance only ever decodes, at the price of moving each
 *            prompt's KV cache across the link.
 *            The two configurations and their rows are cases of the parity
 *            fixture (public/disagg/hero.json is cut from it by
 *            scripts/hero_reference.py), so tests/unit/disagg/hero.test.ts
 *            can require the engine's run to equal the Python package's,
 *            request for request.
 * MDX:       the landing page (and /learn/11-why-disaggregate).
 */
import type { Row, SimConfig, SimRequest } from "./engine";

/** public/disagg/hero.json: two parity cases on the same rows. */
export type HeroFile = {
  simulator: string;
  commit: string;
  rows: Row[];
  modes: { key: HeroMode; name: string; cfg: SimConfig }[];
};

export type HeroMode = "colocated" | "disagg";

export const HERO_URL = "/disagg/hero.json";

/** Requests drawn per mode. */
export const HERO_LANES = 8;

/** An inter-token gap this many times the request's median counts as a stall. */
export const STALL_FACTOR = 3;

/** One request's lane: its stamps and the stalls inside its decode. */
export type Lane = {
  rid: number;
  arrival: number;
  prefillStart: number;
  firstToken: number;
  /** Disaggregated only: the KV hand-off. */
  kvStart: number | null;
  kvReady: number | null;
  decodeStart: number;
  finish: number;
  output: number;
  /** [from, to] of every inter-token gap over STALL_FACTOR × the median. */
  stalls: [number, number][];
};

function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)]! : 0;
}

/** The first `n` requests of a run as lanes. */
export function heroLanes(reqs: readonly SimRequest[], n = HERO_LANES): Lane[] {
  return reqs.slice(0, n).map((r, rid) => {
    const first = r.firstToken as number;
    const med = median(r.itls);
    const stalls: [number, number][] = [];
    // token k + 1 arrives itls[k] after token k; token 1 is the first token
    let t = first;
    for (const g of r.itls) {
      if (med > 0 && g > STALL_FACTOR * med) stalls.push([t, t + g]);
      t += g;
    }
    return {
      rid,
      arrival: r.arrival,
      prefillStart: r.prefillStart as number,
      firstToken: first,
      kvStart: r.kvStart,
      kvReady: r.kvReady,
      decodeStart: r.decodeStart as number,
      finish: r.finish as number,
      output: r.output,
      stalls,
    };
  });
}

/** When the last drawn request of either mode finishes. */
export function heroHorizon(...modes: Lane[][]): number {
  let h = 0;
  for (const lanes of modes) for (const l of lanes) h = Math.max(h, l.finish);
  return h;
}

export type Phase =
  | "not yet"
  | "queued"
  | "prefill"
  | "hand-off"
  | "waiting"
  | "decode"
  | "done";

/** Where one request is at time t. */
export function phaseAt(l: Lane, t: number): Phase {
  if (t < l.arrival) return "not yet";
  if (t < l.prefillStart) return "queued";
  if (t < l.firstToken) return "prefill";
  if (l.kvStart !== null && l.kvReady !== null && t < l.kvReady)
    return t < l.kvStart ? "waiting" : "hand-off";
  if (t < l.decodeStart) return "waiting";
  if (t < l.finish) return "decode";
  return "done";
}

export type LaneFrame = {
  t: number;
  phases: Phase[];
  /** Stalls under way or already over at t, summed over the lanes. */
  stalled: number;
  done: number;
};

export function laneFrame(lanes: readonly Lane[], t: number): LaneFrame {
  let stalled = 0;
  let done = 0;
  const phases = lanes.map((l) => {
    for (const [a] of l.stalls) if (a <= t) stalled++;
    const p = phaseAt(l, t);
    if (p === "done") done++;
    return p;
  });
  return { t, phases, stalled, done };
}

/** Frame k of n, evenly spaced from 0 to the horizon. */
export function heroTime(k: number, n: number, horizon: number): number {
  return n <= 1 ? horizon : (k / (n - 1)) * horizon;
}

const ms = (s: number): string => `${(s * 1e3).toFixed(0)} ms`;

export function heroCaption(
  co: LaneFrame,
  dis: LaneFrame,
  n = HERO_LANES,
): string {
  return `t = ${co.t.toFixed(2)} s. Colocated: ${co.done} of ${n} done, ${co.stalled} decode stalls so far. Disaggregated: ${dis.done} of ${n} done, ${dis.stalled} stalls.`;
}

export { ms as heroMs };
