/*
 * src/lib/inference/metrics.ts
 *
 * Operation: serving metrics from per-request token timestamps.
 * Shapes:    summarise(records, slo) → { ttft, tpot, itl, e2e: Dist, … }
 * Intuition: users feel the first token (TTFT) and the gaps between the
 *            rest (TPOT on average per request, ITL per gap). Operators
 *            care about throughput, and about goodput: requests per second
 *            that met *both* SLOs.
 * MDX:       /learn/10-serving-metrics, /learn/04-batching.
 *
 * Conventions follow Disaggregated_Inference_Sim (metrics.py): percentiles
 * interpolate linearly (numpy's default); TPOT = (finish − first token) /
 * (output − 1), for outputs of at least 2 tokens; goodput = requests that
 * met both SLOs / the span of their arrivals.
 */

/** One served request. `tokenTimes[0]` is the first token. */
export type RequestRecord = {
  id: number;
  arrival: number;
  /** Prompt tokens. */
  prompt: number;
  /** Output tokens, including the first. */
  output: number;
  /** Time each output token appeared, seconds. */
  tokenTimes: number[];
};

export type Dist = {
  mean: number;
  p50: number;
  p90: number;
  p99: number;
  max: number;
};

export type Slo = {
  /** Seconds. */
  ttft: number;
  /** Seconds per output token. */
  tpot: number;
};

/** Linear-interpolated percentile of an ascending array (numpy default). */
export function percentileSorted(s: ArrayLike<number>, p: number): number {
  if (s.length === 0) return NaN;
  const k = ((s.length - 1) * p) / 100;
  const lo = Math.floor(k);
  const hi = Math.ceil(k);
  return s[lo]! + (s[hi]! - s[lo]!) * (k - lo);
}

/** Mean, p50, p90, p99 and max of `xs`. */
export function dist(xs: number[]): Dist {
  const s = Float64Array.from(xs).sort();
  const mean = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
  return {
    mean,
    p50: percentileSorted(s, 50),
    p90: percentileSorted(s, 90),
    p99: percentileSorted(s, 99),
    max: s.length ? s[s.length - 1]! : NaN,
  };
}

export const ttftOf = (r: RequestRecord): number =>
  r.tokenTimes[0]! - r.arrival;

export const e2eOf = (r: RequestRecord): number =>
  r.tokenTimes[r.tokenTimes.length - 1]! - r.arrival;

/** Mean time per output token after the first; NaN for 1-token outputs. */
export function tpotOf(r: RequestRecord): number {
  if (r.output < 2) return NaN;
  return (
    (r.tokenTimes[r.tokenTimes.length - 1]! - r.tokenTimes[0]!) / (r.output - 1)
  );
}

/** Every gap between consecutive output tokens of one request. */
export function itlsOf(r: RequestRecord): number[] {
  const out: number[] = [];
  for (let i = 1; i < r.tokenTimes.length; i++) {
    out.push(r.tokenTimes[i]! - r.tokenTimes[i - 1]!);
  }
  return out;
}

export function meetsSlo(r: RequestRecord, slo: Slo): boolean {
  return ttftOf(r) <= slo.ttft && (r.output < 2 || tpotOf(r) <= slo.tpot);
}

/**
 * Time-average number of requests in the system over [0, horizon],
 * measured by sweeping arrival and departure events (no formula).
 */
export function timeAverageInSystem(
  records: RequestRecord[],
  horizon: number,
): number {
  const events: [number, number][] = [];
  for (const r of records) {
    events.push([r.arrival, +1]);
    events.push([r.arrival + e2eOf(r), -1]);
  }
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let area = 0;
  let n = 0;
  let last = 0;
  for (const [t, d] of events) {
    area += n * (t - last);
    n += d;
    last = t;
  }
  area += n * (horizon - last);
  return area / horizon;
}

export type Summary = {
  completed: number;
  ttft: Dist;
  tpot: Dist;
  itl: Dist;
  e2e: Dist;
  /** Output tokens per second over the horizon. */
  tokPerS: number;
  /** Requests per second that met both SLOs (over the arrival span). */
  goodput: number;
  /** Fraction of requests that met both SLOs. */
  sloAttain: number;
  /** Little's law: measured L, and λ·W from arrivals and mean latency. */
  little: { L: number; lambdaW: number };
  horizon: number;
};

/** Summarise a finished run. `horizon` defaults to the last token time. */
export function summarise(
  records: RequestRecord[],
  slo: Slo,
  horizon?: number,
): Summary {
  const done = records.slice().sort((a, b) => a.arrival - b.arrival);
  const H =
    horizon ??
    done.reduce(
      (m, r) => Math.max(m, r.tokenTimes[r.tokenTimes.length - 1]!),
      0,
    );
  const met = done.filter((r) => meetsSlo(r, slo));
  const win =
    done.length > 1 ? done[done.length - 1]!.arrival - done[0]!.arrival : NaN;
  const e2e = done.map(e2eOf);
  const itl: number[] = [];
  for (const r of done) itl.push(...itlsOf(r));
  const outTok = done.reduce((s, r) => s + r.output, 0);
  const meanW = e2e.reduce((a, b) => a + b, 0) / Math.max(1, e2e.length);
  return {
    completed: done.length,
    ttft: dist(done.map(ttftOf)),
    tpot: dist(done.filter((r) => r.output >= 2).map(tpotOf)),
    itl: dist(itl),
    e2e: dist(e2e),
    tokPerS: outTok / H,
    goodput: win > 0 ? met.length / win : NaN,
    sloAttain: done.length ? met.length / done.length : NaN,
    little: {
      L: timeAverageInSystem(done, H),
      lambdaW: (done.length / H) * meanW,
    },
    horizon: H,
  };
}
