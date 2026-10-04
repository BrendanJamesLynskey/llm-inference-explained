/*
 * src/lib/disagg/engine.ts
 *
 * Operation: a typed wrapper around Disaggregated_Inference_Sim's own
 *            JavaScript engine (`vendor/sim_engine.js`, vendored byte for
 *            byte at the commit in `vendor/VENDORED.json` by
 *            `pnpm vendor:sim`), plus the handful of metrics and number
 *            formats chapters 11–14 quote.
 * Shapes:    simulate(cfg, rows) → SimResult; metricsOf(result) → Metrics.
 * Intuition: a discrete-event simulation of a serving cluster: requests
 *            arrive, queue for prefill, hand their KV cache over a link
 *            and decode in continuous batches. Every timestamp it computes
 *            equals the Python package's (tests/unit/disagg/parity.test.ts).
 * MDX:       /learn/11-why-disaggregate … /learn/14-tradeoffs.
 *
 * The engine is a classic script that sets `globalThis.DisaggSim`; importing
 * it for its side effect keeps the vendored file unmodified.
 */
import "./vendor/sim_engine.js";

/** One request of a workload: [arrival s, prompt tokens, output tokens]. */
export type Row = [number, number, number];

/** The engine's configuration keys (camelCase versions of SimConfig's). */
export type SimConfig = {
  model: string;
  device: string;
  devicesPerInstance: number;
  mode: "disagg" | "colocated";
  nPrefill: number;
  nDecode: number;
  nColocated: number;
  link: string;
  linkChannels?: number;
  ttftSlo: number;
  tpotSlo: number;
  powerCap?: number;
  prefillPowerCap?: number;
  decodePowerCap?: number;
  dvfs?: boolean;
  prefillDevice?: string;
  decodeDevice?: string;
  prefillDevicesPerInstance?: number;
  decodeDevicesPerInstance?: number;
  lmHead?: "all" | "last";
  fftEff?: number;
  engine?: { enob?: number; maskRate?: number; overlap?: boolean };
  kvCompress?: string;
  kvCompressAt?: "transit" | "endpoint";
};

export type SimRequest = {
  arrival: number;
  prompt: number;
  output: number;
  prefillStart: number | null;
  firstToken: number | null;
  kvStart: number | null;
  kvReady: number | null;
  decodeStart: number | null;
  finish: number | null;
  /** Every gap between consecutive output tokens, seconds. */
  itls: number[];
};

export type SimInstance = {
  name: string;
  role: string;
  busy: number;
  ec: number;
  em: number;
  oj: number;
  peakW: number;
};

export type SimResult = {
  reqs: SimRequest[];
  insts: SimInstance[];
  link: {
    energy: number;
    transitJ: number;
    bytes: number;
    busy: number;
    wait: number;
  };
  horizon: number;
};

type Dist = { mean: number; p50: number; p90: number; p99: number };

export type Summary = {
  ttft: Dist;
  tpot: Dist;
  itl: Dist;
  e2e: Dist;
  goodput: number;
  sloAttain: number;
  util: Record<string, number>;
  stageShare: Record<string, number>;
  hot: { stage: string; resource: string; util: number };
  energy: { totalJ: number; avgW: number; jPerTok: number };
  outTokPerS: number;
  raw: { ttft: number[]; itl: number[] };
};

type Engine = {
  MODELS: Record<string, { name: string; L: number; d: number }>;
  DEVICES: Record<string, { name: string }>;
  LINKS: Record<string, { name: string; bw: number; lat: number }>;
  simulate: (cfg: SimConfig, rows: Row[]) => SimResult;
  summarise: (res: SimResult) => Summary;
  percentile: (xs: number[], p: number) => number;
};

/** The vendored engine's API. */
export const engine: Engine = (globalThis as unknown as { DisaggSim: Engine })
  .DisaggSim;

/** The defaults of `disagg-sim` (Llama-3-70B, 4×H100 per instance, 1P1D over InfiniBand NDR, SLOs 1 s and 25 ms). */
export const DEFAULTS: SimConfig = {
  model: "llama3-70b",
  device: "h100",
  devicesPerInstance: 4,
  mode: "disagg",
  nPrefill: 1,
  nDecode: 1,
  nColocated: 2,
  link: "ib-ndr",
  ttftSlo: 1.0,
  tpotSlo: 0.025,
};

/** The numbers chapters 11–14 quote, named as scripts/disagg_reference.py names them. */
export type Metrics = {
  ttft_p99: number;
  tpot_p99: number;
  itl_p50: number;
  itl_p99: number;
  itl_max: number;
  slo: number;
  goodput: number;
  avg_w: number;
  j_tok: number;
  kv_share: number;
  link_util: number;
  /** First token → KV landed on the decode pool, p99 (0 when colocated). */
  handoff_p99: number;
  hotspot: string;
};

export function metricsOf(res: SimResult, s = engine.summarise(res)): Metrics {
  const hand = res.reqs
    .filter((r) => r.kvReady !== null)
    .map((r) => (r.kvReady as number) - (r.firstToken as number));
  let itlMax = -Infinity;
  for (const x of s.raw.itl) if (x > itlMax) itlMax = x;
  return {
    ttft_p99: s.ttft.p99,
    tpot_p99: s.tpot.p99,
    itl_p50: s.itl.p50,
    itl_p99: s.itl.p99,
    itl_max: itlMax,
    slo: s.sloAttain,
    goodput: s.goodput,
    avg_w: s.energy.avgW,
    j_tok: s.energy.jPerTok,
    kv_share: (s.stageShare.kv_wait ?? 0) + (s.stageShare.kv_transfer ?? 0),
    link_util: s.util["kv-link"] ?? 0,
    handoff_p99: hand.length ? engine.percentile(hand, 99) : 0,
    hotspot: `${s.hot.stage} -> ${s.hot.resource}`,
  };
}

/** A run as plain data (safe to post from a worker): metrics and request stamps. */
export type PlainRun = {
  metrics: Metrics;
  result: { reqs: SimRequest[] };
};

/** Run one configuration and keep only plain data. */
export function runPlain(cfg: SimConfig, rows: Row[]): PlainRun {
  const res = engine.simulate(cfg, rows);
  const reqs = res.reqs.map((r) => ({
    arrival: r.arrival,
    prompt: r.prompt,
    output: r.output,
    prefillStart: r.prefillStart,
    firstToken: r.firstToken,
    kvStart: r.kvStart,
    kvReady: r.kvReady,
    decodeStart: r.decodeStart,
    finish: r.finish,
    itls: r.itls.slice(),
  }));
  return { metrics: metricsOf(res), result: { reqs } };
}

/** Run one configuration on a workload and return its metrics. */
export function run(cfg: SimConfig, rows: Row[]): Metrics {
  return metricsOf(engine.simulate(cfg, rows));
}

// ── number formats: the ones examples/results.py writes results.md with ──

/**
 * `f"{x:,.{dp}f}"`: fixed decimals with thousands separators. Python rounds
 * an exact tie (1234.5 to 0 dp) to even; `toFixed` rounds it up, so ties are
 * detected from the exact decimal expansion and made even.
 */
export function fixed(x: number, dp: number): string {
  const a = Math.abs(x);
  let s = a.toFixed(dp);
  const extra = 25;
  const exact = a.toFixed(Math.min(100, dp + extra));
  if (exact.endsWith("5" + "0".repeat(extra - 1))) {
    const truncated = exact.slice(0, exact.length - extra).replace(/\.$/, "");
    if (Number(truncated.slice(-1)) % 2 === 0) s = truncated;
  }
  const [int, frac] = s.split(".");
  const grouped = int!.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  const sign = x < 0 ? "-" : "";
  return frac === undefined ? sign + grouped : `${sign}${grouped}.${frac}`;
}

export const FORMATS = {
  /** results.py `ms()`: 1 decimal below a second, none above. */
  ms: (x: number) =>
    x < 1 ? `${fixed(1e3 * x, 1)} ms` : `${fixed(1e3 * x, 0)} ms`,
  /** results.py `ms2()` (sections 10–15): milliseconds below 10 s, then seconds. */
  ms2: (x: number) => (x < 10 ? `${fixed(1e3 * x, 1)} ms` : `${fixed(x, 1)} s`),
  pct: (x: number) => `${(100 * x).toFixed(1)}%`,
  W: (x: number) => `${fixed(x, 0)} W`,
  f2: (x: number) => x.toFixed(2),
  f3: (x: number) => x.toFixed(3),
} as const;

export type Format = keyof typeof FORMATS;
