/**
 * A typed wrapper around Disaggregated_Inference_Sim's own JavaScript
 * engine (`vendor/sim_engine.js`, vendored byte for byte at the commit in
 * `vendor/VENDORED.json`). It is bit-exact with the simulator's Python
 * package (tested in the simulator's repository, and again here on fourteen
 * sweep configurations: tests/unit/parity.test.ts), and given a sweep
 * point's recorded configuration and the recorded workload it reproduces
 * that point's latencies exactly (tests/unit/sweep.test.ts).
 *
 * The engine is a classic script that sets `globalThis.DisaggSim`; importing
 * it for its side effect keeps the vendored file unmodified.
 */
import "./vendor/sim_engine.js";

/** One request: [arrival s, prompt tokens, output tokens, session extras?]. */
export type Row =
  | [number, number, number]
  | [
      number,
      number,
      number,
      {
        chain: [string, number][];
        emit: string | null;
        after: number | null;
        think: number;
      },
    ];

/** The engine's configuration (camelCase versions of SimConfig's keys). */
export type EngineConfig = Record<string, unknown>;

export type EngineRequest = {
  arrival: number;
  prompt: number;
  output: number;
  prefillStart: number | null;
  firstToken: number | null;
  kvStart: number | null;
  kvReady: number | null;
  decodeStart: number | null;
  finish: number | null;
};

type Dist = { mean: number; p50: number; p90: number; p99: number };

export type EngineSummary = {
  completed: number;
  rejected: number;
  ttft: Dist;
  tpot: Dist;
  itl: Dist;
  e2e: Dist;
  sloAttain: number;
  outTokPerS: number;
  energy: { totalJ: number; avgW: number; jPerTok: number };
  stages: Record<string, number>;
  scheduler: null | {
    preemptions?: number;
    prefixCache?: { hitRate: number; hitTokens: number };
    speculative?: { tokensPerVerify: number };
  };
};

/** One forward pass, as the engine's cost model prices it. */
export type StepCost = {
  time: number;
  flops: number;
  bytes: number;
  bound: string;
  ec: number;
  em: number;
  /** Scale-up communication seconds (parallel instances only). */
  commT?: number;
};

/** The engine's cost model for one instance (CostModel in hardware.py). */
export type CostModel = {
  kvCap: number | null;
  fits: boolean;
  idleW: number;
  prefill: (lens: number[]) => StepCost;
  decodeSum: (ctx: number, batch: number) => StepCost;
};

type Engine = {
  DEVICES: Record<
    string,
    { name: string; tdp: number; idle: number; native: Record<string, number> }
  >;
  LINKS: Record<string, { name: string; bw: number; lat: number }>;
  MODELS: Record<string, Record<string, unknown>>;
  QUANT_FORMATS: Record<string, number>;
  derive: (m: Record<string, unknown>) => Record<string, unknown>;
  deviceFor: (key: string, cfg: EngineConfig) => Record<string, unknown>;
  costModel: (
    model: Record<string, unknown>,
    dev: Record<string, unknown>,
    n: number,
    overhead: number,
    powerCap: number | null,
    dvfs: boolean,
    sMin: number | undefined,
    prefillOnly: boolean,
    opts?: Record<string, unknown>,
  ) => CostModel;
  simulate: (
    cfg: EngineConfig,
    rows: Row[],
  ) => {
    reqs: (EngineRequest & { itls: number[] })[];
    horizon: number;
    insts: { peakW: number; ec: number; em: number; busy: number }[];
  };
  summarise: (res: unknown) => EngineSummary;
  expectedTokens: (alpha: number, gamma: number) => number;
};

/** The vendored engine's API. */
export const engine: Engine = (globalThis as unknown as { DisaggSim: Engine })
  .DisaggSim;

/** The stages a request's time is split into (the engine's `stages`). */
export const STAGES = [
  "prefill_queue",
  "prefill",
  "kv_wait",
  "kv_transfer",
  "decode_queue",
  "decode",
] as const;
export type Stage = (typeof STAGES)[number];

/** The numbers the what-if compares. */
export type RunMetrics = {
  ttft_p50: number;
  ttft_p99: number;
  tpot_p50: number;
  tpot_p99: number;
  itl_p50: number;
  itl_p99: number;
  e2e_p50: number;
  slo_attain: number;
  out_tok_s: number;
  j_per_tok: number;
  completed: number;
  rejected: number;
  /** Mean seconds per request in each stage. */
  stages: Record<Stage, number>;
};

/** A run as plain data (safe to post from a worker). */
export type PlainRun = {
  metrics: RunMetrics;
  /** Every request's stamps, in workload order. */
  reqs: EngineRequest[];
};

export function metricsOf(s: EngineSummary): RunMetrics {
  const stages = {} as Record<Stage, number>;
  for (const k of STAGES) stages[k] = s.stages[k] ?? 0;
  return {
    ttft_p50: s.ttft.p50,
    ttft_p99: s.ttft.p99,
    tpot_p50: s.tpot.p50,
    tpot_p99: s.tpot.p99,
    itl_p50: s.itl.p50,
    itl_p99: s.itl.p99,
    e2e_p50: s.e2e.p50,
    slo_attain: s.sloAttain,
    out_tok_s: s.outTokPerS,
    j_per_tok: s.energy.jPerTok,
    completed: s.completed,
    rejected: s.rejected,
    stages,
  };
}

/** Run one configuration and keep only plain data. */
export function runPlain(cfg: EngineConfig, rows: Row[]): PlainRun {
  const res = engine.simulate(cfg, rows);
  const s = engine.summarise(res);
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
  }));
  return { metrics: metricsOf(s), reqs };
}
