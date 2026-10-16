/**
 * The chapters' mechanism animations, as pure functions of state the
 * simulator recorded. `scripts/mechanisms_reference.py` runs small named
 * scenarios through Disaggregated_Inference_Sim's own Python package (at the
 * vendored commit) and writes every forward pass of every instance, with
 * each running row's state, to `public/tradeoffs/mechanisms/<scenario>.json`;
 * the unit tests run the vendored JavaScript engine on the same
 * configurations and rows and require the same request timestamps, bit for
 * bit. Each animation here turns those records into a frame: a pure
 * function of (variant, step) or (variant, time), so any frame can be
 * reproduced and is tested against the Python record.
 */
import type { EngineRequest } from "./engine";
import { counts, segments } from "./timeline";

/** [rid, prompt tokens computed, prefill target, output tokens, KV units held, cached tokens, output tokens at admission]. */
export type MRow = [number, number, number, number, number, number, number];

/** One forward pass of one instance, as the simulator recorded it at its start. */
export type MStep = {
  inst: number;
  t0: number;
  dt: number;
  label: string;
  bound: string;
  /** Dynamic joules: compute and memory. */
  ec: number;
  em: number;
  /** Seconds of scale-up communication and of draft passes in this step. */
  comm: number;
  draft: number;
  /** KV units in use and the capacity, in the instance's units. */
  kv: number;
  cap: number;
  rows: MRow[];
  queue?: number[];
  swapped?: number[];
  swap_t?: number;
  /** Prefix-cache segments: [key, tokens, units, pinning requests]. */
  cache?: [string, number, number, number][];
  /** What each row did in this step: [rid, prompt tokens computed, output tokens emitted]. */
  work: [number, number, number][];
};

/** [t, kind, rid, instance]: kind is "finish", "preempt-recompute" or "preempt-swap". */
export type MEvent = [number, string, number, number];

/** Seven timestamps (seconds or null), then prompt, output, ITLs, zero ITLs. */
export type MReq = (number | null)[];

export type MSummary = {
  ttft_p50: number;
  ttft_p99: number;
  tpot_p50: number;
  tpot_p99: number;
  itl_p99: number;
  completed: number;
  j_per_tok: number;
  total_J: number;
  preemptions: number;
  hit_rate: number;
  tokens_per_verify: number;
};

export type MVariant = {
  key: string;
  label: string;
  cfg: Record<string, unknown>;
  rows: unknown[];
  insts: string[];
  blk: number[];
  idle_w: number[];
  steps: MStep[];
  events: MEvent[];
  reqs: MReq[];
  handoff_bytes: (number | null)[];
  stamps_sha256: string;
  horizon: number;
  summary: MSummary;
};

export type MFile = {
  commit: string;
  scenario: string;
  title: string;
  variants: MVariant[];
};

/** The scenarios the reference script writes (one file each). */
export const SCENARIOS = [
  "batching",
  "paged",
  "prefix",
  "pools",
  "hetero",
  "handoff",
  "ced",
  "speculative",
  "power",
] as const;
export type Scenario = (typeof SCENARIOS)[number];

/** Where the browser fetches a scenario. */
export const mechUrl = (s: Scenario | "cost_model"): string =>
  `/tradeoffs/mechanisms/${s}.json`;

// ─────────────────────────────────────────────────────────── basics ──

const at = (r: MReq, i: number): number | null => r[i] ?? null;

/** A recorded request as the engine's request type (for timeline.ts). */
export function engineReq(r: MReq): EngineRequest {
  return {
    arrival: at(r, 0)!,
    prefillStart: at(r, 1),
    firstToken: at(r, 2),
    kvStart: at(r, 3),
    kvReady: at(r, 4),
    decodeStart: at(r, 5),
    finish: at(r, 6),
    prompt: at(r, 7)!,
    output: at(r, 8)!,
  };
}

export function variantOf(f: MFile, key: string): MVariant {
  const v = f.variants.find((x) => x.key === key);
  if (!v) throw new Error(`no variant ${key} in ${f.scenario}`);
  return v;
}

/** One instance's steps, in time order. */
export function instSteps(v: MVariant, inst: number): MStep[] {
  return v.steps.filter((s) => s.inst === inst);
}

/** The index of the last step that started at or before t (-1: none yet). */
export function stepIndexAt(steps: readonly MStep[], t: number): number {
  let lo = 0;
  let hi = steps.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (steps[mid]!.t0 <= t) {
      best = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return best;
}

/** The simulated time of frame k of n (frame 0 = 0, the last = the horizon). */
export function frameTime(v: MVariant, k: number, n: number): number {
  return n <= 1 ? v.horizon : (v.horizon * k) / (n - 1);
}

/** The step running on an instance at time t, or null when it is idle. */
export function busyAt(steps: readonly MStep[], t: number): MStep | null {
  const i = stepIndexAt(steps, t);
  if (i < 0) return null;
  const s = steps[i]!;
  return t < s.t0 + s.dt ? s : null;
}

export type StepKind = "prefill" | "decode" | "mixed" | "ced";

/** What a step's label says it was. */
export function kindOf(label: string): StepKind {
  if (label.startsWith("prefill")) return "prefill";
  if (label.startsWith("decode")) return "decode";
  if (label.startsWith("ced")) return "ced";
  return "mixed";
}

/** Whether a step computed prompt tokens (a prefill, or a mixed step with chunks). */
export function hasPrompt(s: MStep): boolean {
  return kindOf(s.label) === "prefill" || s.work.some((w) => w[1] > 0);
}

/** The numbers in a step's label: b (decode rows), n, chunks, tok. */
export function parseLabel(label: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of label.matchAll(/([a-z]+)=(\d+)/g)) out[m[1]!] = Number(m[2]);
  return out;
}

/** Events up to time t. */
export function eventsUpTo(v: MVariant, t: number): MEvent[] {
  return v.events.filter((e) => e[0] <= t);
}

const fmtMs = (s: number): string => {
  const ms = s * 1e3;
  if (ms >= 1000) return `${(ms / 1e3).toFixed(2)} s`;
  return `${ms >= 100 ? ms.toFixed(0) : ms.toFixed(1)} ms`;
};
export { fmtMs as fmtSeconds };

const plural = (n: number, one: string, many = `${one}s`): string =>
  `${n} ${n === 1 ? one : many}`;

// ───────────────────────────────────────────── chapter 1: batching ──

export type BatchSlice = {
  rid: number;
  kind: "decode" | "prefill";
  tokens: number;
};

export type BatchView = {
  k: number;
  t0: number;
  dt: number;
  slices: BatchSlice[];
  decodeRows: number;
  prefillTokens: number;
  waiting: number;
  /** For each decode row: time since its previous token when this step ends (its ITL). */
  gaps: { rid: number; gap: number }[];
  maxGap: number;
  /** The token budget of one step. */
  budget: number;
};

/** The default prefill token budget (SimConfig.max_prefill_tokens). */
export const MAX_PREFILL_TOKENS = 8192;

export function budgetOf(v: MVariant): number {
  const b = v.cfg.maxNumBatchedTokens;
  return typeof b === "number" ? b : MAX_PREFILL_TOKENS;
}

/**
 * Step k of the single instance: what shared the forward pass. A row that
 * computed prompt tokens is a prefill slice (its token out is its first);
 * a row that only emitted tokens is a decode row.
 */
export function batchView(v: MVariant, k: number): BatchView {
  const steps = instSteps(v, 0);
  const last = new Map<number, number>();
  for (let i = 0; i < k; i++) {
    const s = steps[i]!;
    for (const [rid, p, d] of s.work) if (p + d > 0) last.set(rid, s.t0 + s.dt);
  }
  const s = steps[k]!;
  const slices: BatchSlice[] = [];
  const gaps: { rid: number; gap: number }[] = [];
  for (const [rid, p, d] of s.work) {
    if (p > 0) slices.push({ rid, kind: "prefill", tokens: p });
    else if (d > 0) {
      slices.push({ rid, kind: "decode", tokens: 1 });
      const prev = last.get(rid);
      if (prev !== undefined) gaps.push({ rid, gap: s.t0 + s.dt - prev });
    }
  }
  // the old colocated instance's prefill rows are not in its running list
  if (
    kindOf(s.label) === "prefill" &&
    !slices.some((x) => x.kind === "prefill")
  )
    slices.push({
      rid: -1,
      kind: "prefill",
      tokens: parseLabel(s.label).tok ?? 0,
    });
  const decodeRows = slices.filter((x) => x.kind === "decode").length;
  const prefillTokens = slices
    .filter((x) => x.kind === "prefill")
    .reduce((a, x) => a + x.tokens, 0);
  return {
    k,
    t0: s.t0,
    dt: s.dt,
    slices,
    decodeRows,
    prefillTokens,
    waiting: s.queue?.length ?? 0,
    gaps,
    maxGap: gaps.reduce((a, g) => Math.max(a, g.gap), 0),
    budget: budgetOf(v),
  };
}

export function batchCaption(b: BatchView): string {
  const parts: string[] = [];
  if (b.decodeRows) parts.push(plural(b.decodeRows, "decode row"));
  if (b.prefillTokens) parts.push(`${b.prefillTokens} prompt tokens`);
  let s = `Step ${b.k + 1} at ${fmtMs(b.t0)} takes ${fmtMs(b.dt)}: ${parts.join(" and ") || "nothing"}`;
  s += b.waiting ? `; ${plural(b.waiting, "request")} waiting.` : ".";
  if (b.decodeRows)
    s += ` The decode rows waited up to ${fmtMs(b.maxGap)} for this token.`;
  return s;
}

/** Which equation term the frame is about: the budget τ when prompts are chunked, the stall when a decode row waits behind a prompt. */
export function batchHl(b: BatchView): string {
  if (b.prefillTokens && b.decodeRows) return "u";
  if (b.prefillTokens) return "p";
  return "a";
}

// ──────────────────────────────────────────── chapter 2: KV memory ──

export type KvCell = { rid: number; state: "used" | "partial" | "reserved" };

export type KvView = {
  t: number;
  step: MStep | null;
  /** Tokens per drawn cell. */
  cellTokens: number;
  cells: (KvCell | null)[];
  /** Units allocated (the simulator's kv_used) and the capacity, in tokens. */
  allocTokens: number;
  heldTokens: number;
  capTokens: number;
  running: number;
  queued: number;
  swapped: number;
  preemptions: number;
  swaps: number;
};

/** Tokens a row's private KV holds (ScheduledInstance.private_units, in tokens). */
export function heldTokens(r: MRow): number {
  const [, dp, , to, , covered, base] = r;
  return Math.max(0, dp - covered + to - base);
}

/** The KV memory at time t, drawn as cells of `cellTokens` tokens, rows in arrival order. */
export function kvView(v: MVariant, t: number, cellTokens = 64): KvView {
  const steps = instSteps(v, 0);
  const i = stepIndexAt(steps, t);
  const step = i >= 0 ? steps[i]! : null;
  const blk = v.blk[0] ?? 1;
  const cap = step ? step.cap : (steps[0]?.cap ?? 0);
  const capTokens = cap * blk;
  const n = Math.ceil(capTokens / cellTokens);
  const cells: (KvCell | null)[] = new Array<KvCell | null>(n).fill(null);
  let c = 0;
  let allocTokens = 0;
  let held = 0;
  const rows = step ? [...step.rows].sort((a, b) => a[0] - b[0]) : [];
  for (const r of rows) {
    const a = r[4] * blk;
    const h = Math.min(heldTokens(r), a);
    allocTokens += a;
    held += h;
    const ac = Math.ceil(a / cellTokens);
    const full = Math.floor(h / cellTokens);
    const part = h % cellTokens ? 1 : 0;
    for (let j = 0; j < ac && c < n; j++, c++)
      cells[c] = {
        rid: r[0],
        state: j < full ? "used" : j < full + part ? "partial" : "reserved",
      };
  }
  const ev = eventsUpTo(v, t);
  return {
    t,
    step,
    cellTokens,
    cells,
    allocTokens,
    heldTokens: held,
    capTokens,
    running: rows.length,
    queued: step?.queue?.length ?? 0,
    swapped: step?.swapped?.length ?? 0,
    preemptions: ev.filter((e) => e[1].startsWith("preempt")).length,
    swaps: ev.filter((e) => e[1] === "preempt-swap").length,
  };
}

const pct = (x: number): string => `${(100 * x).toFixed(0)}%`;

export function kvCaption(k: KvView): string {
  if (!k.step) return "Nothing has arrived yet.";
  let s = `t = ${fmtMs(k.t)}: ${plural(k.running, "request")} running, ${k.queued} waiting`;
  if (k.swapped) s += `, ${k.swapped} swapped out to host memory`;
  s += `. ${pct(k.allocTokens / k.capTokens)} of the KV memory is allocated, ${pct(k.heldTokens / k.capTokens)} holds tokens`;
  s += k.preemptions ? `; ${plural(k.preemptions, "preemption")} so far.` : ".";
  return s;
}

// ──────────────────────────────────────── chapter 3: prefix caching ──

export type PrefixRow = {
  rid: number;
  prompt: number;
  cached: number;
  computed: number;
  out: number;
};

export type PrefixView = {
  t: number;
  segments: { key: string; tokens: number; refs: number }[];
  cachedTokens: number;
  rows: PrefixRow[];
  /** Over requests admitted so far: prompt tokens, and those found cached. */
  promptTokens: number;
  hitTokens: number;
};

export function prefixView(v: MVariant, t: number): PrefixView {
  const steps = instSteps(v, 0);
  const i = stepIndexAt(steps, t);
  const seen = new Set<number>();
  let promptTokens = 0;
  let hitTokens = 0;
  for (let j = 0; j <= i; j++)
    for (const r of steps[j]!.rows)
      if (!seen.has(r[0])) {
        seen.add(r[0]);
        promptTokens += r[2];
        hitTokens += r[5];
      }
  const step = i >= 0 ? steps[i]! : null;
  const segments = (step?.cache ?? []).map(([key, tokens, , refs]) => ({
    key,
    tokens,
    refs,
  }));
  return {
    t,
    segments,
    cachedTokens: segments.reduce((a, x) => a + x.tokens, 0),
    rows: (step?.rows ?? []).map((r) => ({
      rid: r[0],
      prompt: r[2],
      cached: r[5],
      computed: Math.max(0, r[1] - r[5]),
      out: r[3],
    })),
    promptTokens,
    hitTokens,
  };
}

export function prefixCaption(p: PrefixView): string {
  const hit = p.promptTokens ? p.hitTokens / p.promptTokens : 0;
  let s = `t = ${fmtMs(p.t)}: `;
  s += p.segments.length
    ? `${plural(p.segments.length, "segment")} cached (${p.cachedTokens} tokens, ${p.segments.filter((x) => x.refs > 0).length} in use)`
    : "nothing cached";
  s += `; ${plural(p.rows.length, "request")} running. Prompt tokens found in the cache so far: ${pct(hit)}.`;
  return s;
}

// ──────────────────────────────────── chapters 4–7: pool timelines ──

export type Box = { t0: number; t1: number; kind: StepKind; label: string };

export type PoolsView = {
  t: number;
  /** Per instance: the step running now (or null), and how many have started. */
  now: (MStep | null)[];
  started: number[];
  /** Hand-offs in flight and landed. */
  inFlight: number;
  landed: number;
};

/** Every step of every instance as a box on the time axis. */
export function boxes(v: MVariant): Box[][] {
  return v.insts.map((_, i) =>
    instSteps(v, i).map((s) => ({
      t0: s.t0,
      t1: s.t0 + s.dt,
      kind: kindOf(s.label),
      label: s.label,
    })),
  );
}

/** The KV hand-offs: [rid, start, landed]. */
export function transfers(v: MVariant): [number, number, number][] {
  const out: [number, number, number][] = [];
  v.reqs.forEach((r, rid) => {
    const a = at(r, 3);
    const b = at(r, 4);
    if (a !== null && b !== null) out.push([rid, a, b]);
  });
  return out;
}

export function poolsView(v: MVariant, t: number): PoolsView {
  const now: (MStep | null)[] = [];
  const started: number[] = [];
  v.insts.forEach((_, i) => {
    const st = instSteps(v, i);
    now.push(busyAt(st, t));
    started.push(stepIndexAt(st, t) + 1);
  });
  const tr = transfers(v);
  return {
    t,
    now,
    started,
    inFlight: tr.filter(([, a, b]) => a <= t && t < b).length,
    landed: tr.filter(([, , b]) => b <= t).length,
  };
}

export function poolsCaption(v: MVariant, p: PoolsView): string {
  const doing = v.insts.map((name, i) => {
    const s = p.now[i];
    return `${name} ${s ? s.label : "idle"}`;
  });
  let out = `t = ${fmtMs(p.t)}: ${doing.join("; ")}.`;
  if (v.cfg.mode === "disagg")
    out += ` Link: ${p.inFlight} in flight, ${p.landed} landed.`;
  const reqs = v.reqs.map(engineReq);
  const c = counts(
    reqs.map((r) => segments(r)),
    reqs.map((r) => r.arrival),
    p.t,
  );
  out += ` Requests: ${c.done} done, ${c.decode} decoding, ${c.prefill} in prefill, ${c.queue} queued`;
  if (c.handoff) out += `, ${c.handoff} in hand-off`;
  return `${out}.`;
}

// ──────────────────────────────────── chapter 10: speculative decoding ──

export type DraftRow = {
  rid: number;
  /** Tokens kept: the accepted drafts plus one from the target. */
  kept: number;
  accepted: number;
  /** Drafts proposed (gamma), and how many were rejected or never checked. */
  gamma: number;
  rejected: number;
};

export type SpecView = {
  k: number;
  t0: number;
  dt: number;
  kind: StepKind;
  rows: DraftRow[];
  /** Output tokens so far, per request. */
  out: number[];
  draftTime: number;
};

export function gammaOf(v: MVariant): number {
  const s = v.cfg.speculative as { gamma?: number } | undefined;
  return s?.gamma ?? 0;
}

export function specView(v: MVariant, k: number): SpecView {
  const steps = instSteps(v, 0);
  const s = steps[k]!;
  const g = gammaOf(v);
  const out = new Array<number>(v.reqs.length).fill(0);
  for (let i = 0; i <= k; i++)
    for (const [rid, , d] of steps[i]!.work) out[rid] = (out[rid] ?? 0) + d;
  const kind = kindOf(s.label);
  const rows: DraftRow[] = [];
  if (kind !== "prefill")
    for (const [rid, p, d] of s.work)
      if (p === 0 && d > 0) {
        const accepted = g ? d - 1 : 0;
        rows.push({
          rid,
          kept: d,
          accepted,
          gamma: g,
          rejected: g ? g - accepted : 0,
        });
      }
  return { k, t0: s.t0, dt: s.dt, kind, rows, out, draftTime: s.draft };
}

export function specCaption(v: MVariant, s: SpecView): string {
  const head = `Pass ${s.k + 1} at ${fmtMs(s.t0)} (${fmtMs(s.dt)})`;
  if (s.kind === "prefill") return `${head}: a prompt's prefill.`;
  if (!gammaOf(v))
    return `${head}: one target pass, one token for each of ${plural(s.rows.length, "row")}.`;
  const per = s.rows
    .map(
      (r) =>
        `request ${r.rid} keeps ${r.accepted} of ${r.gamma} drafts + 1 = ${r.kept}`,
    )
    .join("; ");
  return `${head}, ${fmtMs(s.draftTime)} of it drafting: ${per}.`;
}

// ─────────────────────────────────────────── chapter 11: power ──

export type PowerBar = {
  t0: number;
  t1: number;
  /** Watts per GPU: static (idle), compute, memory. */
  idle: number;
  compute: number;
  memory: number;
  bound: string;
};

export function devicesOf(v: MVariant): number {
  const n = v.cfg.devicesPerInstance;
  return typeof n === "number" ? n : 1;
}

/** Per-step power of instance 0, per GPU (the power model: idle + dynamic joules / step time). */
export function powerBars(v: MVariant): PowerBar[] {
  const n = devicesOf(v);
  const idle = v.idle_w[0]! / n;
  return instSteps(v, 0).map((s) => ({
    t0: s.t0,
    t1: s.t0 + s.dt,
    idle,
    compute: s.ec / s.dt / n,
    memory: s.em / s.dt / n,
    bound: s.bound,
  }));
}

/** The instance's peak power over any single step, W (the engine's peakW). */
export function peakPower(v: MVariant): number {
  let p = 0;
  for (const s of instSteps(v, 0)) {
    const w = v.idle_w[0]! + (s.ec + s.em) / s.dt;
    if (w > p) p = w;
  }
  return p;
}

export function powerCaption(v: MVariant, bars: PowerBar[], t: number): string {
  const i = bars.findIndex((b) => b.t0 <= t && t < b.t1);
  const cap = v.cfg.powerCap;
  const head = `t = ${fmtMs(t)}: `;
  if (i < 0)
    return `${head}idle, ${bars[0] ? bars[0].idle.toFixed(0) : 0} W per GPU of static power.`;
  const b = bars[i]!;
  const total = b.idle + b.compute + b.memory;
  let s = `${head}a ${b.bound}-bound step draws ${total.toFixed(0)} W per GPU (static ${b.idle.toFixed(0)}, compute ${b.compute.toFixed(0)}, memory ${b.memory.toFixed(0)})`;
  if (typeof cap === "number") s += `, under a ${cap.toFixed(0)} W cap`;
  return `${s}.`;
}

// ───────────────────────────────── chapter 8: ring all-reduce, GPipe ──

export type RingState = {
  phase: "start" | "reduce-scatter" | "all-gather";
  step: number;
  /** have[gpu][chunk]: how many GPUs' contributions that chunk holds (n = complete). */
  have: number[][];
  sends: { from: number; to: number; chunk: number }[];
};

/**
 * A ring all-reduce over n GPUs, each holding its own vector cut into n
 * chunks: n - 1 reduce-scatter steps (each GPU passes one chunk on and adds
 * the one it receives), then n - 1 all-gather steps (the complete chunks go
 * round). Patarasuk and Yuan's bandwidth-optimal schedule.
 */
export function ringStates(n: number): RingState[] {
  const have = Array.from({ length: n }, () => new Array<number>(n).fill(1));
  const out: RingState[] = [
    { phase: "start", step: 0, have: have.map((r) => [...r]), sends: [] },
  ];
  for (let s = 0; s < n - 1; s++) {
    const sends = Array.from({ length: n }, (_, i) => ({
      from: i,
      to: (i + 1) % n,
      chunk: (((i - s) % n) + n) % n,
    }));
    const before = have.map((r) => [...r]);
    for (const x of sends)
      have[x.to]![x.chunk] =
        before[x.to]![x.chunk]! + before[x.from]![x.chunk]!;
    out.push({
      phase: "reduce-scatter",
      step: s + 1,
      have: have.map((r) => [...r]),
      sends,
    });
  }
  for (let s = 0; s < n - 1; s++) {
    const sends = Array.from({ length: n }, (_, i) => ({
      from: i,
      to: (i + 1) % n,
      chunk: (((i + 1 - s) % n) + n) % n,
    }));
    for (const x of sends) have[x.to]![x.chunk] = n;
    out.push({
      phase: "all-gather",
      step: n + s,
      have: have.map((r) => [...r]),
      sends,
    });
  }
  return out;
}

/** Seconds per ring step: one chunk of S/n bytes over the link, plus its latency. */
export function ringStepTime(
  bytes: number,
  n: number,
  bw: number,
  lat: number,
): number {
  return bytes / n / bw + lat;
}

export function ringCaption(s: RingState, n: number): string {
  if (s.phase === "start")
    return `Each of ${n} GPUs holds its own partial result, cut into ${n} chunks.`;
  const done = s.have.flat().filter((x) => x === n).length;
  if (s.phase === "reduce-scatter")
    return `Reduce-scatter step ${s.step} of ${n - 1}: every GPU passes one chunk to its neighbour, which adds it to its own; ${done} of ${n * n} chunk copies are complete.`;
  return `All-gather step ${s.step - n + 1} of ${n - 1}: the complete chunks travel round the ring; ${done} of ${n * n} chunk copies are complete.`;
}

/** GPipe: which micro-batch each of p stages runs in each of m + p - 1 slots (null: bubble). */
export function gpipe(p: number, m: number): (number | null)[][] {
  const slots = m + p - 1;
  return Array.from({ length: p }, (_, j) =>
    Array.from({ length: slots }, (_, s) =>
      s - j >= 0 && s - j < m ? s - j : null,
    ),
  );
}

export const bubble = (p: number, m: number): number => (p - 1) / (m + p - 1);

export function gpipeCaption(p: number, m: number, slot: number): string {
  const g = gpipe(p, m);
  const busy = g.filter((row) => row[slot] !== null).length;
  return `Slot ${slot + 1} of ${m + p - 1}: ${busy} of ${p} stages busy. Over the whole step the bubble is ${pct(bubble(p, m))} of stage time.`;
}

// ─────────────────────────────────────── chapter 9: bytes per step ──

export type Stream = {
  label: string;
  /** The step's time, its total bytes and the weights' share of them. */
  time: number;
  total: number;
  weights: number;
};

/** How far through its bytes each step is at time t (bytes stream at a constant rate over the step). */
export function streamAt(
  s: Stream,
  t: number,
): { phase: "weights" | "kv" | "done"; frac: number } {
  const done = Math.min(1, t / s.time) * s.total;
  if (done >= s.total) return { phase: "done", frac: 1 };
  if (done < s.weights) return { phase: "weights", frac: done / s.weights };
  return { phase: "kv", frac: (done - s.weights) / (s.total - s.weights) };
}

export function streamCaption(runs: readonly Stream[], t: number): string {
  const parts = runs.map((r) => {
    const a = streamAt(r, t);
    if (a.phase === "done") return `${r.label} done in ${fmtMs(r.time)}`;
    if (a.phase === "weights")
      return `${r.label} ${pct(a.frac)} through its weights`;
    return `${r.label} reading the KV cache (${pct(a.frac)})`;
  });
  return `t = ${fmtMs(t)}: ${parts.join("; ")}.`;
}
