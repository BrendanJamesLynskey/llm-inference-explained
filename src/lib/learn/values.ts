/**
 * The learning layer's numbers (brief 44). This site's chapters quote their
 * numbers as text (tests/unit/inference/chapterNumbers.test.ts and
 * tests/unit/disagg/chapterNumbers.test.ts check each against the code), so
 * the self-checks get theirs here instead: each value is computed at build
 * time by the same tested code the widgets run (the cost model, the KV and
 * attention formulas, the speculative-decoding closed forms, the hand-off
 * model), or read from the simulator's recorded results that the presets
 * carry (`PRESETS`, checked against the Python package's fixture).
 * A self-check cites one as `<Check v="…">`, a reveal prints one as
 * `<V v="…" />`; tests/unit/learn.test.ts checks every one.
 *
 * Values are SI (seconds, bytes, fractions); `formatValue` prints them in
 * the unit the chapters use.
 */
import { hbmFlash, hbmStandard } from "@/lib/inference/attentionKernels";
import {
  H100_SXM,
  LINKS,
  LLAMA3_70B,
  LLAMA3_8B,
  costModel,
  kvBytesPerToken,
  weightBytesTotal,
} from "@/lib/inference/costModel";
import { KV_PRESETS, kvValuesPerToken } from "@/lib/inference/kvMemory";
import { bubbleFraction } from "@/lib/inference/parallel";
import { decodeBreakdown, withFormats } from "@/lib/inference/quantisation";
import { expectedTokens, speedup } from "@/lib/inference/speculative";
import { handoff } from "@/lib/disagg/handoff";
import { PRESETS } from "@/lib/disagg/simulator";

export type Fmt =
  | "int"
  | "num"
  | "d1"
  | "d2"
  | "pct0"
  | "pct1"
  | "ms0"
  | "ms1"
  | "ms3"
  | "kib"
  | "mb1"
  | "gb2"
  | "tps"
  | "rps"
  | "raw";

const group = (v: number, d: number): string =>
  v.toLocaleString("en-GB", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });

/** Up to three significant figures, without trailing zeros. */
function trim(v: number): string {
  if (v === 0) return "0";
  return Number(v.toPrecision(3)).toLocaleString("en-GB", {
    maximumFractionDigits: 6,
  });
}

export function formatValue(v: number | string, fmt: Fmt = "num"): string {
  if (typeof v === "string") return v;
  switch (fmt) {
    case "int":
      return group(Math.round(v), 0);
    case "d1":
      return group(v, 1);
    case "d2":
      return group(v, 2);
    case "pct0":
      return `${group(v * 100, 0)}%`;
    case "pct1":
      return `${group(v * 100, 1)}%`;
    case "ms0":
      return `${group(v * 1e3, 0)} ms`;
    case "ms1":
      return `${group(v * 1e3, 1)} ms`;
    case "ms3":
      return `${group(v * 1e3, 3)} ms`;
    case "kib":
      return `${group(v / 1024, 0)} KiB`;
    case "mb1":
      return `${group(v / 1e6, 1)} MB`;
    case "gb2":
      return `${group(v / 1e9, 2)} GB`;
    case "tps":
      return `${group(Math.round(v), 0)} tok/s`;
    case "rps":
      return `${group(v, 2)} req/s`;
    case "raw":
      return String(v);
    default:
      return trim(v);
  }
}

/** A cell of a recorded results.md row, as SI (ms → s, % → fraction). */
function recorded(id: string, column: string, key: string): number {
  const p = PRESETS.find((x) => x.id === id);
  const c = p?.expected.find((e) => e.column === column && e.key === key);
  if (!c) throw new Error(`no recorded ${id} ${column}.${key}`);
  const n = Number(c.text.replace(/[^\d.]/g, ""));
  if (c.text.endsWith("ms")) return n / 1e3;
  if (c.text.endsWith("%")) return n / 100;
  return n;
}

const h100 = (m = LLAMA3_8B, nDevices = 1) =>
  costModel(m, H100_SXM, { nDevices });

/** Llama-3-8B decode at batch 1, 2,048 tokens of context (chapters 3, 8). */
const decode8b = (wBytes: number, kvB = 2) =>
  decodeBreakdown(withFormats(LLAMA3_8B, wBytes, kvB), H100_SXM, 1, 2048, 1);

const VALUES: Record<string, () => number> = {
  // chapter 1: one prefill pass of a 2,048-token prompt against one decode pass
  "ch01.prefillOverDecode": () => {
    const cm = h100();
    return cm.prefill([2048]).time / cm.decode(2048, 1).time;
  },
  // chapter 2: KV bytes per token (BF16) and GQA's saving
  "ch02.kv8b": () => kvBytesPerToken(LLAMA3_8B),
  "ch02.kv70b": () => kvBytesPerToken(LLAMA3_70B),
  "ch02.mhaOverGqa": () =>
    kvValuesPerToken(KV_PRESETS["llama3-8b"]!, "mha") /
    kvValuesPerToken(KV_PRESETS["llama3-8b"]!, "gqa"),
  // chapter 3: the roofline on the simulator's derated H100
  "ch03.ridge": () => h100().ridgePoint,
  "ch03.decode1.intensity": () => h100().decode(2048, 1).intensity,
  "ch03.decode1.tps": () => 1 / h100().decode(2048, 1).time,
  "ch03.decode64.tps": () => 64 / h100().decode(2048 * 64, 64).time,
  // chapter 6: one head's HBM traffic, 4,096 tokens, d = 64, 192 KiB SRAM
  "ch06.stdOverFlash": () =>
    hbmStandard(4096, 64) / hbmFlash(4096, 64, (192 * 1024) / 2),
  "ch06.scoresMiB8k": () => (8192 * 8192 * 2) / (1024 * 1024),
  // chapter 7: the widget's defaults (α 0.8, γ 4, c 0.05)
  "ch07.tokensPerPass": () => expectedTokens(0.8, 4),
  "ch07.speedup": () => speedup(0.8, 4, 0.05),
  // chapter 8: Llama-3-8B, batch 1, 2,048 context, one H100
  "ch08.bf16.tps": () => decode8b(2).tokensPerSecond,
  "ch08.int4.tps": () => decode8b(0.5).tokensPerSecond,
  "ch08.int4Speedup": () =>
    decode8b(0.5).tokensPerSecond / decode8b(2).tokensPerSecond,
  // chapter 9
  "ch09.weights70b": () => weightBytesTotal(LLAMA3_70B),
  "ch09.bubble4x4": () => bubbleFraction(4, 4),
  // chapter 10: 100 independent requests, each slow 1% of the time
  "ch10.pageSlow": () => 1 - 0.99 ** 100,
  // chapters 11 and 13: recorded results (results.md §4)
  "ch11.coloc.tpot99": () => recorded("sweep-4", "colocated", "tpot_p99"),
  "ch11.disagg.tpot99": () => recorded("sweep-4", "disagg", "tpot_p99"),
  "ch11.disagg.slo": () => recorded("sweep-4", "disagg", "slo"),
  "ch11.coloc.slo": () => recorded("sweep-4", "colocated", "slo"),
  "ch13.disagg8.slo": () => recorded("sweep-8", "disagg", "slo"),
  "ch13.disagg8.ttft99": () => recorded("sweep-8", "disagg", "ttft_p99"),
  // chapter 12: a 2,048-token Llama-3-8B prompt's hand-off
  "ch12.bytes": () => 2048 * kvBytesPerToken(LLAMA3_8B),
  "ch12.eth25": () =>
    handoff({
      model: LLAMA3_8B,
      devices: 1,
      prompt: 2048,
      link: "eth-25g",
      compression: "none",
      layerwise: false,
    }).transferTime,
  "ch12.eth25.rate": () =>
    handoff({
      model: LLAMA3_8B,
      devices: 1,
      prompt: 2048,
      link: "eth-25g",
      compression: "none",
      layerwise: false,
    }).linkSaturationRate,
  "ch12.fp8.tpot99": () => recorded("eth25-14-fp8", "disagg", "tpot_p99"),
  "ch12.none.tpot99": () => recorded("eth25-14-none", "disagg", "tpot_p99"),
  // chapter 14: Llama-3-70B hand-offs over 25 GbE
  "ch14.eth25.rate70b": () =>
    LINKS["eth-25g"]!.bandwidth / (2048 * kvBytesPerToken(LLAMA3_70B)),
};

export const LEARN_VALUE_NAMES = Object.keys(VALUES);

const CACHE = new Map<string, number>();

export function learnValue(name: string): number {
  const cached = CACHE.get(name);
  if (cached !== undefined) return cached;
  const f = VALUES[name];
  if (!f) throw new Error(`no learn value ${name}`);
  const v = f();
  if (!Number.isFinite(v)) throw new Error(`learn value ${name} is ${v}`);
  CACHE.set(name, v);
  return v;
}

/** The adapter's lookup: on this site every path is a learn value. */
export const lookup = learnValue;
