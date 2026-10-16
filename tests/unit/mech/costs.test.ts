/**
 * Copied from Inference Trade-offs Explained's tests/unit/costs.test.ts at
 * 239d29b with its mechanism animations (brief 41): the ring and pipeline
 * animations call the vendored engine's cost model live: every step
 * it prices here equals the simulator's Python cost model
 * (cost_model.json, from scripts/mechanisms_reference.py), bit for bit.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  bytesPer,
  formatSteps,
  native,
  ppSteps,
  tpSteps,
} from "@/lib/tradeoffs/costs";
import type { StepCost } from "@/lib/tradeoffs/engine";

type Row = [number, number, number, string, number, number, number];
const J = JSON.parse(
  readFileSync(
    join(process.cwd(), "public/tradeoffs/mechanisms/cost_model.json"),
    "utf-8",
  ),
) as {
  tp: {
    tp: number;
    decode_b1: Row;
    decode_b64: Row;
    prefill_8192: Row;
    kv_tokens: number;
  }[];
  pp: {
    tp: number;
    pp: number;
    mb: number | null;
    prefill: Row;
    decode: Row;
  }[];
  formats: {
    device: string;
    weights: string;
    kv: string;
    compute: string;
    speedup: number;
    decode_b1: Row;
    decode_b64: Row;
    prefill_8192: Row;
    kv_tokens: number;
  }[];
  quant_bytes: Record<string, number>;
};
const row = (c: StepCost): Row => [
  c.time,
  c.flops,
  c.bytes,
  c.bound,
  c.ec,
  c.em,
  c.commT ?? 0,
];

describe("the engine's cost model equals the Python cost model", () => {
  it("tensor parallelism, TP2, TP4, TP8", () => {
    for (const t of J.tp) {
      const s = tpSteps(t.tp);
      expect(row(s.decodeB1)).toEqual(t.decode_b1);
      expect(row(s.decodeB64)).toEqual(t.decode_b64);
      expect(row(s.prefill8192)).toEqual(t.prefill_8192);
      expect(s.kvTokens).toBe(t.kv_tokens);
    }
  });
  it("pipeline parallelism and micro-batches", () => {
    for (const p of J.pp) {
      const s = ppSteps(p.tp, p.pp, p.mb);
      expect(row(s.prefill)).toEqual(p.prefill);
      expect(row(s.decode)).toEqual(p.decode);
    }
  });
  it("storage and compute formats on H100 and B200", () => {
    for (const f of J.formats) {
      const s = formatSteps(f.device, f.weights, f.kv, f.compute);
      expect(s.speedup).toBe(f.speedup);
      expect(row(s.decodeB1)).toEqual(f.decode_b1);
      expect(row(s.decodeB64)).toEqual(f.decode_b64);
      expect(row(s.prefill8192)).toEqual(f.prefill_8192);
      expect(s.kvTokens).toBe(f.kv_tokens);
    }
  });
  it("bytes per value and native units", () => {
    for (const [k, v] of Object.entries(J.quant_bytes))
      expect(bytesPer(k)).toBe(v);
    expect(native("h100", "fp8")).toBe(true);
    expect(native("h100", "fp4")).toBe(false);
    expect(native("b200", "fp4")).toBe(true);
    expect(native("h100", "bf16")).toBe(true);
  });
});
