import { describe, expect, it } from "vitest";

import {
  H100_SXM,
  LINKS,
  LLAMA3_70B,
  LLAMA3_8B,
  costModel,
} from "@/lib/inference/costModel";
import {
  bubbleFraction,
  epComm,
  ppComm,
  ringAllReduce,
  tpComm,
} from "@/lib/inference/parallel";
import { decodeBreakdown, withFormats } from "@/lib/inference/quantisation";

const nv = LINKS.nvlink4!;
const eth = LINKS["eth-100g"]!;

describe("parallelism communication", () => {
  it("ring all-reduce: 2(n−1)/n of the message, 2(n−1) hops", () => {
    const c = ringAllReduce(1e6, 4, nv);
    expect(c.bytes).toBe(1.5e6);
    expect(c.messages).toBe(6);
    expect(c.time).toBeCloseTo(1.5e6 / nv.bandwidth + 6 * nv.latency, 15);
    expect(ringAllReduce(1e6, 1, nv).time).toBe(0);
  });

  it("tensor parallelism: two all-reduces per layer", () => {
    const one = ringAllReduce(16 * 8192 * 2, 4, nv);
    const c = tpComm(LLAMA3_70B, 4, nv, 16);
    expect(c.bytes).toBe(160 * one.bytes);
    expect(c.messages).toBe(160 * 6);
    expect(tpComm(LLAMA3_70B, 1, nv, 16)).toEqual({
      bytes: 0,
      time: 0,
      messages: 0,
    });
  });

  it("TP on 100 GbE costs far more than on NVLink, per decode step", () => {
    const a = tpComm(LLAMA3_70B, 4, nv, 16).time;
    const b = tpComm(LLAMA3_70B, 4, eth, 16).time;
    expect(b / a).toBeGreaterThan(4);
    // Decode-sized messages are tiny, so on NVLink the per-message latency,
    // not the bandwidth, is most of the cost; and it is a real share of a
    // 4xH100 decode step (≈ 14.6 ms, results.md §1).
    const c = tpComm(LLAMA3_70B, 4, nv, 16);
    expect(c.messages * nv.latency).toBeGreaterThan(0.9 * c.time);
    const step = costModel(LLAMA3_70B, H100_SXM, { nDevices: 4 }).decode(
      2300 * 16,
      16,
    ).time;
    expect(a / step).toBeGreaterThan(0.1);
    expect(a / step).toBeLessThan(0.5);
  });

  it("pipeline: one activation per boundary, bubble (n−1)/(m+n−1)", () => {
    const c = ppComm(LLAMA3_8B, 4, nv, 8);
    expect(c.bytes).toBe(8 * 4096 * 2);
    expect(c.time).toBeCloseTo(3 * (c.bytes / nv.bandwidth + nv.latency), 15);
    expect(bubbleFraction(4, 1)).toBe(0.75);
    expect(bubbleFraction(4, 12)).toBeCloseTo(3 / 15, 15);
    expect(ppComm(LLAMA3_8B, 1, nv, 8).time).toBe(0);
  });

  it("expert parallelism: 2·k copies per token per MoE layer, (n−1)/n leave", () => {
    const c = epComm(4096, 1, 2, 8, nv, 10);
    expect(c.bytes).toBe((2 * 10 * 2 * 4096 * 2 * 7) / 8);
    expect(c.messages).toBe(14);
    expect(epComm(4096, 1, 2, 1, nv, 10).bytes).toBe(0);
  });
});

describe("quantisation", () => {
  it("BF16 breakdown adds up to the cost model's step", () => {
    const b = decodeBreakdown(LLAMA3_8B, H100_SXM, 1, 2048, 1);
    const s = costModel(LLAMA3_8B, H100_SXM).decode(2048, 1);
    expect(b.weightBytes + b.kvBytes).toBe(s.bytes);
    expect(b.time).toBe(s.time);
    // results.md §1: 6.201 ms.
    expect((b.time * 1e3).toFixed(3)).toBe("6.201");
  });

  it("8-bit and 4-bit weights cut a memory-bound decode step nearly in proportion", () => {
    const bf = decodeBreakdown(LLAMA3_8B, H100_SXM, 1, 2048, 1);
    const i8 = decodeBreakdown(
      withFormats(LLAMA3_8B, 1, 2),
      H100_SXM,
      1,
      2048,
      1,
    );
    const i4 = decodeBreakdown(
      withFormats(LLAMA3_8B, 0.5, 2),
      H100_SXM,
      1,
      2048,
      1,
    );
    expect(i8.weightBytes).toBeCloseTo(bf.weightBytes / 2, 0);
    expect(i8.time / bf.time).toBeLessThan(0.6);
    expect(i4.time / bf.time).toBeLessThan(0.35);
    expect(i4.residentWeightBytes).toBe(bf.residentWeightBytes / 4);
  });

  it("KV quantisation matters most at long context and large batch", () => {
    const a = decodeBreakdown(LLAMA3_8B, H100_SXM, 1, 8192, 64);
    const b = decodeBreakdown(
      withFormats(LLAMA3_8B, 2, 1),
      H100_SXM,
      1,
      8192,
      64,
    );
    expect(a.kvBytes).toBeGreaterThan(a.weightBytes);
    expect(b.kvBytes).toBe(a.kvBytes / 2);
    expect(b.time / a.time).toBeLessThan(0.75);
  });
});
