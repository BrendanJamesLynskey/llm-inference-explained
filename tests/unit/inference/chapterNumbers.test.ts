/**
 * Every number quoted in a chapter's prose is recomputed here by the tested
 * code, and the test also checks the chapter still quotes it. Change a model
 * and this fails until the prose is updated (house rule: every number comes
 * from a run).
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  A100_SXM,
  H100_SXM,
  LINKS,
  LLAMA3_70B,
  LLAMA3_8B,
  costModel,
  kvBytesPerToken,
  weightBytesTotal,
} from "@/lib/inference/costModel";
import { bubbleFraction, ppComm, tpComm } from "@/lib/inference/parallel";
import { decodeBreakdown, withFormats } from "@/lib/inference/quantisation";
import { expectedTokens, speedup } from "@/lib/inference/speculative";

const mdx = (slug: string): string =>
  readFileSync(
    join(process.cwd(), "content", "inference", `${slug}.mdx`),
    "utf-8",
  ).replace(/\s+/g, " ");
const gb = (b: number, dp = 2): string => (b / 1e9).toFixed(dp);
const ms = (s: number, dp = 3): string => (s * 1e3).toFixed(dp);
const quoted = (slug: string, ...xs: string[]): void => {
  const t = mdx(slug);
  for (const x of xs) expect(t, `${slug} should quote "${x}"`).toContain(x);
};

const cm8 = costModel(LLAMA3_8B, H100_SXM);

describe("chapter 2: KV cache", () => {
  it("bytes per token and the 16 × 8,192 example", () => {
    expect(kvBytesPerToken(LLAMA3_8B)).toBe(131072);
    expect(kvBytesPerToken(LLAMA3_70B) / 1024).toBe(320);
    expect(gb(16 * 8192 * kvBytesPerToken(LLAMA3_8B))).toBe("17.18");
    expect(gb(weightBytesTotal(LLAMA3_8B))).toBe("16.06");
    quoted(
      "02-kv-cache",
      "131{,}072",
      "128 KiB per token",
      "320 KiB per token",
      "17.18 GB",
      "16.06 GB",
    );
  });
});

describe("chapter 3: roofline", () => {
  it("ridge points, the decode and prefill steps of results.md §1", () => {
    expect(cm8.ridgePoint.toFixed(0)).toBe("203");
    expect(costModel(LLAMA3_8B, A100_SXM).ridgePoint.toFixed(0)).toBe("105");
    const d1 = cm8.decode(2048, 1);
    expect(gb(d1.bytes)).toBe("15.28");
    expect((d1.flops / 1e9).toFixed(0)).toBe("16");
    expect(d1.intensity.toFixed(1)).toBe("1.1");
    expect(ms(d1.time)).toBe("6.201");
    expect((1 / d1.time).toFixed(0)).toBe("161");
    const p = cm8.prefill([2048]);
    expect(p.intensity.toFixed(1)).toBe("2081.7");
    expect(ms(p.time)).toBe("59.033");
    const d64 = cm8.decode(2048 * 64, 64);
    expect(ms(d64.time)).toBe("12.514");
    expect(Math.round(64 / d64.time).toLocaleString("en-GB")).toBe("5,114");
    expect(d64.intensity.toFixed(0)).toBe("32");
    quoted(
      "03-roofline",
      "203 FLOP per byte",
      "about 105",
      "15.28 GB",
      "about 16 billion FLOPs",
      "1.1 FLOP/B",
      "6.201 ms",
      "about 161 tokens per second",
      "2,081.7 FLOP/B",
      "59.033 ms",
      "12.514 ms",
      "about 5,114 tokens",
      "32 FLOP/B",
    );
  });
});

describe("chapter 6: attention kernels", () => {
  it("an 8,192-token score matrix at 16 bits is 128 MiB", () => {
    expect((8192 * 8192 * 2) / 2 ** 20).toBe(128);
    quoted("06-attention-kernels", "128 MiB");
  });
});

describe("chapter 7: speculative decoding", () => {
  it("α = 0.8, γ = 4, c = 0.05", () => {
    expect(expectedTokens(0.8, 4).toFixed(2)).toBe("3.36");
    expect(speedup(0.8, 4, 0.05).toFixed(2)).toBe("2.80");
    quoted("07-speculative-decoding", "3.36 tokens per pass", "2.80×");
  });
});

describe("chapter 8: quantisation", () => {
  it("batch 1, context 2,048 table", () => {
    const row = [2, 1, 0.5].map((w) =>
      decodeBreakdown(withFormats(LLAMA3_8B, w, 2), H100_SXM, 1, 2048, 1),
    );
    expect(row.map((r) => ms(r.time))).toEqual(["6.201", "3.400", "2.000"]);
    expect(row.map((r) => r.tokensPerSecond.toFixed(0))).toEqual([
      "161",
      "294",
      "500",
    ]);
    quoted(
      "08-quantisation",
      "6.201 ms, 161 tok/s",
      "3.400 ms, 294 tok/s",
      "2.000 ms, 500 tok/s",
    );
  });
  it("batch 32, context 8,192: the cache outweighs the weights", () => {
    const [a, b, c] = [
      [2, 2],
      [2, 1],
      [1, 1],
    ].map(([w, k]) =>
      decodeBreakdown(withFormats(LLAMA3_8B, w!, k!), H100_SXM, 1, 8192, 32),
    );
    expect(gb(a!.kvBytes)).toBe("34.36");
    expect(gb(a!.weightBytes)).toBe("15.01");
    expect([a, b, c].map((x) => ms(x!.time, 2))).toEqual([
      "18.92",
      "12.51",
      "9.71",
    ]);
    expect([a, b, c].every((x) => x!.bound === "memory")).toBe(true);
    // Cache passes the weights at ≈ 115,000 tokens of context.
    expect(Math.round(a!.weightBytes / kvBytesPerToken(LLAMA3_8B) / 1000)).toBe(
      115,
    );
    quoted(
      "08-quantisation",
      "34.36 GB",
      "15.01 GB",
      "18.92 ms to 12.51 ms",
      "9.71 ms",
      "about 115,000 tokens",
    );
  });
});

describe("chapter 9: parallelism", () => {
  it("70B weights, TP and PP communication, bubbles", () => {
    expect(gb(weightBytesTotal(LLAMA3_70B))).toBe("141.10");
    const step = costModel(LLAMA3_70B, H100_SXM, { nDevices: 4 }).decode(
      2048 * 16,
      16,
    );
    expect(ms(step.time, 2)).toBe("14.47");
    const nv = tpComm(LLAMA3_70B, 4, LINKS.nvlink4!, 16);
    expect(ms(nv.time, 2)).toBe("4.94");
    expect(ms(nv.messages * LINKS.nvlink4!.latency, 2)).toBe("4.80");
    expect(ms(tpComm(LLAMA3_70B, 4, LINKS["eth-100g"]!, 16).time, 2)).toBe(
      "24.23",
    );
    const pp = ppComm(LLAMA3_70B, 4, LINKS["ib-ndr"]!, 16);
    expect(pp.bytes / 1024).toBe(256);
    expect((pp.time * 1e6).toFixed(0)).toBe("46");
    expect((bubbleFraction(4, 4) * 100).toFixed(0)).toBe("43");
    expect((bubbleFraction(4, 16) * 100).toFixed(0)).toBe("16");
    quoted(
      "09-parallelism",
      "141.10 GB",
      "14.47 ms",
      "4.94 ms",
      "24.23 ms",
      "(4.80 ms)",
      "256 KiB",
      "about 46 µs",
      "idle 43%",
      "16 micro-batches 16%",
    );
  });
});

describe("chapter 10: serving metrics", () => {
  it("the fan-out tail", () => {
    expect(((1 - 0.99 ** 100) * 100).toFixed(0)).toBe("63");
    quoted("10-serving-metrics", "63% of page loads");
  });
});
