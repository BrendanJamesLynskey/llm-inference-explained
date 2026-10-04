import { describe, expect, it } from "vitest";

import {
  dist,
  itlsOf,
  meetsSlo,
  percentileSorted,
  summarise,
  timeAverageInSystem,
  tpotOf,
  ttftOf,
  type RequestRecord,
} from "@/lib/inference/metrics";
import { makeWorkload, simulateBatching } from "@/lib/inference/batching";
import { H100_SXM, LLAMA3_8B, costModel } from "@/lib/inference/costModel";

const R = (
  id: number,
  arrival: number,
  tokenTimes: number[],
): RequestRecord => ({
  id,
  arrival,
  prompt: 10,
  output: tokenTimes.length,
  tokenTimes,
});

describe("percentiles", () => {
  it("interpolate linearly like numpy", () => {
    const s = [1, 2, 3, 4];
    expect(percentileSorted(s, 50)).toBe(2.5);
    expect(percentileSorted(s, 0)).toBe(1);
    expect(percentileSorted(s, 100)).toBe(4);
    // numpy.percentile([1,2,3,4], 90) = 3.7
    expect(percentileSorted(s, 90)).toBeCloseTo(3.7, 12);
    expect(percentileSorted([], 50)).toBeNaN();
  });

  it("dist reports mean and max", () => {
    const d = dist([3, 1, 2]);
    expect(d.mean).toBe(2);
    expect(d.max).toBe(3);
    expect(d.p50).toBe(2);
    expect(dist([]).mean).toBeNaN();
  });
});

describe("per-request metrics", () => {
  const r = R(0, 1.0, [1.5, 1.6, 1.8, 1.9]);
  it("TTFT, TPOT and ITL", () => {
    expect(ttftOf(r)).toBe(0.5);
    expect(tpotOf(r)).toBeCloseTo(0.4 / 3, 12);
    expect(itlsOf(r).map((x) => +x.toFixed(6))).toEqual([0.1, 0.2, 0.1]);
    expect(tpotOf(R(1, 0, [1]))).toBeNaN();
  });
  it("SLOs need both TTFT and TPOT", () => {
    expect(meetsSlo(r, { ttft: 0.5, tpot: 0.2 })).toBe(true);
    expect(meetsSlo(r, { ttft: 0.4, tpot: 0.2 })).toBe(false);
    expect(meetsSlo(r, { ttft: 0.5, tpot: 0.1 })).toBe(false);
    expect(meetsSlo(R(1, 0, [0.1]), { ttft: 0.5, tpot: 0 })).toBe(true);
  });
});

describe("Little's law", () => {
  it("time-average number in system equals λ·W on a hand-made trace", () => {
    const recs = [R(0, 0, [1, 2]), R(1, 1, [3]), R(2, 2, [2.5, 4])];
    // In system: [0,2) one... integrate by hand: r0 0–2, r1 1–3, r2 2–4 → area 6.
    expect(timeAverageInSystem(recs, 4)).toBe(1.5);
    const s = summarise(recs, { ttft: 10, tpot: 10 });
    expect(s.little.L).toBeCloseTo(s.little.lambdaW, 12);
  });

  it("holds on a simulated run, measured independently by an event sweep", () => {
    const reqs = makeWorkload({
      rate: 5,
      n: 400,
      prompt: 512,
      output: 96,
      cv: 0.5,
      seed: 2,
    });
    const { records } = simulateBatching(reqs, {
      policy: "continuous",
      maxBatch: 32,
      chunk: 512,
      timeout: 0,
      cm: costModel(LLAMA3_8B, H100_SXM),
    });
    const s = summarise(records, { ttft: 1, tpot: 0.025 });
    expect(Math.abs(s.little.L - s.little.lambdaW) / s.little.L).toBeLessThan(
      1e-9,
    );
    expect(s.goodput).toBeGreaterThan(0);
    expect(s.sloAttain).toBeLessThanOrEqual(1);
  });

  it("goodput is NaN with a single request", () => {
    expect(summarise([R(0, 0, [1])], { ttft: 2, tpot: 1 }).goodput).toBeNaN();
  });
});
