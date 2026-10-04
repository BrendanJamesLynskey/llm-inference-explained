import { describe, expect, it } from "vitest";

import {
  POLICIES,
  makeWorkload,
  simulateBatching,
  type BatchingConfig,
  type Request,
} from "@/lib/inference/batching";
import { H100_SXM, LLAMA3_8B, costModel } from "@/lib/inference/costModel";
import { summarise } from "@/lib/inference/metrics";

const cm = costModel(LLAMA3_8B, H100_SXM);
const base: Omit<BatchingConfig, "policy"> = {
  maxBatch: 16,
  chunk: 512,
  timeout: 0.05,
  cm,
};
const SLO = { ttft: 1, tpot: 0.025 };

describe("makeWorkload", () => {
  it("is deterministic per seed, with Poisson-ish arrivals at the mean rate", () => {
    const w = { rate: 4, n: 2000, prompt: 512, output: 128, cv: 0.5, seed: 3 };
    const a = makeWorkload(w);
    expect(makeWorkload(w)).toEqual(a);
    const span = a[a.length - 1]!.arrival;
    expect(a.length / span).toBeGreaterThan(3.7);
    expect(a.length / span).toBeLessThan(4.3);
    const meanOut = a.reduce((s, r) => s + r.output, 0) / a.length;
    expect(Math.abs(meanOut - 128) / 128).toBeLessThan(0.05);
  });

  it("fixes the lengths when cv = 0, and puts every arrival at 0 when rate = 0", () => {
    const a = makeWorkload({
      rate: 0,
      n: 5,
      prompt: 100,
      output: 10,
      cv: 0,
      seed: 1,
    });
    expect(
      a.every((r) => r.arrival === 0 && r.prompt === 100 && r.output === 10),
    ).toBe(true);
  });
});

describe("simulateBatching", () => {
  const reqs = makeWorkload({
    rate: 6,
    n: 300,
    prompt: 1024,
    output: 128,
    cv: 0.6,
    seed: 11,
  });

  for (const policy of POLICIES) {
    it(`${policy}: every request gets exactly its output tokens, in time order`, () => {
      const { records, steps, horizon } = simulateBatching(reqs, {
        ...base,
        policy,
      });
      expect(records).toHaveLength(reqs.length);
      for (const r of records) {
        expect(r.tokenTimes).toHaveLength(r.output);
        expect(r.tokenTimes[0]!).toBeGreaterThan(r.arrival);
        for (let i = 1; i < r.tokenTimes.length; i++) {
          expect(r.tokenTimes[i]!).toBeGreaterThan(r.tokenTimes[i - 1]!);
        }
      }
      expect(steps.every((s) => s.batch <= base.maxBatch + base.chunk)).toBe(
        true,
      );
      expect(horizon).toBeGreaterThanOrEqual(
        records[records.length - 1]!.tokenTimes.at(-1)!,
      );
    });
  }

  it("static and continuous agree when every request is identical and arrives at once", () => {
    const same: Request[] = makeWorkload({
      rate: 0,
      n: 8,
      prompt: 256,
      output: 32,
      cv: 0,
      seed: 1,
    });
    const s = simulateBatching(same, { ...base, policy: "static" });
    const c = simulateBatching(same, { ...base, policy: "continuous" });
    expect(c.records).toEqual(s.records);
  });

  it("continuous batching beats static on throughput and latency with mixed lengths", () => {
    const st = summarise(
      simulateBatching(reqs, { ...base, policy: "static" }).records,
      SLO,
    );
    const co = summarise(
      simulateBatching(reqs, { ...base, policy: "continuous" }).records,
      SLO,
    );
    expect(co.tokPerS).toBeGreaterThan(st.tokPerS);
    expect(co.ttft.p99).toBeLessThan(st.ttft.p99);
  });

  it("dynamic batching cuts static's wait at low load", () => {
    const light = makeWorkload({
      rate: 1,
      n: 60,
      prompt: 512,
      output: 64,
      cv: 0.5,
      seed: 5,
    });
    const st = summarise(
      simulateBatching(light, { ...base, policy: "static" }).records,
      SLO,
    );
    const dy = summarise(
      simulateBatching(light, { ...base, policy: "dynamic" }).records,
      SLO,
    );
    expect(dy.ttft.mean).toBeLessThan(st.ttft.mean);
  });

  it("chunked prefill removes the decode stalls behind long prompts", () => {
    const long = makeWorkload({
      rate: 4,
      n: 200,
      prompt: 4096,
      output: 128,
      cv: 0.5,
      seed: 9,
    });
    const co = summarise(
      simulateBatching(long, { ...base, policy: "continuous" }).records,
      SLO,
    );
    const ch = summarise(
      simulateBatching(long, { ...base, policy: "chunked" }).records,
      SLO,
    );
    expect(ch.itl.p99).toBeLessThan(co.itl.p99);
  });
});
