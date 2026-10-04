/**
 * The roofline port against (1) fixtures written by the Python simulator
 * (scripts/cost_reference.py), compared exactly, and (2) the published
 * table in Disaggregated_Inference_Sim/examples/results.md §1.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  DEVICES,
  H100_SXM,
  LINKS,
  LLAMA3_70B,
  LLAMA3_8B,
  MODELS,
  costModel,
  kvBytesPerToken,
  params,
  weightBytesTotal,
} from "@/lib/inference/costModel";

type Case = {
  model: keyof typeof MODELS;
  weight_bytes: number;
  kv_bytes: number;
  device: keyof typeof DEVICES;
  n_devices: number;
  kind: "prefill" | "decode";
  prompt_lens?: number[];
  ctx?: number;
  batch?: number;
  flops: number;
  bytes: number;
  time: number;
  bound: string;
};

const fx = JSON.parse(
  readFileSync(
    join(process.cwd(), "tests", "unit", "fixtures", "cost_model.json"),
    "utf-8",
  ),
) as {
  simulator_commit: string;
  kv_bytes_per_token: Record<string, number>;
  ridge_point: Record<string, number>;
  params: Record<string, number>;
  links: Record<string, { name: string; bandwidth: number; latency: number }>;
  cases: Case[];
};

describe("cost model port", () => {
  it(`matches every Python fixture exactly (${fx.cases.length} steps, simulator ${fx.simulator_commit.slice(0, 7)})`, () => {
    let power = 0;
    for (const c of fx.cases) {
      const m = {
        ...MODELS[c.model],
        weight_bytes: c.weight_bytes,
        kv_bytes: c.kv_bytes,
      };
      const cm = costModel(m, DEVICES[c.device], { nDevices: c.n_devices });
      const s =
        c.kind === "prefill"
          ? cm.prefill(c.prompt_lens!)
          : cm.decode(c.ctx!, c.batch!);
      expect(s.flops).toBe(c.flops);
      expect(s.bytes).toBe(c.bytes);
      expect(s.bound).toBe(c.bound);
      // The cube root in the power branch is the one documented tolerance.
      if (c.bound === "power") {
        power++;
        expect(Math.abs(s.time - c.time) / c.time).toBeLessThan(1e-12);
      } else {
        expect(s.time).toBe(c.time);
      }
    }
    expect(power).toBeGreaterThan(0);
  });

  it("matches the fixture's model and device constants", () => {
    expect(kvBytesPerToken(LLAMA3_8B)).toBe(fx.kv_bytes_per_token["llama3-8b"]);
    expect(kvBytesPerToken(LLAMA3_70B)).toBe(
      fx.kv_bytes_per_token["llama3-70b"],
    );
    expect(params(LLAMA3_8B)).toBe(fx.params["llama3-8b"]);
    expect(params(LLAMA3_70B)).toBe(fx.params["llama3-70b"]);
    expect(costModel(LLAMA3_8B, H100_SXM).ridgePoint).toBe(fx.ridge_point.h100);
    expect(costModel(LLAMA3_8B, DEVICES.a100).ridgePoint).toBe(
      fx.ridge_point.a100,
    );
  });

  // results.md §1, "after" columns (rounded as published).
  const ROWS: [
    string,
    () => ReturnType<ReturnType<typeof costModel>["decode"]>,
    number,
    number,
    string,
  ][] = [
    [
      "8B decode b=1 ctx 2048, 1xH100",
      () => costModel(LLAMA3_8B, H100_SXM).decode(2048, 1),
      15.278,
      6.201,
      "1.1",
    ],
    [
      "8B decode b=64 ctx 2048, 1xH100",
      () => costModel(LLAMA3_8B, H100_SXM).decode(2048 * 64, 64),
      32.198,
      12.514,
      "32.0",
    ],
    [
      "70B decode b=16 ctx 2300, 4xH100",
      () =>
        costModel(LLAMA3_70B, H100_SXM, { nDevices: 4 }).decode(2300 * 16, 16),
      151.068,
      14.592,
      "15.4",
    ],
    [
      "70B decode b=1 ctx 2048, 4xH100",
      () => costModel(LLAMA3_70B, H100_SXM, { nDevices: 4 }).decode(2048, 1),
      139.675,
      13.529,
      "1.0",
    ],
    [
      "70B prefill 2048, 4xH100",
      () => costModel(LLAMA3_70B, H100_SXM, { nDevices: 4 }).prefill([2048]),
      139.708,
      133.867,
      "2077.0",
    ],
    [
      "8B prefill 2048, 1xH100",
      () => costModel(LLAMA3_8B, H100_SXM).prefill([2048]),
      15.295,
      59.033,
      "2081.7",
    ],
  ];
  for (const [label, run, gb, ms, intensity] of ROWS) {
    it(`reproduces results.md §1: ${label}`, () => {
      const s = run();
      expect((s.bytes / 1e9).toFixed(3)).toBe(gb.toFixed(3));
      expect((s.time * 1e3).toFixed(3)).toBe(ms.toFixed(3));
      expect(s.intensity.toFixed(1)).toBe(intensity);
    });
  }

  it("reproduces the ridge point (203 FLOP/B) and the resident weights of results.md §1", () => {
    expect(costModel(LLAMA3_8B, H100_SXM).ridgePoint.toFixed(0)).toBe("203");
    expect((weightBytesTotal(LLAMA3_8B) / 1e9).toFixed(2)).toBe("16.06");
    expect((weightBytesTotal(LLAMA3_70B) / 1e9).toFixed(2)).toBe("141.10");
  });

  it("reports no KV room when the weights don't fit", () => {
    expect(costModel(LLAMA3_70B, H100_SXM).kvCapacityTokens()).toBe(0);
    expect(
      costModel(LLAMA3_70B, H100_SXM, { nDevices: 4 }).kvCapacityTokens(),
    ).toBeGreaterThan(0);
  });

  it("mixed step reduces to prefill and to decode exactly", () => {
    for (const [m, n] of [
      [LLAMA3_8B, 1],
      [LLAMA3_70B, 4],
    ] as const) {
      const cm = costModel(m, H100_SXM, { nDevices: n });
      expect(cm.mixed([[0, 2048]], 0, 0)).toEqual(cm.prefill([2048]));
      expect(cm.mixed([], 2048 * 16, 16)).toEqual(cm.decode(2048 * 16, 16));
      // Two chunks of one prompt cost more in total than one prefill
      // (weights read twice), but the attention FLOPs add up exactly.
      const a = cm.mixed([[0, 1024]], 0, 0);
      const b = cm.mixed([[1024, 2048]], 0, 0);
      const whole = cm.prefill([2048]);
      expect(a.flops + b.flops).toBe(whole.flops);
      expect(a.time + b.time).toBeGreaterThan(whole.time);
    }
  });

  it("copies the link table from hardware.py", () => {
    expect(Object.keys(LINKS).sort()).toEqual(Object.keys(fx.links).sort());
    for (const [k, l] of Object.entries(fx.links)) {
      expect(LINKS[k]).toEqual(l);
    }
  });
});
