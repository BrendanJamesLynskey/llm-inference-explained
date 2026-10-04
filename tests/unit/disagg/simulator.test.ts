/**
 * The live simulator's presets: each one, run through the same
 * controls → configuration path the widget uses, must print exactly the
 * cells of its results.md line (from the fixture, not retyped).
 */
import { describe, expect, it } from "vitest";

import { FORMATS, run, type Metrics } from "@/lib/disagg/engine";
import {
  DEFAULT_CONTROLS,
  PRESETS,
  colocatedDevice,
  configs,
} from "@/lib/disagg/simulator";

import { results, workload } from "./helpers";

const byId = new Map(results.rows.map((r) => [r.id, r]));

describe("simulator presets reproduce results.md", () => {
  for (const p of PRESETS) {
    it(`${p.id}: ${p.label}`, () => {
      const row = byId.get(p.id);
      expect(row, `fixture row ${p.id}`).toBeDefined();
      expect(p.line).toBe(row!.line);
      expect(p.section).toBe(row!.section);
      expect(p.controls.rate).toBe(row!.rate);
      // every fixture cell is shown by the preset, in the same format
      const column = (run: string) =>
        run === "colocated" ? "colocated" : "disagg";
      expect(
        p.expected.map((e) => `${e.column}.${e.key}=${e.text}`).sort(),
      ).toEqual(
        row!.cells
          .filter((c) =>
            [
              "ttft_p99",
              "tpot_p99",
              "slo",
              "j_tok",
              "link_util",
              "handoff_p99",
            ].includes(c.key),
          )
          .map((c) => `${column(c.run)}.${c.key}=${c.text}`)
          .sort(),
      );
      const rows = workload(p.controls.rate).rows;
      const cfg = configs(p.controls);
      const m: Record<string, Metrics> = {
        colocated: run(cfg.colocated, rows),
        disagg: run(cfg.disagg, rows),
      };
      for (const e of p.expected)
        expect(
          FORMATS[e.fmt](m[e.column]![e.key] as number),
          `${p.id} ${e.column}.${e.key}`,
        ).toBe(e.text);
    });
  }
});

describe("controls → configurations", () => {
  it("colocated runs the same number of instances, without compression", () => {
    const c = configs({
      ...DEFAULT_CONTROLS,
      nPrefill: 2,
      nDecode: 1,
      compression: "fp8",
    });
    expect(c.colocated.nColocated).toBe(3);
    expect(c.colocated.mode).toBe("colocated");
    expect(c.colocated.kvCompress).toBeUndefined();
    expect(c.disagg.kvCompress).toBe("fp8");
    expect(c.disagg.kvCompressAt).toBe("transit");
  });

  it("optical prefill maps to the optical-fft device; colocated stays on GPUs", () => {
    const base = {
      ...DEFAULT_CONTROLS,
      model: "llama3-8b-hyena-circ" as const,
    };
    const d = configs({ ...base, prefillDevice: "optical-default" });
    expect(d.disagg.prefillDevice).toBe("optical-fft");
    expect(d.disagg.engine).toBeUndefined();
    expect(d.disagg.lmHead).toBe("last");
    expect(d.colocated.device).toBe("h100");
    const o = configs({ ...base, prefillDevice: "optical-optimistic" });
    expect(o.disagg.engine).toEqual({
      enob: 11,
      maskRate: 20000,
      overlap: true,
    });
  });

  it("the colocated pool shares the GPU when both pools use it", () => {
    expect(
      colocatedDevice({
        ...DEFAULT_CONTROLS,
        prefillDevice: "a100",
        decodeDevice: "a100",
      }),
    ).toBe("a100");
    expect(
      colocatedDevice({
        ...DEFAULT_CONTROLS,
        prefillDevice: "a100",
        decodeDevice: "h100",
      }),
    ).toBe("h100");
  });

  it("explicit identical pools equal the plain --device run (results.md §10)", () => {
    const rows = workload(8).rows;
    const c = configs({ ...DEFAULT_CONTROLS, model: "llama3-8b", rate: 8 });
    const hh = byId.get("hetero-hh")!;
    expect(run(c.disagg, rows)).toEqual(run(hh.runs.run!, rows));
  });
});
