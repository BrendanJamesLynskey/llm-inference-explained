/**
 * The results.md rows chapters 11–14 quote, reproduced by the vendored
 * engine on the very workloads Python used. scripts/disagg_reference.py
 * checked that each value, formatted as examples/results.py formats it,
 * appears in the recorded results.md line; here the engine must produce the
 * same values and the same text.
 */
import { describe, expect, it } from "vitest";

import { FORMATS, run, type Metrics } from "@/lib/disagg/engine";

import { results, workload } from "./helpers";

describe("live engine reproduces examples/results.md", () => {
  it("covers sections 3, 4, 10, 11, 14 and 15", () => {
    expect(new Set(results.rows.map((r) => r.section))).toEqual(
      new Set(["3", "4", "10", "11", "14", "15"]),
    );
  });

  for (const row of results.rows) {
    it(`§${row.section} ${row.id}`, () => {
      const wl = workload(row.rate);
      expect(wl.commit).toBe(results.commit);
      const js: Record<string, Metrics> = {};
      for (const [k, cfg] of Object.entries(row.runs))
        js[k] = run(cfg, wl.rows);
      for (const [k, py] of Object.entries(row.python)) {
        const m = js[k]!;
        // order statistics of identical timestamps: identical
        for (const key of [
          "ttft_p99",
          "tpot_p99",
          "itl_p50",
          "itl_p99",
          "itl_max",
          "handoff_p99",
          "slo",
          "goodput",
        ] as const)
          expect(m[key], `${row.id} ${k}.${key}`).toBe(py[key]);
        expect(m.hotspot).toBe(py.hotspot);
        // sums of energies and stage times: Python's sum() is compensated
        for (const key of ["avg_w", "j_tok", "kv_share", "link_util"] as const)
          expect(
            m[key] / ((py[key] as number) || 1),
            `${row.id} ${k}.${key}`,
          ).toBeCloseTo(py[key] === 0 ? m[key] : 1, 12);
      }
      for (const c of row.cells) {
        const text = FORMATS[c.fmt](
          js[c.run]![c.key as keyof Metrics] as number,
        );
        expect(text).toBe(c.text);
        expect(row.line).toContain(text);
      }
    });
  }
});
