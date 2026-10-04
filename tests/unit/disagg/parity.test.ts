/**
 * Engine parity: the vendored engine against the Python package at the same
 * commit (fixtures from scripts/disagg_reference.py), the way the simulator's
 * own tests compare its JS port. Every request's six timestamps, every
 * instance's energy counters and the link's counters must be identical.
 * The one documented exception: when a power cap binds hard enough, the
 * cost model solves a cubic for the clock with cube roots, which can differ
 * by an ulp between libms (Python's `x ** (1/3)` vs V8's Math.pow), so that
 * configuration is compared to 1e-9 relative, as the simulator's own tests
 * do. (Measured at the pinned commit: 105 of its 1,200 timestamps differ, by
 * at most 2.2e-16 relative; the other twelve configurations are identical.)
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { engine, type Row, type SimConfig } from "@/lib/disagg/engine";

type Case = {
  name: string;
  cfg: SimConfig;
  rows: Row[];
  stamps: (number | null)[][];
  inst: number[][];
  link: number[];
  capped: boolean;
  summary: Record<string, number>;
};
const fixture = JSON.parse(
  readFileSync(
    join(process.cwd(), "tests/unit/fixtures/disagg_parity.json"),
    "utf-8",
  ),
) as { commit: string; cases: Case[] };

function close(a: number | null, b: number | null, tol: number): boolean {
  if (a === null || b === null) return a === b;
  return tol === 0
    ? a === b
    : Math.abs(a - b) <= tol * Math.max(1, Math.abs(a));
}

const CUBE_ROOT = new Set(["70B colocated, cap 300 W"]);

describe("vendored engine = Python package, request by request", () => {
  it("covers thirteen configurations, all but one compared bit for bit", () => {
    expect(fixture.cases).toHaveLength(13);
    expect(fixture.cases.filter((c) => !CUBE_ROOT.has(c.name))).toHaveLength(
      12,
    );
  });

  for (const c of fixture.cases) {
    it(c.name, () => {
      const tol = CUBE_ROOT.has(c.name) ? 1e-9 : 0;
      const r = engine.simulate(c.cfg, c.rows);
      const s = engine.summarise(r);
      r.reqs.forEach((q, i) => {
        const js = [
          q.prefillStart,
          q.firstToken,
          q.kvStart,
          q.kvReady,
          q.decodeStart,
          q.finish,
        ];
        js.forEach((x, k) =>
          expect(
            close(c.stamps[i]![k]!, x, tol),
            `${c.name} req ${i} stamp ${k}`,
          ).toBe(true),
        );
      });
      r.insts.forEach((inst, i) => {
        const js = [inst.ec, inst.em, inst.oj, inst.busy, inst.peakW];
        js.forEach((x, k) =>
          expect(
            close(c.inst[i]![k]!, x, tol),
            `${c.name} inst ${i} field ${k}`,
          ).toBe(true),
        );
      });
      const l = r.link;
      [l.energy, l.transitJ, l.bytes, l.busy, l.wait].forEach((x, k) =>
        expect(close(c.link[k]!, x, tol), `${c.name} link ${k}`).toBe(true),
      );
      // percentiles of identical inputs are identical; J/token sums energies
      // in a different order from Python's sum(), so it gets 1e-12
      expect(close(c.summary.ttft_p99!, s.ttft.p99, tol)).toBe(true);
      expect(close(c.summary.tpot_p99!, s.tpot.p99, tol)).toBe(true);
      expect(close(c.summary.itl_p99!, s.itl.p99, tol)).toBe(true);
      expect(s.sloAttain).toBe(c.summary.slo);
      expect(s.energy.jPerTok).toBeCloseTo(c.summary.j_tok!, 12);
    });
  }
});
