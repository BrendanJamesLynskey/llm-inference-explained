/**
 * The landing hero: public/disagg/hero.json is two cases of the parity
 * fixture (the Python package's own runs), the vendored engine reproduces
 * their timestamps exactly, and the lanes and frames the hero draws are
 * read off those timestamps.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { runPlain, type Row, type SimConfig } from "@/lib/disagg/engine";
import {
  HERO_LANES,
  HERO_URL,
  heroCaption,
  heroHorizon,
  heroLanes,
  heroTime,
  laneFrame,
  phaseAt,
  type HeroFile,
} from "@/lib/disagg/hero";

const read = <T>(p: string): T =>
  JSON.parse(readFileSync(join(process.cwd(), p), "utf-8")) as T;
const hero = read<HeroFile>(`public${HERO_URL}`);
const parity = read<{
  commit: string;
  cases: {
    name: string;
    cfg: SimConfig;
    rows: Row[];
    stamps: (number | null)[][];
  }[];
}>("tests/unit/fixtures/disagg_parity.json");

const STAMPS = [
  "prefillStart",
  "firstToken",
  "kvStart",
  "kvReady",
  "decodeStart",
  "finish",
] as const;

describe("the landing hero's runs", () => {
  it("are parity cases, recorded at the vendored commit", () => {
    expect(hero.commit).toBe(parity.commit);
    expect(hero.modes.map((m) => m.key)).toEqual(["colocated", "disagg"]);
    for (const m of hero.modes) {
      const c = parity.cases.find((x) => x.name === m.name)!;
      expect(m.cfg).toEqual(c.cfg);
      expect(hero.rows).toEqual(c.rows);
    }
  });
  for (const m of hero.modes)
    it(`${m.name}: the engine gives the Python package's timestamps`, () => {
      const c = parity.cases.find((x) => x.name === m.name)!;
      const r = runPlain(m.cfg, hero.rows);
      r.result.reqs.forEach((q, i) =>
        STAMPS.forEach((k, j) => expect(q[k]).toBe(c.stamps[i]![j])),
      );
    });
});

describe("lanes and frames", () => {
  const runs = Object.fromEntries(
    hero.modes.map((m) => [m.key, runPlain(m.cfg, hero.rows)]),
  );
  const co = heroLanes(runs.colocated!.result.reqs);
  const dis = heroLanes(runs.disagg!.result.reqs);
  it("the first requests, with their stamps", () => {
    expect(co).toHaveLength(HERO_LANES);
    const q = runs.disagg!.result.reqs[0]!;
    expect(dis[0]!.firstToken).toBe(q.firstToken);
    expect(dis[0]!.kvReady).toBe(q.kvReady);
    expect(co[0]!.kvReady).toBeNull();
  });
  it("colocated decodes stall behind prefills; disaggregated ones barely", () => {
    const n = (l: typeof co) => l.reduce((a, x) => a + x.stalls.length, 0);
    expect(n(co)).toBeGreaterThan(3 * n(dis));
    const worst = (l: typeof co) =>
      Math.max(...l.flatMap((x) => x.stalls.map(([a, b]) => b - a)));
    expect(worst(co)).toBeGreaterThan(5 * worst(dis));
    // every stall sits inside its request's decode
    for (const l of [...co, ...dis])
      for (const [a, b] of l.stalls) {
        expect(a).toBeGreaterThanOrEqual(l.firstToken);
        expect(b).toBeLessThanOrEqual(l.finish + 1e-9);
      }
  });
  it("the metrics the landing quotes: split, ITL p99 falls and TTFT p99 rises", () => {
    const c = runs.colocated!.metrics;
    const d = runs.disagg!.metrics;
    expect(d.itl_p99).toBeLessThan(c.itl_p99 / 5);
    expect(d.ttft_p99).toBeGreaterThan(c.ttft_p99);
  });
  it("phases at key times", () => {
    const l = dis[0]!;
    expect(phaseAt(l, l.arrival - 1e-3)).toBe("not yet");
    expect(phaseAt(l, (l.prefillStart + l.firstToken) / 2)).toBe("prefill");
    expect(phaseAt(l, (l.kvStart! + l.kvReady!) / 2)).toBe("hand-off");
    expect(phaseAt(l, (l.decodeStart + l.finish) / 2)).toBe("decode");
    expect(phaseAt(l, l.finish)).toBe("done");
    const queued = co.find((x) => x.prefillStart > x.arrival)!;
    expect(phaseAt(queued, queued.arrival)).toBe("queued");
    const h = heroHorizon(co, dis);
    expect(h).toBe(Math.max(...[...co, ...dis].map((x) => x.finish)));
    expect(heroTime(0, 50, h)).toBe(0);
    expect(heroTime(49, 50, h)).toBe(h);
    expect(heroTime(3, 1, h)).toBe(h);
    const end = laneFrame(co, h);
    expect(end.done).toBe(HERO_LANES);
    expect(end.stalled).toBe(co.reduce((a, x) => a + x.stalls.length, 0));
    expect(laneFrame(co, 0).done).toBe(0);
    expect(heroCaption(end, laneFrame(dis, h))).toMatch(
      /^t = \d\.\d\d s\. Colocated: 8 of 8 done, \d+ decode stalls so far\. Disaggregated: 8 of 8 done, \d+ stalls\.$/,
    );
  });
});
