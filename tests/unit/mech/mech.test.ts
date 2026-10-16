/**
 * The mechanism animations vendored from Inference Trade-offs Explained
 * (brief 41; src/lib/tradeoffs/VENDORED.json), tested here as they are
 * tested there: this file is that site's tests/unit/mech.test.ts at
 * 239d29b, cut to the scenarios this site shows (batching, paged, pools,
 * handoff, speculative) and the ring and pipeline. Against the simulator's
 * records (public/tradeoffs/mechanisms/*.json, written by the Python package
 * at the Trade-offs site's vendored commit through its
 * scripts/mechanisms_reference.py):
 *
 * - parity: the vendored engine, run on every recorded scenario's
 *   configuration and rows, gives the same request timestamps, bit for bit
 *   (a SHA-256 of the doubles; to 1e-9 for runs with power-capped steps,
 *   whose cube roots differ by an ulp across libms, as in the simulator's
 *   own parity test);
 * - frames: at least three key frames of every animation, the frame's
 *   state against the Python record (decode rows and prompt tokens against
 *   the step's own label, KV cells against the instance's KV in use, hit
 *   rate against the simulator's, tokens per verify pass against its
 *   statistics, peak power against the engine's), plus pinned captions.
 */
import { describe, expect, it } from "vitest";

import { bubble, gpipe, gpipeCaption } from "@/lib/tradeoffs/mech";
import { engine, type Row } from "@/lib/tradeoffs/engine";
import {
  SCENARIOS,
  batchCaption,
  batchHl,
  batchView,
  boxes,
  budgetOf,
  busyAt,
  devicesOf,
  engineReq,
  eventsUpTo,
  frameTime,
  gammaOf,
  hasPrompt,
  heldTokens,
  instSteps,
  kindOf,
  kvCaption,
  kvView,
  mechUrl,
  parseLabel,
  poolsCaption,
  poolsView,
  ringCaption,
  ringStates,
  ringStepTime,
  specCaption,
  specView,
  stepIndexAt,
  transfers,
  variantOf,
  type MFile,
  type MVariant,
} from "@/lib/tradeoffs/mech";
import { tpSteps } from "@/lib/tradeoffs/costs";

import { readJson, stampDigest } from "./helpers";

/** The recorded scenarios this site vendors (of the Trade-offs site's nine). */
const HERE = ["batching", "paged", "pools", "handoff", "speculative"] as const;
const FILES = Object.fromEntries(
  HERE.map((s) => [
    s,
    readJson<MFile>(`public/tradeoffs/mechanisms/${s}.json`),
  ]),
) as Record<(typeof HERE)[number], MFile>;
const VENDORED = readJson<{ simulator: { commit: string } }>(
  "src/lib/tradeoffs/VENDORED.json",
).simulator;
const v = (s: (typeof HERE)[number], k: string): MVariant =>
  variantOf(FILES[s], k);

describe("recorded scenarios = the vendored engine, request by request", () => {
  it("are a subset of the Trade-offs site's", () => {
    for (const s of HERE) expect(SCENARIOS).toContain(s);
  });
  for (const s of HERE) {
    const f = FILES[s];
    it(`${s}: recorded at the vendored commit`, () => {
      expect(f.commit).toBe(VENDORED.commit);
      expect(f.scenario).toBe(s);
      expect(f.variants.length).toBeGreaterThanOrEqual(2);
      expect(mechUrl(s)).toBe(`/tradeoffs/mechanisms/${s}.json`);
    });
    for (const x of f.variants)
      it(`${s} / ${x.key}`, () => {
        const res = engine.simulate(x.cfg, x.rows as Row[]);
        const capped = x.steps.some((st) => st.bound === "power");
        if (!capped) expect(stampDigest(res.reqs)).toBe(x.stamps_sha256);
        else
          res.reqs.forEach((r, i) => {
            const rec = x.reqs[i]!;
            const js = [
              r.arrival,
              r.prefillStart,
              r.firstToken,
              r.kvStart,
              r.kvReady,
              r.decodeStart,
              r.finish,
            ];
            js.forEach((a, j) => {
              const b = rec[j] ?? null;
              if (a === null || b === null) expect(a).toBe(b);
              else
                expect(Math.abs(a - b)).toBeLessThanOrEqual(
                  1e-9 * Math.max(1, Math.abs(b)),
                );
            });
          });
        // every request finished, and the records agree on how many ITLs were zero
        expect(res.reqs.every((r) => r.finish !== null)).toBe(true);
        res.reqs.forEach((r, i) => {
          expect(r.itls.length).toBe(x.reqs[i]![9]);
          expect(r.itls.filter((t) => t === 0).length).toBe(x.reqs[i]![10]);
        });
      });
  }
});

describe("basics", () => {
  it("time and step lookups", () => {
    const x = v("batching", "prefill-priority");
    const st = instSteps(x, 0);
    expect(stepIndexAt(st, -1)).toBe(-1);
    expect(stepIndexAt(st, st[3]!.t0)).toBe(3);
    expect(stepIndexAt(st, 1e9)).toBe(st.length - 1);
    expect(busyAt(st, -1)).toBeNull();
    expect(busyAt(st, st[2]!.t0)).toBe(st[2]);
    expect(busyAt(st, 1e9)).toBeNull();
    expect(frameTime(x, 0, 10)).toBe(0);
    expect(frameTime(x, 9, 10)).toBe(x.horizon);
    expect(frameTime(x, 3, 1)).toBe(x.horizon);
    expect(kindOf("prefill n=1 tok=3")).toBe("prefill");
    expect(kindOf("decode b=4")).toBe("decode");
    expect(kindOf("ced replay")).toBe("ced");
    expect(kindOf("step b=4 chunks=1 tok=512")).toBe("mixed");
    expect(parseLabel("step b=4 chunks=1 tok=512")).toEqual({
      b: 4,
      chunks: 1,
      tok: 512,
    });
    expect(() => variantOf(FILES.batching, "nope")).toThrow();
    expect(eventsUpTo(x, -1)).toEqual([]);
    expect(eventsUpTo(x, 1e9)).toHaveLength(x.events.length);
    const r = engineReq(x.reqs[0]!);
    expect(r.arrival).toBe(x.reqs[0]![0]);
    expect(r.prompt).toBe(x.reqs[0]![7]);
    expect(engineReq([0, null]).prefillStart).toBeNull();
    expect(heldTokens([0, 100, 100, 20, 0, 40, 5])).toBe(75);
    expect(heldTokens([0, 0, 100, 0, 0, 40, 0])).toBe(0);
    expect(budgetOf(x)).toBe(8192);
    expect(budgetOf(v("batching", "chunked-512"))).toBe(512);
    expect(devicesOf({ ...x, cfg: { devicesPerInstance: 4 } })).toBe(4);
    expect(devicesOf({ ...x, cfg: {} })).toBe(1);
  });
});

describe("chapter 1: batching frames = the simulator's steps", () => {
  for (const key of [
    "prefill-priority",
    "decode-priority",
    "chunked-256",
    "chunked-512",
    "chunked-2048",
  ]) {
    it(key, () => {
      const x = v("batching", key);
      const st = instSteps(x, 0);
      const frames = [
        0,
        1,
        Math.floor(st.length / 3),
        Math.floor(st.length / 2),
        st.length - 1,
      ];
      for (const k of frames) {
        const b = batchView(x, k);
        const lab = parseLabel(st[k]!.label);
        const kind = kindOf(st[k]!.label);
        if (kind === "mixed") {
          expect(b.decodeRows).toBe(lab.b);
          expect(b.decodeRows + b.prefillTokens).toBe(lab.tok);
          expect(b.slices.filter((s) => s.kind === "prefill")).toHaveLength(
            lab.chunks!,
          );
          expect(b.decodeRows + b.prefillTokens).toBeLessThanOrEqual(b.budget);
        } else if (kind === "prefill") {
          expect(b.decodeRows).toBe(0);
          expect(b.prefillTokens).toBe(lab.tok);
        } else expect(b.decodeRows).toBe(lab.b);
        expect(hasPrompt(st[k]!)).toBe(b.prefillTokens > 0);
      }
    });
  }
  it("prefill-priority stalls decodes, chunking bounds the stall", () => {
    const worst = (key: string) => {
      const x = v("batching", key);
      return Math.max(...instSteps(x, 0).map((_, k) => batchView(x, k).maxGap));
    };
    expect(worst("chunked-256")).toBeLessThan(worst("prefill-priority"));
    expect(worst("decode-priority")).toBeLessThan(worst("chunked-256"));
  });
  it("captions and the highlighted term", () => {
    const x = v("batching", "chunked-512");
    const b = batchView(x, 1);
    expect(batchCaption(b)).toBe(
      "Step 2 at 25.5 ms takes 43.2 ms: 1 decode row and 511 prompt tokens; 1 request waiting. The decode rows waited up to 43.2 ms for this token.",
    );
    expect(batchHl(b)).toBe("u");
    expect(batchHl(batchView(x, 0))).toBe("p");
    const d = batchView(v("batching", "decode-priority"), 20);
    expect(batchHl(d)).toBe("a");
    expect(
      batchCaption({ ...d, decodeRows: 0, prefillTokens: 0, waiting: 0 }),
    ).toMatch(/nothing\.$/);
    const pp = v("batching", "prefill-priority");
    const p0 = batchView(pp, 0);
    expect(p0.slices[0]!.rid).toBe(-1);
    expect(batchCaption(p0)).toMatch(
      /^Step 1 at 0\.0 ms takes .*: 300 prompt tokens/,
    );
  });
});

describe("chapter 2: KV cells = the instance's KV in use", () => {
  for (const key of ["oracle", "paged", "paged-swap"]) {
    it(key, () => {
      const x = v("paged", key);
      const blk = x.blk[0]!;
      for (const s of instSteps(x, 0))
        expect(s.rows.reduce((a, r) => a + r[4], 0)).toBe(s.kv);
      for (const q of [0.1, 0.3, 0.6, 0.9]) {
        const t = q * x.horizon;
        const k = kvView(x, t);
        expect(k.allocTokens).toBe(k.step!.kv * blk);
        const cells = k.cells.filter((c) => c !== null).length;
        expect(cells).toBe(
          k.step!.rows.reduce(
            (a, r) => a + Math.ceil((r[4] * blk) / k.cellTokens),
            0,
          ),
        );
        expect(k.heldTokens).toBeLessThanOrEqual(k.allocTokens);
        expect(k.running).toBe(k.step!.rows.length);
      }
      const end = kvView(x, x.horizon);
      expect(end.preemptions).toBe(x.summary.preemptions);
      expect(end.swaps).toBe(key === "paged-swap" ? x.summary.preemptions : 0);
    });
  }
  it("reserved memory holds fewer tokens than it books; paged holds nearly all", () => {
    const share = (key: string) => {
      const x = v("paged", key);
      const k = kvView(x, 0.3 * x.horizon);
      return k.heldTokens / k.allocTokens;
    };
    expect(share("oracle")).toBeLessThan(0.8);
    expect(share("paged")).toBeGreaterThan(0.9);
  });
  it("captions", () => {
    const x = v("paged", "paged-swap");
    expect(kvCaption(kvView(x, -1))).toBe("Nothing has arrived yet.");
    const k = kvView(x, x.horizon * 0.5);
    expect(kvCaption(k)).toMatch(/^t = .*: \d+ requests? running, \d+ waiting/);
    const c = kvCaption({ ...k, swapped: 2, preemptions: 1 });
    expect(c).toContain("2 swapped out to host memory");
    expect(c).toContain("1 preemption so far");
    expect(kvView({ ...x, steps: [] }, 1).cells).toHaveLength(0);
  });
});

describe("chapters 4–7: pool timelines = the simulator's steps and hand-offs", () => {
  for (const s of ["pools", "handoff"] as const) {
    for (const x of FILES[s].variants) {
      it(`${s} / ${x.key}`, () => {
        const bx = boxes(x);
        bx.forEach((lane, i) =>
          expect(lane).toHaveLength(instSteps(x, i).length),
        );
        const tr = transfers(x);
        if (x.cfg.mode === "disagg") expect(tr).toHaveLength(x.reqs.length);
        else expect(tr).toHaveLength(0);
        for (const q of [0.15, 0.5, 0.85]) {
          const t = q * x.horizon;
          const p = poolsView(x, t);
          x.insts.forEach((_, i) => {
            const st = instSteps(x, i);
            const on = st.find((u) => u.t0 <= t && t < u.t0 + u.dt) ?? null;
            expect(p.now[i]).toBe(on);
            expect(p.started[i]).toBe(st.filter((u) => u.t0 <= t).length);
          });
          expect(p.inFlight).toBe(
            tr.filter(([, a, b]) => a <= t && t < b).length,
          );
          expect(poolsCaption(x, p)).toMatch(/Requests: \d+ done/);
        }
      });
    }
  }
  it("captions", () => {
    const x = v("pools", "1p1d");
    const c = poolsCaption(x, poolsView(x, x.horizon));
    expect(c).toContain(`Link: 0 in flight, ${x.reqs.length} landed.`);
    expect(c).toContain(`Requests: ${x.reqs.length} done`);
    const mid = poolsCaption(x, poolsView(x, x.reqs[3]![4]! - 1e-9));
    expect(mid).toMatch(/in hand-off|1 in flight/);
    expect(
      poolsCaption(
        v("pools", "colocated"),
        poolsView(v("pools", "colocated"), 0),
      ),
    ).not.toContain("Link");
  });
});

describe("chapter 10: drafts kept = the simulator's draws", () => {
  for (const key of ["g3a07", "g5a08", "g3a04"]) {
    it(key, () => {
      const x = v("speculative", key);
      const st = instSteps(x, 0);
      let rows = 0;
      let kept = 0;
      st.forEach((_, k) => {
        const s = specView(x, k);
        for (const r of s.rows) {
          rows++;
          kept += r.kept;
          expect(r.accepted).toBe(r.kept - 1);
          expect(r.accepted + r.rejected).toBe(gammaOf(x));
        }
      });
      expect(kept / rows).toBeCloseTo(x.summary.tokens_per_verify, 12);
      const last = specView(x, st.length - 1);
      x.reqs.forEach((r, i) => expect(last.out[i]).toBe(r[8]));
    });
  }
  it("captions", () => {
    const x = v("speculative", "g3a07");
    expect(specCaption(x, specView(x, 0))).toMatch(/: a prompt's prefill\.$/);
    expect(specCaption(x, specView(x, 2))).toMatch(
      /request 0 keeps \d of 3 drafts \+ 1 = \d/,
    );
    const off = v("speculative", "off");
    expect(gammaOf(off)).toBe(0);
    expect(specCaption(off, specView(off, 5))).toMatch(
      /one token for each of 2 rows\.$/,
    );
    expect(specView(off, 5).rows.every((r) => r.rejected === 0)).toBe(true);
  });
});

describe("chapter 8: the ring all-reduce and the pipeline", () => {
  for (const n of [2, 4, 8]) {
    it(`ring of ${n}`, () => {
      const st = ringStates(n);
      expect(st).toHaveLength(2 * (n - 1) + 1);
      for (const s of st.slice(1)) expect(s.sends).toHaveLength(n);
      // after reduce-scatter each GPU owns one complete chunk; at the end all are complete
      const rs = st[n - 1]!;
      rs.have.forEach((row) =>
        expect(row.filter((c) => c === n)).toHaveLength(1),
      );
      expect(st[st.length - 1]!.have.flat().every((c) => c === n)).toBe(true);
      // the ring's time = the engine's all-reduce, per all-reduce (two per layer)
      const d = Number(engine.derive(engine.MODELS["llama3-70b"]!).d);
      const L = Number(engine.derive(engine.MODELS["llama3-70b"]!).L);
      const link = engine.LINKS.nvlink4!;
      const perAr = 2 * (n - 1) * ringStepTime(d * 2, n, link.bw, link.lat);
      expect(perAr * 2 * L).toBeCloseTo(tpSteps(n).decodeB1.commT!, 12);
      expect(ringCaption(st[0]!, n)).toMatch(/^Each of \d GPUs holds/);
      expect(ringCaption(st[1]!, n)).toMatch(/^Reduce-scatter step 1 of/);
      expect(ringCaption(st[st.length - 1]!, n)).toContain(
        `${n * n} of ${n * n} chunk copies are complete`,
      );
    });
  }
  it("GPipe slots and the bubble = results.md section 24", () => {
    // the cells of the simulator's results.md §24 (the Trade-offs site
    // checks them against its vendored results.md)
    for (const [m, cell] of [
      [1, "50.0%"],
      [2, "33.3%"],
      [4, "20.0%"],
    ] as const) {
      const g = gpipe(2, m);
      g.forEach((stage) =>
        expect(stage.filter((x) => x !== null)).toHaveLength(m),
      );
      expect(`${(100 * bubble(2, m)).toFixed(1)}%`).toBe(cell);
    }
    expect(gpipe(4, 4)[3]![0]).toBeNull();
    expect(gpipeCaption(2, 2, 0)).toBe(
      "Slot 1 of 3: 1 of 2 stages busy. Over the whole step the bubble is 33% of stage time.",
    );
  });
});
