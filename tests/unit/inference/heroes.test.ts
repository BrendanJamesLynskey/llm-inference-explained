/**
 * Brief 41's chapter heroes against scripts/hero_reference.py
 * (tests/unit/fixtures/heroes.json): the timeline and the roofline climb
 * against the simulator's Python cost model, exactly; the KV-cache hero
 * against a float64 numpy forward pass recomputed at every step (to 1e-12;
 * tokens exactly); the FlashAttention tiles against a plain-Python online
 * softmax (to 1e-12). Captions are pinned where they quote the model.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/heroes.json";

import {
  FLASH_HERO,
  attentionNaive,
  flashCaption,
  flashHeroInputs,
  flashTileStates,
} from "@/lib/inference/attentionKernels";
import {
  ROOF_DEFAULT,
  TIMELINE_DEFAULT,
  generationTimeline,
  roofCaption,
  rooflineClimb,
  timelineCaption,
} from "@/lib/inference/heroes";
import { kvHeroCaption, kvHeroStates } from "@/lib/inference/kvHero";
import "@/lib/inference/trainedData";
import { toyWeights } from "@/lib/inference/toyModel";

function close(a: unknown, b: unknown, tol: number, path = "$"): string | null {
  if (typeof b === "number" && typeof a === "number")
    return Math.abs(a - b) <= tol * Math.max(1, Math.abs(b))
      ? null
      : `${path}: ${a} vs ${b}`;
  if (b === null)
    return a === null || a === -Infinity
      ? null
      : `${path}: ${String(a)} vs null`;
  if (Array.isArray(b)) {
    if (!Array.isArray(a) || a.length !== b.length) return `${path}: length`;
    for (let i = 0; i < b.length; i++) {
      const m = close(a[i], b[i], tol, `${path}[${i}]`);
      if (m) return m;
    }
    return null;
  }
  if (typeof b === "object") {
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    for (const k of new Set([...Object.keys(ao), ...Object.keys(bo)])) {
      if (!(k in bo)) return `${path}.${k}: unexpected`;
      const m = close(ao[k], bo[k], tol, `${path}.${k}`);
      if (m) return m;
    }
    return null;
  }
  return a === b ? null : `${path}: ${String(a)} vs ${String(b)}`;
}

describe("chapter 1: the generation timeline = the Python cost model", () => {
  const steps = generationTimeline();
  it("uses the reference's request", () =>
    expect(TIMELINE_DEFAULT).toEqual(fx.timeline.opts));
  it("every pass, exactly", () => {
    expect(close(steps, fx.timeline.steps, 0)).toBe(null);
    expect(steps).toHaveLength(TIMELINE_DEFAULT.output);
  });
  it("prefill is compute-bound, decode memory-bound", () => {
    expect(steps[0]!.bound).toBe("compute");
    expect(steps.slice(1).every((s) => s.bound === "memory")).toBe(true);
    expect(steps[0]!.dt / steps[1]!.dt).toBeGreaterThan(5);
  });
  it("captions", () => {
    expect(timelineCaption(steps, 0)).toMatch(
      /^Prefill: all 2,048 prompt tokens in one pass, \d+\.\d ms, compute-bound \(\d,\d{3} FLOP per byte\)\..* time to first token \d+\.\d ms\.$/,
    );
    expect(timelineCaption(steps, 1)).toMatch(
      /^Decode step 1: one new token, reading the weights and 2,048 cached tokens, \d\.\d ms, memory-bound \(\d\.\d FLOP per byte\)\. 2 tokens out at \d+\.\d ms\.$/,
    );
  });
});

describe("chapter 3: the roofline climb = the Python cost model", () => {
  const c = rooflineClimb();
  it("uses the reference's batches", () =>
    expect(ROOF_DEFAULT).toEqual(fx.roofline.opts));
  it("every point, exactly", () => {
    expect(close(c.points, fx.roofline.points, 0)).toBe(null);
    expect(close(c.prefill, fx.roofline.prefill, 0)).toBe(null);
    expect([c.peak, c.bandwidth, c.ridge]).toEqual([
      fx.roofline.peak,
      fx.roofline.bandwidth,
      fx.roofline.ridge,
    ]);
  });
  it("the point climbs but the KV reads keep it under the ridge", () => {
    const i = c.points.map((p) => p.intensity);
    i.slice(1).forEach((v, k) => expect(v).toBeGreaterThan(i[k]!));
    expect(Math.max(...i)).toBeLessThan(c.ridge);
    expect(c.prefill.intensity).toBeGreaterThan(c.ridge);
  });
  it("captions", () => {
    expect(roofCaption(c, 0)).toMatch(
      /^Batch 1: 1\.1 FLOP per byte, below the ridge at 203: memory-bound\. The step takes \d\.\d ms and yields 161 tokens\/s\.$/,
    );
    expect(roofCaption({ ...c, ridge: 1 }, 8)).toMatch(/past the ridge/);
  });
});

describe("chapter 2: the KV cache filling = a forward pass recomputed each step", () => {
  const st = kvHeroStates(fx.kv.prompt, fx.kv.rounds, toyWeights("trained"));
  it("every state", () => {
    expect(st).toHaveLength(fx.kv.states.length);
    st.forEach((s, k) =>
      expect(close(s, fx.kv.states[k], 1e-12), `state ${k}`).toBe(null),
    );
  });
  it("the trained model continues “said” as “said the”", () => {
    const ids = st[st.length - 1]!.ids;
    expect(ids).toHaveLength(8);
    expect(kvHeroCaption(st[0]!)).toMatch(
      /^Prefill: the 4 prompt tokens go through the model in one pass, and each block caches their 4 keys and values\. Most likely next token: “␣” \(\d+%\)\.$/,
    );
    expect(kvHeroCaption(st[1]!)).toMatch(
      /^Step 1: the new token “␣” at position 4 asks every cached key; block 1, head 0 reads most from position \d “.” \(\d+%\)\.$/,
    );
    expect(kvHeroCaption(st[2]!)).toBe(
      "Step 1: its own key and value join the cache, one column per block (5 now); the 4 earlier columns are unchanged.",
    );
    expect(kvHeroCaption(st[3]!)).toMatch(/^Step 1: the output head picks “t”/);
  });
  it("the cache only grows: earlier columns never change", () => {
    for (let k = 1; k < st.length; k++)
      st[k]!.K.forEach((K, b) =>
        st[k - 1]!.K[b]!.forEach((row, i) => expect(K[i]).toEqual(row)),
      );
  });
});

describe("chapter 6: FlashAttention's tiles = a plain-Python online softmax", () => {
  const { Q, K, V } = flashHeroInputs();
  const st = flashTileStates(Q, K, V, FLASH_HERO.br, FLASH_HERO.bc);
  it("uses the reference's problem", () => {
    expect(FLASH_HERO).toEqual(fx.flash.opts);
    // the generator's log and cos may round differently in the last place
    expect(close(Q[0], fx.flash.Q0, 1e-15)).toBe(null);
  });
  it("every tile's running max and sum", () => {
    expect(st).toHaveLength(16);
    st.forEach((s, k) =>
      expect(close(s, fx.flash.states[k], 1e-12), `tile ${k}`).toBe(null),
    );
  });
  it("traffic: Q once, K and V once per query block, O once", () => {
    const { n, d, br } = FLASH_HERO;
    expect(st[st.length - 1]!.hbm).toBe(n * d + (n / br) * 2 * n * d + n * d);
    expect(st[st.length - 1]!.written).toHaveLength(n);
    expect(fx.flash.maxDiffVsNaive).toBeLessThan(1e-14);
    expect(attentionNaive(Q, K, V)).toHaveLength(n);
  });
  it("captions", () => {
    expect(flashCaption(st[0]!, 16)).toMatch(
      /^Tile 1: queries 0–3 against keys 0–3\. Their scores live only in on-chip memory; row 0's running max is now -?\d\.\d\d and its sum \d\.\d\d\. HBM traffic so far: 96 numbers\.$/,
    );
    expect(flashCaption(st[3]!, 16)).toContain(
      "Rows 0–3 have seen every key: O is normalised and written.",
    );
  });
});
