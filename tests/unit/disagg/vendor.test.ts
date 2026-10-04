/**
 * The vendored engine is the file at the pinned commit, byte for byte, and
 * every fixture and workload was generated at that same commit.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { engine, fixed, FORMATS } from "@/lib/disagg/engine";
import { WORKLOAD_RATES } from "@/lib/disagg/workloads";

import { readJson, results, workload } from "./helpers";

const vendored = readJson<{ commit: string; sha256: string; path: string }>(
  "src/lib/disagg/vendor/VENDORED.json",
);

describe("vendored sim_engine.js", () => {
  it("matches the SHA-256 recorded with its commit", () => {
    const src = readFileSync(
      join(process.cwd(), "src/lib/disagg/vendor/sim_engine.js"),
    );
    expect(createHash("sha256").update(src).digest("hex")).toBe(
      vendored.sha256,
    );
    expect(vendored.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(vendored.path).toBe("web/sim_engine.js");
  });

  it("fixtures and workloads come from the same commit", () => {
    expect(
      readJson<{ commit: string }>("tests/unit/fixtures/disagg_parity.json")
        .commit,
    ).toBe(vendored.commit);
    expect(results.commit).toBe(vendored.commit);
    for (const r of WORKLOAD_RATES) {
      const w = workload(r);
      expect(w.commit).toBe(vendored.commit);
      expect(w.rate).toBe(r);
      expect(w.rows).toHaveLength(800);
    }
  });

  it("exposes the tables the chapters use", () => {
    expect(Object.keys(engine.LINKS)).toEqual(
      expect.arrayContaining([
        "nvlink4",
        "ib-ndr",
        "eth-100g",
        "eth-25g",
        "cpo-optical",
      ]),
    );
    expect(engine.MODELS["llama3-70b"]!.L).toBe(80);
  });
});

describe("number formats (examples/results.py)", () => {
  it("groups thousands and switches units like Python", () => {
    expect(fixed(1234567.891, 1)).toBe("1,234,567.9");
    expect(fixed(-1234.5, 0)).toBe("-1,234");
    expect(fixed(999, 0)).toBe("999");
    expect(fixed(1235.5, 0)).toBe("1,236");
    expect(fixed(0.125, 2)).toBe("0.12");
    expect(fixed(0.375, 2)).toBe("0.38");
    expect(FORMATS.ms(0.8542)).toBe("854.2 ms");
    expect(FORMATS.ms(6.0851)).toBe("6,085 ms");
    expect(FORMATS.ms2(9.6578)).toBe("9,657.8 ms");
    expect(FORMATS.ms2(47.1)).toBe("47.1 s");
    expect(FORMATS.pct(0.997)).toBe("99.7%");
    expect(FORMATS.W(2687.4)).toBe("2,687 W");
    expect(FORMATS.f2(2.774)).toBe("2.77");
    expect(FORMATS.f3(0.3264)).toBe("0.326");
  });
});
