/**
 * The files copied from Inference Trade-offs Explained (brief 41) are byte
 * for byte the ones its commit recorded in src/lib/tradeoffs/VENDORED.json,
 * and the scenarios were recorded at the simulator commit it names.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { readJson } from "./helpers";

const V = readJson<{
  commit: string;
  simulator: { commit: string };
  files: { path: string; sha256: string }[];
}>("src/lib/tradeoffs/VENDORED.json");

describe("vendored from Inference Trade-offs Explained", () => {
  it("names its commits", () => {
    expect(V.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(V.simulator.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(V.files.length).toBeGreaterThan(20);
  });
  for (const f of V.files)
    it(f.path, () => {
      const got = createHash("sha256")
        .update(readFileSync(join(process.cwd(), f.path)))
        .digest("hex");
      expect(got).toBe(f.sha256);
      if (f.path.endsWith(".json") && f.path.includes("mechanisms/"))
        if (!f.path.endsWith("cost_model.json"))
          expect(readJson<{ commit: string }>(f.path).commit).toBe(
            V.simulator.commit,
          );
    });
});
