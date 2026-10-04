/**
 * `types.ts` is purely type aliases (Vector, Matrix, Tensor3D) with no
 * runtime exports — vitest's coverage tracker can't compute "lines covered"
 * for a file that emits no JS. Importing it here is the smallest signal
 * that the module participates in the build, which keeps the file out of
 * "uncovered" reports.
 */
import { describe, expect, it } from "vitest";

import type { Matrix, Tensor3D, Vector } from "@/lib/transformer/types";

describe("types", () => {
  it("type aliases are usable", () => {
    const v: Vector = [1, 2];
    const m: Matrix = [[1, 2]];
    const t: Tensor3D = [[[1]]];
    expect(v.length + m.length + t.length).toBe(4);
  });
});
