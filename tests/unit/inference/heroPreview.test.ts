import { describe, expect, it } from "vitest";

import { heroPreview } from "@/lib/inference/heroPreview";
import { toyLabel } from "@/lib/inference/toyModel";

describe("landing preview", () => {
  it("is exact, and the cache saves most of the work", () => {
    const h = heroPreview();
    expect(h.maxDiff).toBe(0);
    expect(h.cached).toHaveLength(24);
    expect(h.cached.at(-1)! / h.uncached.at(-1)!).toBeLessThan(0.2);
  });
  it("labels whitespace visibly", () => {
    expect(toyLabel(36)).toBe("␣");
    expect(toyLabel(999)).toBe("?");
  });
});
