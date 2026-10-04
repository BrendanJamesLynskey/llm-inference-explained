/** The hand-off closed forms against their definitions and results.md §14–15. */
import { describe, expect, it } from "vitest";

import { COMPRESSION, handoff } from "@/lib/disagg/handoff";
import { LLAMA3_70B, LLAMA3_8B } from "@/lib/inference/costModel";

const base = {
  model: LLAMA3_8B,
  devices: 1,
  prompt: 2048,
  link: "eth-25g",
  compression: "none",
  layerwise: false,
} as const;

describe("KV hand-off", () => {
  it("a 2,048-token Llama-3-8B prompt is 268.4 MB and saturates 25 GbE near 11.6 req/s", () => {
    const h = handoff(base);
    expect(h.bytes).toBe(2048 * 131072);
    expect((h.bytes / 1e6).toFixed(1)).toBe("268.4");
    expect(h.transferTime).toBe(20e-6 + h.bytes / 3.125e9);
    expect((h.transferTime * 1e3).toFixed(1)).toBe("85.9");
    expect(h.linkSaturationRate.toFixed(1)).toBe("11.6");
    expect(h.exposedTime).toBe(h.transferTime);
  });

  it("70B: 671.1 MB, 13.4 ms over InfiniBand NDR", () => {
    const h = handoff({
      ...base,
      model: LLAMA3_70B,
      devices: 4,
      link: "ib-ndr",
    });
    expect((h.bytes / 1e6).toFixed(1)).toBe("671.1");
    expect((h.transferTime * 1e3).toFixed(1)).toBe("13.4");
  });

  it("compression divides the wire bytes by the preset's ratio", () => {
    expect(COMPRESSION["fp4-block"]).toBe(64 / 17);
    const h = handoff({ ...base, compression: "fp8" });
    expect(h.wireBytes).toBe(h.bytes / 2);
    expect(h.linkSaturationRate).toBeCloseTo(
      2 * handoff(base).linkSaturationRate,
      12,
    );
  });

  it("layer-wise streaming leaves one layer's slice when the link keeps up", () => {
    const h = handoff({ ...base, link: "ib-ndr", layerwise: true });
    const wire = h.bytes / 50e9;
    expect(wire).toBeLessThan(h.prefillTime);
    expect(h.exposedTime).toBeCloseTo(10e-6 + wire / 32, 15);
  });

  it("…and the excess over the prefill when it doesn't", () => {
    const h = handoff({ ...base, layerwise: true });
    const wire = h.bytes / 3.125e9;
    expect(wire).toBeGreaterThan(h.prefillTime);
    expect(h.exposedTime).toBeCloseTo(
      20e-6 + wire - (h.prefillTime * 31) / 32,
      15,
    );
    expect(h.exposedTime).toBeLessThan(handoff(base).exposedTime);
  });
});
