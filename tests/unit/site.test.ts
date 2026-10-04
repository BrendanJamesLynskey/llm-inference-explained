import { describe, expect, it } from "vitest";

import { decoderChapter, infsimGlossary } from "@/lib/site";

describe("site links", () => {
  it("build explainer chapter and glossary URLs", () => {
    expect(decoderChapter("07-sampling")).toBe(
      "https://transformer-decoder-explained.vercel.app/learn/07-sampling",
    );
    expect(infsimGlossary("kvcache")).toBe(
      "https://brendanjameslynskey.github.io/LLM_Hub_Inference_Simulators/#g-kvcache",
    );
  });
});
