import { describe, expect, it } from "vitest";

import {
  LLAMA3_70B,
  LLAMA3_8B,
  kvBytesPerToken,
} from "@/lib/inference/costModel";
import {
  KV_PRESETS,
  kvBytes,
  kvHeadsFor,
  kvValuesPerToken,
  tokensThatFit,
} from "@/lib/inference/kvMemory";

describe("KV-cache size", () => {
  it("GQA presets equal the simulator's kv_bytes_per_token (BF16)", () => {
    expect(kvBytes(KV_PRESETS["llama3-8b"]!, "gqa", 2, 1)).toBe(
      kvBytesPerToken(LLAMA3_8B),
    );
    expect(kvBytes(KV_PRESETS["llama3-70b"]!, "gqa", 2, 1)).toBe(
      kvBytesPerToken(LLAMA3_70B),
    );
    // 128 KiB and 320 KiB per token.
    expect(kvBytesPerToken(LLAMA3_8B)).toBe(128 * 1024);
    expect(kvBytesPerToken(LLAMA3_70B)).toBe(320 * 1024);
  });

  it("follows DeepSeek-V2 Table 1 for MHA, GQA, MQA and MLA", () => {
    const s = KV_PRESETS["llama3-8b"]!; // 32 layers, 32 heads of 128, 8 KV heads
    expect(kvValuesPerToken(s, "mha")).toBe(2 * 32 * 32 * 128);
    expect(kvValuesPerToken(s, "gqa")).toBe(2 * 8 * 128 * 32);
    expect(kvValuesPerToken(s, "mqa")).toBe(2 * 128 * 32);
    expect(kvValuesPerToken(s, "mla")).toBe((512 + 64) * 32);
    expect(kvHeadsFor(s, "mqa")).toBe(1);
    // MLA ≈ GQA with 2.25 groups (DeepSeek-V2 §2.1.4): 576 / (2·128) = 2.25.
    const d = KV_PRESETS["deepseek-v2"]!;
    expect(kvValuesPerToken(d, "mla") / (2 * d.headDim * d.layers)).toBe(2.25);
  });

  it("scales with tokens and bytes per value", () => {
    const s = KV_PRESETS["llama3-70b"]!;
    expect(kvBytes(s, "gqa", 1, 4096)).toBe(kvBytes(s, "gqa", 2, 4096) / 2);
    expect(kvBytes(s, "gqa", 2, 8192)).toBe(2 * kvBytes(s, "gqa", 2, 4096));
  });

  it("counts how many tokens fit", () => {
    const s = KV_PRESETS["llama3-8b"]!;
    expect(tokensThatFit(s, "gqa", 2, 128 * 1024 * 10 + 5)).toBe(10);
    expect(tokensThatFit(s, "gqa", 2, -1)).toBe(0);
  });
});
