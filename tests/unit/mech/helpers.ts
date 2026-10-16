/** Loaders and the timestamp digest for the vendored mechanism tests (from Inference Trade-offs Explained's tests/unit/helpers/load.ts). */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import type { EngineRequest } from "@/lib/tradeoffs/engine";

export const readJson = <T>(path: string): T =>
  JSON.parse(readFileSync(join(process.cwd(), path), "utf-8")) as T;

const STAMP_KEYS = [
  "arrival",
  "prefillStart",
  "firstToken",
  "kvStart",
  "kvReady",
  "decodeStart",
  "finish",
] as const;

/** SHA-256 of every request's seven stamps as little-endian doubles (null as the canonical NaN), as the Python script writes it. */
export function stampDigest(reqs: readonly EngineRequest[]): string {
  const buf = Buffer.alloc(reqs.length * 7 * 8);
  let o = 0;
  for (const r of reqs)
    for (const k of STAMP_KEYS) {
      const v = r[k];
      if (v === null) buf.writeBigUInt64LE(0x7ff8000000000000n, o);
      else buf.writeDoubleLE(v, o);
      o += 8;
    }
  return createHash("sha256").update(buf).digest("hex");
}
