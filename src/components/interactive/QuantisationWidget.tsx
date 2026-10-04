"use client";

/**
 * Bytes per decode step, split into weights and KV cache, for each weight
 * and KV format; and what that does to step time and tokens per second.
 */
import { useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { H100_SXM, LLAMA3_70B, LLAMA3_8B } from "@/lib/inference/costModel";
import {
  KV_FORMATS,
  WEIGHT_FORMATS,
  decodeBreakdown,
  withFormats,
} from "@/lib/inference/quantisation";

const gb = (b: number): string => `${(b / 1e9).toFixed(2)} GB`;

export function QuantisationWidget(): JSX.Element {
  const [model, setModel] = useState<"8b" | "70b">("8b");
  const [w, setW] = useState("bf16");
  const [kv, setKv] = useState("bf16");
  const [batch, setBatch] = useState(16);
  const [ctx, setCtx] = useState(4096);
  const base = model === "8b" ? LLAMA3_8B : LLAMA3_70B;
  const n = model === "8b" ? 1 : 4;
  const wb = WEIGHT_FORMATS.find((f) => f.key === w)!.bytes;
  const kb = KV_FORMATS.find((f) => f.key === kv)!.bytes;

  const ref = decodeBreakdown(base, H100_SXM, n, ctx, batch);
  const cur = decodeBreakdown(
    withFormats(base, wb, kb),
    H100_SXM,
    n,
    ctx,
    batch,
  );
  const rows = WEIGHT_FORMATS.flatMap((wf) =>
    KV_FORMATS.slice(0, 2).map((kf) => ({
      label: `W ${wf.key} · KV ${kf.key}`,
      d: decodeBreakdown(
        withFormats(base, wf.bytes, kf.bytes),
        H100_SXM,
        n,
        ctx,
        batch,
      ),
    })),
  );
  const maxBytes = ref.weightBytes + ref.kvBytes;

  return (
    <WidgetFrame
      testId="quantisation"
      title="Fewer bytes per step, faster decode"
      caption={`${base.name} on ${n}× H100 (the simulator's cost model). The FLOP rate stays at BF16: weight-only formats dequantise before multiplying, and faster 8-bit tensor-core maths is not modelled. Accuracy is not modelled either.`}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Segmented
          label="Model"
          value={model}
          options={[
            { value: "8b", label: "Llama-3-8B ×1" },
            { value: "70b", label: "Llama-3-70B ×4" },
          ]}
          onChange={setModel}
        />
        <Segmented
          label="Weights"
          value={w}
          options={WEIGHT_FORMATS.map((f) => ({ value: f.key, label: f.key }))}
          onChange={setW}
        />
        <Segmented
          label="KV cache"
          value={kv}
          options={KV_FORMATS.map((f) => ({ value: f.key, label: f.key }))}
          onChange={setKv}
        />
        <div className="grid gap-3">
          <Slider
            label="Batch"
            value={batch}
            min={1}
            max={256}
            onChange={setBatch}
          />
          <Slider
            label="Context per sequence"
            value={ctx}
            min={256}
            max={32768}
            step={256}
            onChange={setCtx}
          />
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Weight bytes / step"
          value={gb(cur.weightBytes)}
          hint={`was ${gb(ref.weightBytes)}`}
        />
        <Stat
          label="KV bytes / step"
          value={gb(cur.kvBytes)}
          hint={`was ${gb(ref.kvBytes)}`}
        />
        <Stat
          label="Step time"
          value={`${(cur.time * 1e3).toFixed(2)} ms`}
          hint={`${cur.bound}-bound; was ${(ref.time * 1e3).toFixed(2)} ms`}
        />
        <Stat
          label="Tokens / s"
          value={cur.tokensPerSecond.toFixed(0)}
          hint={`${(cur.tokensPerSecond / ref.tokensPerSecond).toFixed(2)}× BF16`}
        />
      </div>
      <p className="mt-2 text-sm" role="status">
        Memory: {gb(cur.residentWeightBytes)} of weights + {gb(cur.kvBytes)} of
        cache = {gb(cur.residentWeightBytes + cur.kvBytes)} of {n * 80} GB.{" "}
        {cur.residentWeightBytes + cur.kvBytes > n * 80e9 * 0.9 ? (
          <strong className="text-rose-700 dark:text-rose-300">
            Doesn&rsquo;t fit (over 90% of HBM): lower the batch or context.
          </strong>
        ) : (
          "Fits."
        )}
      </p>

      <div
        className="mt-4 space-y-1.5"
        aria-label="Bytes per decode step by format"
      >
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2 text-[0.7rem]">
            <span className="w-28 shrink-0 font-mono text-neutral-600 dark:text-neutral-400">
              {r.label}
            </span>
            <span className="flex min-w-0 flex-1">
              <span
                className="block h-3 bg-indigo-500"
                style={{ width: `${(r.d.weightBytes / maxBytes) * 100}%` }}
              />
              <span
                className="block h-3 bg-amber-500"
                style={{ width: `${(r.d.kvBytes / maxBytes) * 100}%` }}
              />
            </span>
            <span className="w-16 shrink-0 text-right font-mono">
              {(r.d.time * 1e3).toFixed(1)} ms
            </span>
          </div>
        ))}
        <p className="text-xs text-neutral-600 dark:text-neutral-400">
          Blue: weight bytes. Amber: KV-cache bytes. Right: step time.
        </p>
      </div>
    </WidgetFrame>
  );
}
