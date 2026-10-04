"use client";

/**
 * Interactive roofline: attainable FLOP/s against arithmetic intensity for
 * the simulator's H100 and A100 (derated peaks), with a prefill step and a
 * decode step placed on it by the ported cost model.
 */
import { scaleLog } from "d3";
import { useMemo, useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  A100_SXM,
  H100_SXM,
  LLAMA3_70B,
  LLAMA3_8B,
  costModel,
  type StepCost,
} from "@/lib/inference/costModel";

const W = 560;
const H = 300;
const M = { l: 52, r: 12, t: 12, b: 40 };

const fmtTime = (s: number): string =>
  s >= 1 ? `${s.toFixed(2)} s` : `${(s * 1e3).toFixed(2)} ms`;

export function RooflineWidget(): JSX.Element {
  const [dev, setDev] = useState<"h100" | "a100">("h100");
  const [model, setModel] = useState<"8b" | "70b">("8b");
  const [batch, setBatch] = useState(1);
  const [ctx, setCtx] = useState(2048);
  const [prompt, setPrompt] = useState(2048);

  const device = dev === "h100" ? H100_SXM : A100_SXM;
  const m = model === "8b" ? LLAMA3_8B : LLAMA3_70B;
  const n = model === "8b" ? 1 : 4;
  const cm = useMemo(
    () => costModel(m, device, { nDevices: n }),
    [m, device, n],
  );
  const pre = cm.prefill([prompt]);
  const dec = cm.decode(ctx * batch, batch);

  const x = scaleLog()
    .domain([0.5, 20000])
    .range([M.l, W - M.r]);
  const peak = cm.flopsRate;
  const y = scaleLog()
    .domain([1e12, peak * 2])
    .range([H - M.b, M.t]);
  const roof = (i: number): number => Math.min(peak, i * cm.byteRate);
  const pts: number[] = [];
  for (let e = Math.log10(0.5); e <= Math.log10(20000); e += 0.02)
    pts.push(10 ** e);
  const path = pts
    .map(
      (i, k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(roof(i)).toFixed(1)}`,
    )
    .join("");
  const ridge = cm.ridgePoint;
  const attained = (s: StepCost): number => s.flops / (s.time - 0.5e-3);

  const point = (s: StepCost, label: string, cls: string) => (
    <g>
      <circle
        cx={x(s.intensity)}
        cy={y(Math.max(1e12, attained(s)))}
        r={6}
        className={cls}
      />
      <text
        x={x(s.intensity)}
        y={y(Math.max(1e12, attained(s))) - 10}
        textAnchor="middle"
        className="fill-neutral-800 text-[11px] dark:fill-neutral-200"
      >
        {label}
      </text>
    </g>
  );

  return (
    <WidgetFrame
      testId="roofline"
      title="Prefill and decode on the roofline"
      caption={`Roofs are the simulator's derated peaks (55% of peak FLOP/s, 80% of peak bandwidth; hardware.py). ${m.name} on ${n}× ${device.name}. Points are the cost model's steps; their height excludes the fixed 0.5 ms step overhead.`}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Segmented
          label="Device"
          value={dev}
          options={[
            { value: "h100", label: "H100-SXM" },
            { value: "a100", label: "A100-SXM" },
          ]}
          onChange={setDev}
        />
        <Segmented
          label="Model"
          value={model}
          options={[
            { value: "8b", label: "Llama-3-8B ×1" },
            { value: "70b", label: "Llama-3-70B ×4" },
          ]}
          onChange={setModel}
        />
        <Slider
          label="Prefill: prompt tokens"
          value={prompt}
          min={16}
          max={8192}
          step={16}
          onChange={setPrompt}
        />
        <Slider
          label="Decode: batch"
          value={batch}
          min={1}
          max={512}
          onChange={setBatch}
        />
        <Slider
          label="Decode: context per sequence"
          value={ctx}
          min={128}
          max={32768}
          step={128}
          onChange={setCtx}
        />
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mt-4 h-auto w-full"
        role="img"
        aria-label={`Roofline. Ridge point ${ridge.toFixed(0)} FLOP per byte. Prefill intensity ${pre.intensity.toFixed(0)}, ${pre.bound}-bound. Decode intensity ${dec.intensity.toFixed(1)}, ${dec.bound}-bound.`}
      >
        {[1, 10, 100, 1000, 10000].map((t) => (
          <g key={t}>
            <line
              x1={x(t)}
              x2={x(t)}
              y1={M.t}
              y2={H - M.b}
              className="stroke-neutral-200 dark:stroke-neutral-800"
            />
            <text
              x={x(t)}
              y={H - M.b + 14}
              textAnchor="middle"
              className="fill-neutral-500 text-[10px] dark:fill-neutral-400"
            >
              {t}
            </text>
          </g>
        ))}
        {[1e12, 1e13, 1e14, 1e15]
          .filter((v) => v < peak * 2)
          .map((v) => (
            <g key={v}>
              <line
                x1={M.l}
                x2={W - M.r}
                y1={y(v)}
                y2={y(v)}
                className="stroke-neutral-200 dark:stroke-neutral-800"
              />
              <text
                x={M.l - 6}
                y={y(v) + 3}
                textAnchor="end"
                className="fill-neutral-500 text-[10px] dark:fill-neutral-400"
              >
                {v / 1e12} T
              </text>
            </g>
          ))}
        <text
          x={(M.l + W - M.r) / 2}
          y={H - 6}
          textAnchor="middle"
          className="fill-neutral-600 text-[11px] dark:fill-neutral-400"
        >
          Arithmetic intensity (FLOP / byte, log)
        </text>
        <text
          x={12}
          y={(M.t + H - M.b) / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${(M.t + H - M.b) / 2})`}
          className="fill-neutral-600 text-[11px] dark:fill-neutral-400"
        >
          FLOP/s (log)
        </text>
        <path
          d={path}
          fill="none"
          strokeWidth={2.5}
          className="stroke-neutral-800 dark:stroke-neutral-200"
        />
        <line
          x1={x(ridge)}
          x2={x(ridge)}
          y1={y(peak)}
          y2={H - M.b}
          strokeDasharray="4 3"
          className="stroke-neutral-500"
        />
        <text
          x={x(ridge) + 4}
          y={H - M.b - 6}
          className="fill-neutral-600 text-[10px] dark:fill-neutral-400"
        >
          ridge {ridge.toFixed(0)}
        </text>
        {point(dec, "decode", "fill-indigo-500")}
        {point(pre, "prefill", "fill-emerald-500")}
      </svg>

      <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Prefill step"
          value={fmtTime(pre.time)}
          hint={`${pre.bound}-bound, ${pre.intensity.toFixed(0)} FLOP/B`}
        />
        <Stat
          label="Decode step"
          value={fmtTime(dec.time)}
          hint={`${dec.bound}-bound, ${dec.intensity.toFixed(1)} FLOP/B`}
        />
        <Stat
          label="Decode tokens/s"
          value={(batch / dec.time).toFixed(0)}
          hint={`batch ${batch}`}
        />
        <Stat
          label="Decode bytes / step"
          value={`${(dec.bytes / 1e9).toFixed(2)} GB`}
          hint="weights + KV"
        />
      </div>
    </WidgetFrame>
  );
}
