"use client";

/**
 * Chapter 8's second mechanism: pipeline parallelism. The model's layers
 * are cut into p stages, one per group of GPUs; a step's batch is cut into
 * m micro-batches that flow through the stages, so a stage idles until the
 * first micro-batch reaches it and after the last has left: the bubble.
 * Underneath, the vendored engine prices the same instance's prefill and
 * decode steps live (each micro-batch re-reads its stage's weights, and
 * successive steps are not overlapped: the simulator's known gap).
 */
import { useMemo, useState } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import { ppSteps } from "@/lib/tradeoffs/costs";
import { bubble, fmtSeconds, gpipe, gpipeCaption } from "@/lib/tradeoffs/mech";
import { OKABE_ITO } from "@/lib/viz/palette";

import { Hatch } from "./common";

const MB = [
  OKABE_ITO.blue,
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.purple,
];

const LAYOUTS = {
  "2x2": { tp: 2, pp: 2, label: "TP2 × PP2" },
  "1x4": { tp: 1, pp: 4, label: "PP4" },
} as const;

export default function GpipeWidget({
  equation,
}: {
  equation?: React.ReactNode;
}): JSX.Element {
  const [layout, setLayout] = useState<keyof typeof LAYOUTS>("2x2");
  const [m, setM] = useState("2");
  const L = LAYOUTS[layout];
  const M = Number(m);
  const grid = useMemo(() => gpipe(L.pp, M), [L.pp, M]);
  const slots = M + L.pp - 1;
  const stepper = useStepper(slots + 1, {
    stepMs: 900,
    resetKey: `${layout}-${m}`,
  });
  const [ref, width] = useWidth(640);
  const pipe = useMemo(() => ppSteps(L.tp, L.pp, M), [L.tp, L.pp, M]);
  const tp4 = useMemo(() => ppSteps(4, 1, null), []);
  const shown = stepper.step; // slots completed
  const W = width;
  const label = 64;
  const cw = Math.min(90, (W - label - 12) / slots);
  const ch = 26;
  const H = 24 + L.pp * (ch + 6) + 40;
  return (
    <AnimationPanel
      title="Micro-batches through pipeline stages"
      summary={
        <>
          Four H100s split into pipeline stages. Each column is a slot; each
          stage works on one micro-batch per slot (colour = micro-batch) or
          idles (hatched: the bubble). The step times are the engine&apos;s, for
          a prefill of four 4,096-token prompts, live for the layout chosen.
        </>
      }
      stepper={stepper}
      stepLabel="slot"
      countFrom={0}
      caption={
        shown === 0
          ? `Before the step: ${L.pp} stages, ${M} micro-batch${M > 1 ? "es" : ""}, ${slots} slots.`
          : gpipeCaption(L.pp, M, shown - 1)
      }
      testId="mech-gpipe"
      equation={equation}
      hl={shown > 0 && grid.some((row) => row[shown - 1] === null) ? "p" : "a"}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Pipeline stages against slots, each slot running a micro-batch or idle"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <defs>
              <Hatch id="gp-bubble" />
            </defs>
            <text x={4} y={14} fontSize={12} fill="currentColor">
              {L.label}, {M} micro-batch{M > 1 ? "es" : ""}: bubble{" "}
              {(100 * bubble(L.pp, M)).toFixed(0)}%
            </text>
            {grid.map((row, j) => (
              <g key={j}>
                <text
                  x={4}
                  y={24 + j * (ch + 6) + 17}
                  fontSize={11}
                  fill="currentColor"
                >
                  stage {j}
                </text>
                {row.map((mb, s) => (
                  <g key={s} opacity={s >= shown ? 0.15 : 1}>
                    <rect
                      x={label + s * cw}
                      y={24 + j * (ch + 6)}
                      width={cw - 3}
                      height={ch}
                      fill={
                        mb === null ? "url(#gp-bubble)" : MB[mb % MB.length]
                      }
                      stroke={mb === null ? "#a3a3a3" : "none"}
                    />
                    {mb !== null && cw > 22 && (
                      <text
                        x={label + s * cw + (cw - 3) / 2}
                        y={24 + j * (ch + 6) + 17}
                        fontSize={11}
                        textAnchor="middle"
                        fill={mb % MB.length === 0 ? "#fff" : "#000"}
                      >
                        {mb}
                      </text>
                    )}
                  </g>
                ))}
              </g>
            ))}
            <text x={4} y={H - 8} fontSize={11} fill="currentColor">
              prefill {fmtSeconds(pipe.prefill.time)} against TP4&apos;s{" "}
              {fmtSeconds(tp4.prefill.time)}
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Prefill step" value={fmtSeconds(pipe.prefill.time)} />
          <Stat label="TP4 prefill" value={fmtSeconds(tp4.prefill.time)} />
          <Stat label="Decode step" value={fmtSeconds(pipe.decode.time)} />
          <Stat label="TP4 decode" value={fmtSeconds(tp4.decode.time)} />
        </div>
      }
      params={
        <>
          <Segmented
            label="Layout on four GPUs"
            value={layout}
            options={(Object.keys(LAYOUTS) as (keyof typeof LAYOUTS)[]).map(
              (k) => ({
                value: k,
                label: LAYOUTS[k].label,
              }),
            )}
            onChange={setLayout}
          />
          <Segmented
            label="Micro-batches m"
            value={m}
            options={["1", "2", "4"].map((x) => ({ value: x, label: x }))}
            onChange={setM}
          />
        </>
      }
    />
  );
}
