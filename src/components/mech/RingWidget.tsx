"use client";

/**
 * Chapter 8's first mechanism: the tensor-parallel all-reduce. Each GPU
 * holds a partial result cut into n chunks; in the reduce-scatter half each
 * passes one chunk to its neighbour, which adds it, until each GPU owns one
 * complete chunk; in the all-gather half the complete chunks go round.
 * Underneath, the vendored engine's cost model prices one decode step of
 * Llama-3-70B on n H100s: its compute and its all-reduces, live for the n
 * chosen (two all-reduces per layer, not overlapped with compute).
 */
import { useMemo, useState } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import { tpSteps } from "@/lib/tradeoffs/costs";
import { engine } from "@/lib/tradeoffs/engine";
import {
  fmtSeconds,
  ringCaption,
  ringStates,
  ringStepTime,
} from "@/lib/tradeoffs/mech";
import { OKABE_ITO } from "@/lib/viz/palette";

const CHUNK = [
  OKABE_ITO.blue,
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.purple,
  OKABE_ITO.sky,
  OKABE_ITO.vermillion,
  OKABE_ITO.yellow,
  "#737373",
];

export default function RingWidget({
  equation,
}: {
  equation?: React.ReactNode;
}): JSX.Element {
  const [n, setN] = useState("4");
  const [batch, setBatch] = useState("1");
  const N = Number(n);
  const states = useMemo(() => ringStates(N), [N]);
  const steps = useMemo(() => tpSteps(N), [N]);
  const stepper = useStepper(states.length, { stepMs: 1100, resetKey: n });
  const [ref, width] = useWidth(640);
  const s = states[Math.min(stepper.step, states.length - 1)]!;
  const W = width;
  const narrow = W < 560;
  const gpuW = narrow
    ? Math.min(78, (W - 16) / Math.min(N, 4) - 8)
    : Math.min(120, (W - 16) / N - 10);
  const perRow = narrow ? Math.min(N, 4) : N;
  const cellH = 12;
  const boxH = 20 + N * (cellH + 2);
  const rowsOfGpus = Math.ceil(N / perRow);
  const gpuX = (i: number) => 8 + (i % perRow) * (gpuW + 10);
  const gpuY = (i: number) => 24 + Math.floor(i / perRow) * (boxH + 26);
  const top2 = 24 + rowsOfGpus * (boxH + 26) + 8;
  const dec = batch === "1" ? steps.decodeB1 : steps.decodeB64;
  const comm = dec.commT ?? 0;
  const barW = W - 16;
  const H = top2 + 64;
  // one ring step's time for this decode step's all-reduce (activations of the batch)
  const d = Number(engine.derive(engine.MODELS["llama3-70b"]!).d);
  const link = engine.LINKS.nvlink4!;
  const bytes = (batch === "1" ? 1 : 64) * d * 2;
  const tStep = ringStepTime(bytes, N, link.bw, link.lat);
  return (
    <AnimationPanel
      title="A ring all-reduce, and what it costs a decode step"
      summary={
        <>
          Chunks pass round the ring (shade: contributions summed so far).
          Below: one decode step, compute against all-reduce, priced live.
        </>
      }
      stepper={stepper}
      stepLabel="ring step"
      countFrom={0}
      caption={ringCaption(s, N)}
      testId="mech-ring"
      equation={equation}
      hl={
        s.phase === "reduce-scatter"
          ? "b"
          : s.phase === "all-gather"
            ? "n"
            : "t"
      }
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Each GPU's chunks during a ring all-reduce, and one decode step split into compute and all-reduce time"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <text x={8} y={14} fontSize={12} fill="currentColor">
              {s.phase === "start" ? "Before" : s.phase}: {N} GPUs in a ring
            </text>
            {s.have.map((chunks, g) => (
              <g key={g}>
                <rect
                  x={gpuX(g)}
                  y={gpuY(g)}
                  width={gpuW}
                  height={boxH}
                  rx={4}
                  fill="none"
                  stroke="currentColor"
                  strokeOpacity={0.4}
                />
                <text
                  x={gpuX(g) + 4}
                  y={gpuY(g) + 13}
                  fontSize={11}
                  fill="currentColor"
                >
                  GPU {g}
                </text>
                {chunks.map((c, j) => (
                  <rect
                    key={j}
                    x={gpuX(g) + 4}
                    y={gpuY(g) + 18 + j * (cellH + 2)}
                    width={(gpuW - 8) * (c / N)}
                    height={cellH}
                    fill={CHUNK[j % CHUNK.length]}
                    stroke={
                      s.sends.some((x) => x.to === g && x.chunk === j)
                        ? "currentColor"
                        : "none"
                    }
                    strokeWidth={1.5}
                  >
                    <title>{`GPU ${g}, chunk ${j}: ${c} of ${N} contributions`}</title>
                  </rect>
                ))}
                <text
                  x={gpuX(g) + gpuW / 2}
                  y={gpuY(g) + boxH + 16}
                  fontSize={11}
                  textAnchor="middle"
                  fill="currentColor"
                >
                  {s.sends.length ? `→ GPU ${(g + 1) % N}` : ""}
                </text>
              </g>
            ))}
            <text x={8} y={top2 + 10} fontSize={12} fill="currentColor">
              Decode step, batch {batch}: {fmtSeconds(dec.time)} (
              {fmtSeconds(comm)} all-reduce)
            </text>
            <rect
              x={8}
              y={top2 + 18}
              width={barW}
              height={18}
              fill={OKABE_ITO.purple}
              opacity={0.75}
            />
            <rect
              x={8 + barW * (1 - comm / dec.time)}
              y={top2 + 18}
              width={barW * (comm / dec.time)}
              height={18}
              fill={OKABE_ITO.vermillion}
            />
            <text x={8} y={top2 + 52} fontSize={11} fill="currentColor">
              purple: compute and memory · vermillion: all-reduces (
              {((100 * comm) / dec.time).toFixed(0)}%)
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="One ring step"
            value={`${(tStep * 1e6).toFixed(1)} µs`}
          />
          <Stat label="Ring steps" value={String(2 * (N - 1))} />
          <Stat label="Decode b=64" value={fmtSeconds(steps.decodeB64.time)} />
          <Stat
            label="KV room"
            value={`${steps.kvTokens.toLocaleString("en-GB")} tok`}
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Tensor-parallel GPUs"
            value={n}
            options={["2", "4", "8"].map((x) => ({
              value: x,
              label: `TP${x}`,
            }))}
            onChange={setN}
          />
          <Segmented
            label="Decode batch"
            value={batch}
            options={[
              { value: "1", label: "1 row" },
              { value: "64", label: "64 rows" },
            ]}
            onChange={setBatch}
          />
        </>
      }
    />
  );
}
