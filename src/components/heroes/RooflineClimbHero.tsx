"use client";

/**
 * Chapter 3's hero: the decode step climbing the roofline as the batch
 * grows. Each point is one decode step of Llama-3-8B on one H100 with
 * every sequence at the chosen context, priced by the cost model
 * (`rooflineClimb`): batching shares the weight reads, so the intensity
 * rises along the slanted (memory) roof; each sequence still reads its own
 * KV cache, so the climb flattens below the ridge. The prefill point sits
 * past the ridge for contrast.
 */
import { scaleLog } from "d3";
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import {
  ROOF_DEFAULT,
  roofCaption,
  rooflineClimb,
} from "@/lib/inference/heroes";
import { OKABE_ITO, PHASE_COLOUR } from "@/lib/viz/palette";

const CTX = ["512", "2048", "8192"] as const;

const tf = (x: number): string => `${(x / 1e12).toFixed(x >= 1e14 ? 0 : 1)}`;

export default function RooflineClimbHero({
  equation,
}: {
  equation?: ReactNode;
}): JSX.Element {
  const [ctx, setCtx] = useState<(typeof CTX)[number]>("2048");
  const c = useMemo(
    () => rooflineClimb({ ...ROOF_DEFAULT, ctx: Number(ctx) }),
    [ctx],
  );
  const st = useStepper(c.points.length, { stepMs: 900, resetKey: ctx });
  const [ref, width] = useWidth(640);
  const k = st.step;
  const p = c.points[k]!;
  const W = width;
  const H = W < 520 ? 250 : 290;
  const M = { l: 48, r: 12, t: 14, b: 38 };
  const x = scaleLog()
    .domain([0.5, 5000])
    .range([M.l, W - M.r]);
  const y = scaleLog()
    .domain([3e11, c.peak * 2.5])
    .range([H - M.b, M.t]);
  const roof = (i: number) => Math.min(c.peak, i * c.bandwidth);
  const xs: number[] = [];
  for (let e = Math.log10(0.5); e <= Math.log10(5000) + 1e-9; e += 0.02)
    xs.push(10 ** e);
  const path = xs
    .map(
      (i, j) => `${j ? "L" : "M"}${x(i).toFixed(1)},${y(roof(i)).toFixed(1)}`,
    )
    .join("");
  const ticksX = [1, 10, 100, 1000];
  const ticksY = [1e12, 1e13, 1e14, 1e15].filter((v) => v < c.peak * 2.5);
  return (
    <AnimationPanel
      testId="hero-roofline"
      title="Batching walks decode up the roofline, until the KV reads stop it"
      summary={
        <>
          Llama-3-8B decode steps on one H100, every sequence at{" "}
          {Number(ctx).toLocaleString("en-GB")} tokens of context, batch 1 to
          256. The roof is the H100&apos;s derated peaks; each point is a step
          the cost model priced. Orange: one{" "}
          {Number(ctx).toLocaleString("en-GB")}-token prefill.
        </>
      }
      stepper={st}
      stepLabel="batch size"
      caption={roofCaption(c, k)}
      equation={equation}
      hl={p.intensity < c.ridge ? "bw" : "p"}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="A roofline: attainable FLOP per second against arithmetic intensity, log-log, with decode points for growing batch sizes climbing the slanted memory roof and a prefill point on the flat compute roof"
            className="text-neutral-800 dark:text-neutral-200"
          >
            {ticksX.map((t) => (
              <g key={`x${t}`}>
                <line
                  x1={x(t)}
                  x2={x(t)}
                  y1={M.t}
                  y2={H - M.b}
                  stroke="#a3a3a3"
                  strokeOpacity={0.3}
                />
                <text
                  x={x(t)}
                  y={H - M.b + 14}
                  fontSize={11}
                  textAnchor="middle"
                  fill="currentColor"
                >
                  {t.toLocaleString("en-GB")}
                </text>
              </g>
            ))}
            {ticksY.map((t) => (
              <g key={`y${t}`}>
                <line
                  x1={M.l}
                  x2={W - M.r}
                  y1={y(t)}
                  y2={y(t)}
                  stroke="#a3a3a3"
                  strokeOpacity={0.3}
                />
                <text
                  x={M.l - 4}
                  y={y(t) + 4}
                  fontSize={11}
                  textAnchor="end"
                  fill="currentColor"
                >
                  {tf(t)}
                </text>
              </g>
            ))}
            <text x={M.l} y={H - 6} fontSize={11} fill="currentColor">
              FLOP per byte →
            </text>
            <text x={4} y={M.t + 2} fontSize={11} fill="currentColor">
              TFLOP/s
            </text>
            <path d={path} fill="none" stroke="currentColor" strokeWidth={2} />
            <line
              x1={x(c.ridge)}
              x2={x(c.ridge)}
              y1={y(c.peak)}
              y2={H - M.b}
              stroke="currentColor"
              strokeDasharray="3 3"
            />
            <text
              x={x(c.ridge) + 4}
              y={H - M.b - 6}
              fontSize={11}
              fill="currentColor"
            >
              ridge {c.ridge.toFixed(0)}
            </text>
            {c.points.map((q, j) => (
              <circle
                key={q.batch}
                data-testid={`roof-point-${q.batch}`}
                cx={x(q.intensity)}
                cy={y(q.attainable)}
                r={j === k ? 7 : 4}
                fill={PHASE_COLOUR.decode}
                opacity={j > k ? 0.15 : j === k ? 1 : 0.5}
                stroke={j === k ? "currentColor" : "none"}
              />
            ))}
            <text
              x={x(p.intensity) + 9}
              y={y(p.attainable) + 4}
              fontSize={11}
              fill="currentColor"
            >
              batch {p.batch}
            </text>
            <rect
              x={x(c.prefill.intensity) - 5}
              y={y(c.prefill.attainable) - 5}
              width={10}
              height={10}
              fill={OKABE_ITO.orange}
            />
            <text
              x={x(c.prefill.intensity)}
              y={y(c.prefill.attainable) + 20}
              fontSize={11}
              textAnchor="middle"
              fill="currentColor"
            >
              prefill
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Batch" value={String(p.batch)} />
          <Stat label="FLOP per byte" value={p.intensity.toFixed(1)} />
          <Stat label="Step time" value={`${(p.time * 1e3).toFixed(1)} ms`} />
          <Stat
            label="Tokens/s"
            value={Math.round(p.tokensPerS).toLocaleString("en-GB")}
          />
        </div>
      }
      params={
        <Segmented
          label="Context per sequence"
          value={ctx}
          options={CTX.map((v) => ({
            value: v,
            label: Number(v).toLocaleString("en-GB"),
          }))}
          onChange={setCtx}
        />
      }
    />
  );
}
