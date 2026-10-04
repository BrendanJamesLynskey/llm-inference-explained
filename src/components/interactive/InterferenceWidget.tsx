"use client";

/**
 * Prefill/decode interference: the same request stream on two colocated
 * instances and on one prefill + one decode instance (the same four GPUs
 * each), run live by Disaggregated_Inference_Sim's own engine on the
 * workload results.md §4 used. One request's token gaps are drawn in both
 * modes, so the stalls a colocated prefill causes are visible.
 */
import { scaleLinear } from "d3";
import { useMemo, useState } from "react";

import { Segmented, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { DEFAULTS, FORMATS, type SimResult } from "@/lib/disagg/engine";
import type { WorkloadRate } from "@/lib/disagg/workloads";

import { RunStatus, useRuns, type Run } from "./disagg/useRuns";

const RATES = [2, 3, 4, 5, 6, 8] as const;
const W = 560;
const ROW = 70;

/** The steady-state request whose worst colocated token gap is largest. */
function worstRequest(res: Pick<SimResult, "reqs">): number {
  const done = res.reqs.filter((r) => r.finish !== null);
  const skip = Math.floor(done.length * 0.1);
  let best = 0;
  let worst = -1;
  res.reqs.forEach((r, i) => {
    if (i < skip || r.itls.length < 2) return;
    const m = Math.max(...r.itls);
    if (m > worst) {
      worst = m;
      best = i;
    }
  });
  return best;
}

function Gaps({
  gaps,
  y,
  label,
  cls,
  max,
}: {
  gaps: number[];
  y: number;
  label: string;
  cls: string;
  max: number;
}): JSX.Element {
  const x = scaleLinear()
    .domain([0, Math.max(1, gaps.length)])
    .range([60, W - 8]);
  const h = scaleLinear()
    .domain([0, max])
    .range([0, ROW - 18]);
  const bw = Math.max(0.6, x(1) - x(0) - 0.3);
  return (
    <g>
      <text
        x={0}
        y={y + ROW - 8}
        className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
      >
        {label}
      </text>
      <line
        x1={60}
        x2={W - 8}
        y1={y + ROW - 4}
        y2={y + ROW - 4}
        className="stroke-neutral-300 dark:stroke-neutral-700"
      />
      {gaps.map((g, i) => (
        <rect
          key={i}
          x={x(i)}
          width={bw}
          y={y + ROW - 4 - h(g)}
          height={Math.max(0.5, h(g))}
          className={cls}
        />
      ))}
    </g>
  );
}

export function InterferenceWidget(): JSX.Element {
  const [rate, setRate] = useState<WorkloadRate>(4);
  const cfgs = useMemo(
    () => ({
      colocated: { ...DEFAULTS, mode: "colocated" as const },
      disagg: DEFAULTS,
    }),
    [],
  );
  const state = useRuns(rate, cfgs);

  return (
    <WidgetFrame
      testId="interference"
      title="One request stream, colocated and disaggregated"
      caption="Llama-3-70B, 4× H100 per instance: two colocated instances against one prefill + one decode instance over InfiniBand NDR. 800 requests (prompts ~2,048 tokens, outputs ~256), the recorded workload of results.md §4, run by the simulator's own engine. SLOs: TTFT 1 s, TPOT 25 ms."
    >
      <Segmented
        label="Offered load (req/s)"
        value={String(rate)}
        options={RATES.map((r) => ({ value: String(r), label: String(r) }))}
        onChange={(v) => setRate(Number(v) as WorkloadRate)}
      />
      {state.status !== "ready" ? (
        <RunStatus state={state} />
      ) : (
        <Results rate={rate} runs={state.runs} />
      )}
    </WidgetFrame>
  );
}

function Results({
  rate,
  runs,
}: {
  rate: number;
  runs: Record<"colocated" | "disagg", Run>;
}): JSX.Element {
  const c = runs.colocated.metrics;
  const d = runs.disagg.metrics;
  const idx = worstRequest(runs.colocated.result);
  const gc = runs.colocated.result.reqs[idx]!.itls;
  const gd = runs.disagg.result.reqs[idx]!.itls;
  const max = Math.max(...gc, ...gd);
  const ratio = c.itl_p99 / d.itl_p99;
  const { ms, pct } = FORMATS;
  return (
    <>
      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="ITL p99, colocated"
          value={ms(c.itl_p99)}
          hint={`max ${ms(c.itl_max)}`}
        />
        <Stat
          label="ITL p99, disaggregated"
          value={ms(d.itl_p99)}
          hint={`max ${ms(d.itl_max)}`}
        />
        <Stat
          label="TTFT p99, colocated / disagg."
          value={`${ms(c.ttft_p99)} / ${ms(d.ttft_p99)}`}
        />
        <Stat
          label="SLO met, colocated / disagg."
          value={`${pct(c.slo)} / ${pct(d.slo)}`}
        />
      </div>
      <p
        role="status"
        className="mt-3 text-sm text-neutral-700 dark:text-neutral-300"
      >
        At {rate} req/s, disaggregation cuts p99 inter-token latency{" "}
        <strong>{ratio.toFixed(1)}×</strong>; TPOT p99 {ms(c.tpot_p99)} →{" "}
        {ms(d.tpot_p99)}.
      </p>
      <svg
        viewBox={`0 0 ${W} ${2 * ROW + 16}`}
        className="mt-3 h-auto w-full"
        role="img"
        aria-label={`Token gaps of request ${idx}: largest ${ms(Math.max(...gc))} colocated, ${ms(Math.max(...gd))} disaggregated`}
      >
        <Gaps gaps={gc} y={0} max={max} label="colocated" cls="fill-rose-500" />
        <Gaps
          gaps={gd}
          y={ROW}
          max={max}
          label="disagg."
          cls="fill-indigo-500"
        />
        <text
          x={W / 2}
          y={2 * ROW + 14}
          textAnchor="middle"
          className="fill-neutral-600 text-[10px] dark:fill-neutral-400"
        >
          request {idx}: each bar is the gap before one output token (tallest{" "}
          {ms(max)})
        </text>
      </svg>
    </>
  );
}
