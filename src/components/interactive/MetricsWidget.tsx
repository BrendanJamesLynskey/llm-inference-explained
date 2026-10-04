"use client";

/**
 * Serving metrics from a simulated run: TTFT and TPOT distributions against
 * SLOs, goodput, and Little's law measured two ways.
 */
import { scaleLinear } from "d3";
import { useMemo, useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  makeWorkload,
  simulateBatching,
  type Policy,
} from "@/lib/inference/batching";
import { H100_SXM, LLAMA3_8B, costModel } from "@/lib/inference/costModel";
import { summarise, tpotOf, ttftOf } from "@/lib/inference/metrics";

const cm = costModel(LLAMA3_8B, H100_SXM);

function Cdf({
  xs,
  slo,
  label,
  unit,
}: {
  xs: number[];
  slo: number;
  label: string;
  unit: number;
}): JSX.Element {
  const s = xs.slice().sort((a, b) => a - b);
  const max = Math.max(slo * 1.5, s[Math.floor(s.length * 0.995)] ?? 0);
  const x = scaleLinear().domain([0, max]).range([30, 290]);
  const y = scaleLinear().domain([0, 1]).range([110, 10]);
  const d = s
    .map(
      (v, i) =>
        `${i ? "L" : "M"}${x(Math.min(v, max)).toFixed(1)},${y((i + 1) / s.length).toFixed(1)}`,
    )
    .join("");
  const met = s.filter((v) => v <= slo).length / Math.max(1, s.length);
  return (
    <svg
      viewBox="0 0 300 130"
      className="h-auto w-full"
      role="img"
      aria-label={`${label} CDF; ${(met * 100).toFixed(0)}% within the SLO`}
    >
      {[0, 0.5, 0.9, 0.99, 1].map((q) => (
        <g key={q}>
          <line
            x1={30}
            x2={290}
            y1={y(q)}
            y2={y(q)}
            className="stroke-neutral-200 dark:stroke-neutral-800"
          />
          <text
            x={26}
            y={y(q) + 3}
            textAnchor="end"
            className="fill-neutral-500 text-[8px] dark:fill-neutral-400"
          >
            {q}
          </text>
        </g>
      ))}
      <path d={d} fill="none" strokeWidth={2} className="stroke-indigo-500" />
      <line
        x1={x(slo)}
        x2={x(slo)}
        y1={10}
        y2={110}
        strokeDasharray="4 3"
        className="stroke-rose-500"
      />
      <text
        x={x(slo) + 3}
        y={20}
        className="fill-rose-600 text-[9px] dark:fill-rose-400"
      >
        SLO
      </text>
      <text
        x={160}
        y={126}
        textAnchor="middle"
        className="fill-neutral-600 text-[9px] dark:fill-neutral-400"
      >
        {label} (ms), 0 – {(max * unit).toFixed(0)}
      </text>
    </svg>
  );
}

export function MetricsWidget(): JSX.Element {
  const [rate, setRate] = useState(6);
  const [ttftSlo, setTtftSlo] = useState(500);
  const [tpotSlo, setTpotSlo] = useState(25);
  const [policy, setPolicy] = useState<Policy>("continuous");

  const run = useMemo(() => {
    const reqs = makeWorkload({
      rate,
      n: 300,
      prompt: 1024,
      output: 128,
      cv: 0.6,
      seed: 4,
    });
    return simulateBatching(reqs, {
      policy,
      maxBatch: 32,
      chunk: 512,
      timeout: 0.05,
      cm,
    });
  }, [rate, policy]);
  const slo = { ttft: ttftSlo / 1e3, tpot: tpotSlo / 1e3 };
  const s = summarise(run.records, slo);
  const ttfts = run.records.map(ttftOf);
  const tpots = run.records.filter((r) => r.output >= 2).map(tpotOf);

  return (
    <WidgetFrame
      testId="metrics"
      title="Percentiles, SLOs, goodput and Little's law"
      caption="300 requests on one H100 running Llama-3-8B (the batching chapter's simplified scheduler, max batch 32). Percentiles interpolate linearly, as Disaggregated_Inference_Sim's do."
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Slider
          label="Arrival rate"
          value={rate}
          min={1}
          max={14}
          onChange={setRate}
          format={(v) => `${v} req/s`}
        />
        <Segmented
          label="Scheduler"
          value={policy}
          options={[
            { value: "continuous", label: "continuous" },
            { value: "chunked", label: "chunked" },
          ]}
          onChange={setPolicy}
        />
        <Slider
          label="TTFT SLO"
          value={ttftSlo}
          min={100}
          max={2000}
          step={50}
          onChange={setTtftSlo}
          format={(v) => `${v} ms`}
        />
        <Slider
          label="TPOT SLO"
          value={tpotSlo}
          min={5}
          max={60}
          onChange={setTpotSlo}
          format={(v) => `${v} ms`}
        />
      </div>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Cdf xs={ttfts} slo={slo.ttft} label="TTFT" unit={1e3} />
        <Cdf xs={tpots} slo={slo.tpot} label="TPOT" unit={1e3} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="TTFT p50 / p99"
          value={`${(s.ttft.p50 * 1e3).toFixed(0)} / ${(s.ttft.p99 * 1e3).toFixed(0)} ms`}
        />
        <Stat
          label="TPOT p50 / p99"
          value={`${(s.tpot.p50 * 1e3).toFixed(1)} / ${(s.tpot.p99 * 1e3).toFixed(1)} ms`}
        />
        <Stat
          label="ITL p99 / max"
          value={`${(s.itl.p99 * 1e3).toFixed(1)} / ${(s.itl.max * 1e3).toFixed(0)} ms`}
        />
        <Stat label="Throughput" value={`${s.tokPerS.toFixed(0)} tok/s`} />
        <Stat
          label="SLO met"
          value={`${(s.sloAttain * 100).toFixed(1)}%`}
          hint="both TTFT and TPOT"
        />
        <Stat
          label="Goodput"
          value={`${s.goodput.toFixed(2)} req/s`}
          hint={`offered ${rate} req/s`}
        />
        <Stat
          plain
          label="L (measured)"
          value={s.little.L.toFixed(3)}
          hint="time-average in system"
        />
        <Stat
          plain
          label="λ·W"
          value={s.little.lambdaW.toFixed(3)}
          hint="arrival rate × mean latency"
        />
      </div>
    </WidgetFrame>
  );
}
