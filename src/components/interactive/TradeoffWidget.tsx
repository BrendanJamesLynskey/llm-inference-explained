"use client";

/**
 * When does disaggregation pay? A load sweep, colocated against 1P1D on the
 * same GPUs, run by the simulator's engine on the recorded workloads: SLO
 * attainment and energy per token at every offered load.
 */
import { scaleLinear } from "d3";
import { useMemo, useState } from "react";

import { Segmented } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { FORMATS, type Metrics, type SimConfig } from "@/lib/disagg/engine";
import {
  configs,
  DEFAULT_CONTROLS,
  type SimLink,
} from "@/lib/disagg/simulator";
import { WORKLOAD_RATES } from "@/lib/disagg/workloads";

import { RunStatus, useRuns } from "./disagg/useRuns";

const W = 560;
const H = 200;
const M = { l: 44, r: 10, t: 10, b: 30 };

type Key = `${"c" | "d"}${(typeof WORKLOAD_RATES)[number]}`;

function Chart({
  title,
  ys,
  y,
  fmt,
}: {
  title: string;
  ys: { c: number[]; d: number[] };
  y: ReturnType<typeof scaleLinear<number, number>>;
  fmt: (v: number) => string;
}): JSX.Element {
  const x = scaleLinear()
    .domain([2, 14])
    .range([M.l, W - M.r]);
  const path = (v: number[]) =>
    v
      .map(
        (val, i) =>
          `${i ? "L" : "M"}${x(WORKLOAD_RATES[i]!).toFixed(1)},${y(val).toFixed(1)}`,
      )
      .join("");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label={`${title}: colocated ${ys.c.map(fmt).join(", ")}; disaggregated ${ys.d.map(fmt).join(", ")} at ${WORKLOAD_RATES.join(", ")} req/s`}
    >
      {y.ticks(4).map((t) => (
        <g key={t}>
          <line
            x1={M.l}
            x2={W - M.r}
            y1={y(t)}
            y2={y(t)}
            className="stroke-neutral-200 dark:stroke-neutral-800"
          />
          <text
            x={M.l - 4}
            y={y(t) + 3}
            textAnchor="end"
            className="fill-neutral-600 text-[10px] dark:fill-neutral-400"
          >
            {fmt(t)}
          </text>
        </g>
      ))}
      {WORKLOAD_RATES.map((r) => (
        <text
          key={r}
          x={x(r)}
          y={H - M.b + 14}
          textAnchor="middle"
          className="fill-neutral-600 text-[10px] dark:fill-neutral-400"
        >
          {r}
        </text>
      ))}
      <text
        x={(M.l + W) / 2}
        y={H - 2}
        textAnchor="middle"
        className="fill-neutral-600 text-[10px] dark:fill-neutral-400"
      >
        offered load (req/s)
      </text>
      <path
        d={path(ys.c)}
        fill="none"
        strokeWidth={2}
        className="stroke-rose-500"
      />
      <path
        d={path(ys.d)}
        fill="none"
        strokeWidth={2}
        className="stroke-indigo-500"
      />
      {ys.c.map((v, i) => (
        <circle
          key={`c${i}`}
          cx={x(WORKLOAD_RATES[i]!)}
          cy={y(v)}
          r={3}
          className="fill-rose-500"
        />
      ))}
      {ys.d.map((v, i) => (
        <rect
          key={`d${i}`}
          x={x(WORKLOAD_RATES[i]!) - 3}
          y={y(v) - 3}
          width={6}
          height={6}
          className="fill-indigo-500"
        />
      ))}
    </svg>
  );
}

export function TradeoffWidget(): JSX.Element {
  const [model, setModel] = useState<"llama3-8b" | "llama3-70b">("llama3-8b");
  const [link, setLink] = useState<SimLink>("ib-ndr");
  const cfgs = useMemo(() => {
    const { colocated, disagg } = configs({ ...DEFAULT_CONTROLS, model, link });
    const out = {} as Record<Key, SimConfig>;
    for (const r of WORKLOAD_RATES) {
      out[`c${r}`] = colocated;
      out[`d${r}`] = disagg;
    }
    return out;
  }, [model, link]);

  return (
    <WidgetFrame
      testId="tradeoffs"
      title="A load sweep: colocated ×2 against 1P1D"
      caption="Two colocated instances against one prefill + one decode instance, the same GPUs (Llama-3-8B on 1 H100 per instance, or Llama-3-70B on 4), at nine offered loads. Each point is a full 800-request run of the simulator's engine. SLOs: TTFT 1 s, TPOT 25 ms."
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Segmented
          label="Model"
          value={model}
          options={[
            { value: "llama3-8b", label: "Llama-3-8B ×1" },
            { value: "llama3-70b", label: "Llama-3-70B ×4" },
          ]}
          onChange={setModel}
        />
        <Segmented
          label="KV link"
          value={link}
          options={[
            { value: "ib-ndr", label: "ib-ndr" },
            { value: "eth-100g", label: "eth-100g" },
            { value: "eth-25g", label: "eth-25g" },
          ]}
          onChange={setLink}
        />
      </div>
      <Sweep cfgs={cfgs} />
    </WidgetFrame>
  );
}

/** Runs the sweep one load at a time (each load's workload is a separate file). */
function Sweep({ cfgs }: { cfgs: Record<Key, SimConfig> }): JSX.Element {
  const states = WORKLOAD_RATES.map((r) =>
    // eslint-disable-next-line react-hooks/rules-of-hooks -- fixed-length list
    useRuns(r, { c: cfgs[`c${r}`], d: cfgs[`d${r}`] }),
  );
  for (const s of states)
    if (s.status !== "ready") return <RunStatus state={s} />;
  const ms: { c: Metrics[]; d: Metrics[] } = { c: [], d: [] };
  for (const s of states)
    if (s.status === "ready") {
      ms.c.push(s.runs.c.metrics);
      ms.d.push(s.runs.d.metrics);
    }
  const slo = { c: ms.c.map((m) => m.slo), d: ms.d.map((m) => m.slo) };
  const jt = { c: ms.c.map((m) => m.j_tok), d: ms.d.map((m) => m.j_tok) };
  const jmax = Math.max(...jt.c, ...jt.d);
  const wins = slo.d.filter((v, i) => v > slo.c[i]!).length;
  const ties = slo.d.filter((v, i) => v === slo.c[i]!).length;
  return (
    <>
      <p className="mt-4 flex flex-wrap gap-x-4 text-xs text-neutral-700 dark:text-neutral-300">
        <span className="flex items-center gap-1">
          <span className="inline-block size-2 rounded-full bg-rose-500" />{" "}
          colocated ×2
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block size-2 bg-indigo-500" /> disaggregated
          1P1D
        </span>
      </p>
      <p className="mt-2 text-xs font-medium text-neutral-700 dark:text-neutral-300">
        SLO attainment
      </p>
      <Chart
        title="SLO attainment"
        ys={slo}
        y={scaleLinear()
          .domain([0, 1])
          .range([H - M.b, M.t])}
        fmt={(v) => `${Math.round(v * 100)}%`}
      />
      <p className="mt-2 text-xs font-medium text-neutral-700 dark:text-neutral-300">
        Energy per output token (J)
      </p>
      <Chart
        title="Joules per output token"
        ys={jt}
        y={scaleLinear()
          .domain([0, jmax * 1.1])
          .nice()
          .range([H - M.b, M.t])}
        fmt={(v) => (jmax < 1 ? v.toFixed(2) : v.toFixed(1))}
      />
      <p
        role="status"
        className="mt-2 text-sm text-neutral-700 dark:text-neutral-300"
      >
        Disaggregation meets more SLOs at {wins} of {WORKLOAD_RATES.length}{" "}
        loads
        {ties ? ` (tied at ${ties})` : ""}; colocated wins at{" "}
        {WORKLOAD_RATES.length - wins - ties}. TTFT p99 at 8 req/s: colocated{" "}
        {FORMATS.ms(ms.c[5]!.ttft_p99)}, disaggregated{" "}
        {FORMATS.ms(ms.d[5]!.ttft_p99)}.
      </p>
    </>
  );
}
