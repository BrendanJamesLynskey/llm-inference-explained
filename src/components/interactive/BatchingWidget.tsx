"use client";

/**
 * Four batching policies on the same request stream: throughput against
 * latency, plus a timeline of the first steps of one policy.
 */
import { useMemo, useState } from "react";

import { Segmented, Slider } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  POLICIES,
  makeWorkload,
  simulateBatching,
  type Policy,
} from "@/lib/inference/batching";
import { H100_SXM, LLAMA3_8B, costModel } from "@/lib/inference/costModel";
import { summarise } from "@/lib/inference/metrics";

const cm = costModel(LLAMA3_8B, H100_SXM);
const SLO = { ttft: 1, tpot: 0.025 };
const WINDOW = 2; // seconds of timeline shown

const ms = (s: number): string =>
  Number.isFinite(s) ? `${(s * 1e3).toFixed(0)}` : "–";

export function BatchingWidget(): JSX.Element {
  const [rate, setRate] = useState(6);
  const [maxBatch, setMaxBatch] = useState(16);
  const [chunk, setChunk] = useState(512);
  const [prompt, setPrompt] = useState(1024);
  const [shown, setShown] = useState<Policy>("continuous");

  const results = useMemo(() => {
    const reqs = makeWorkload({
      rate,
      n: 160,
      prompt,
      output: 128,
      cv: 0.6,
      seed: 11,
    });
    return POLICIES.map((policy) => {
      const r = simulateBatching(reqs, {
        policy,
        maxBatch,
        chunk,
        timeout: 0.05,
        cm,
      });
      return { policy, run: r, s: summarise(r.records, SLO) };
    });
  }, [rate, maxBatch, chunk, prompt]);

  const sel = results.find((r) => r.policy === shown)!;
  const firstArrival = sel.run.records.reduce(
    (a, r) => Math.min(a, r.arrival),
    Infinity,
  );
  const t0 = Math.max(0, firstArrival);
  const visible = sel.run.steps.filter((s) => s.start < t0 + WINDOW);
  const bestTok = Math.max(...results.map((r) => r.s.tokPerS));

  return (
    <WidgetFrame
      testId="batching"
      title="Static, dynamic, continuous and chunked batching"
      caption="160 requests (Poisson arrivals, lognormal lengths, cv 0.6, 128 output tokens on average) served by one H100 running Llama-3-8B. Step times come from the roofline cost model; the scheduler is this site's simplified model. SLOs: TTFT 1 s, TPOT 25 ms."
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
        <Slider
          label="Mean prompt"
          value={prompt}
          min={128}
          max={4096}
          step={128}
          onChange={setPrompt}
          format={(v) => `${v} tokens`}
        />
        <Slider
          label="Max batch"
          value={maxBatch}
          min={1}
          max={64}
          onChange={setMaxBatch}
        />
        <Slider
          label="Chunk budget (chunked)"
          value={chunk}
          min={64}
          max={4096}
          step={64}
          onChange={setChunk}
          format={(v) => `${v} tokens/step`}
        />
      </div>

      <div
        className="focus-ring mt-4 overflow-x-auto rounded"
        role="region"
        tabIndex={0}
        aria-label="Results per policy"
      >
        <table className="w-full text-left text-xs">
          <caption className="sr-only">Results per policy</caption>
          <thead>
            <tr className="text-neutral-500 dark:text-neutral-400">
              <th className="py-1 pr-2 font-normal">Policy</th>
              <th className="py-1 pr-2 font-normal">tok/s</th>
              <th className="py-1 pr-2 font-normal">TTFT p50 / p99 (ms)</th>
              <th className="py-1 pr-2 font-normal">ITL p99 (ms)</th>
              <th className="py-1 font-normal">SLO met</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {results.map(({ policy, s }) => (
              <tr
                key={policy}
                className="border-t border-neutral-200 dark:border-neutral-800"
              >
                <td className="py-1 pr-2">{policy}</td>
                <td className="py-1 pr-2">
                  <span className="flex items-center gap-1">
                    <span
                      className="inline-block h-2 rounded bg-indigo-500"
                      style={{ width: `${(s.tokPerS / bestTok) * 48}px` }}
                    />
                    {s.tokPerS.toFixed(0)}
                  </span>
                </td>
                <td className="py-1 pr-2">
                  {ms(s.ttft.p50)} / {ms(s.ttft.p99)}
                </td>
                <td className="py-1 pr-2">{ms(s.itl.p99)}</td>
                <td className="py-1">{(s.sloAttain * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4">
        <Segmented
          label="Timeline"
          value={shown}
          options={POLICIES.map((p) => ({ value: p, label: p }))}
          onChange={setShown}
        />
        <svg
          viewBox="0 0 600 110"
          className="mt-2 h-auto w-full"
          role="img"
          aria-label={`First ${WINDOW} seconds of steps under ${shown} batching`}
        >
          {visible.map((s, i) => {
            const x0 = ((s.start - t0) / WINDOW) * 600;
            const w = Math.max(0.6, (s.time / WINDOW) * 600);
            const cls =
              s.kind === "prefill"
                ? "fill-emerald-500"
                : s.kind === "mixed"
                  ? "fill-amber-500"
                  : "fill-indigo-500";
            const h = 10 + Math.min(70, s.batch * 3);
            return (
              <rect
                key={i}
                x={x0}
                y={92 - h}
                width={Math.min(w, 600 - x0)}
                height={h}
                className={cls}
              />
            );
          })}
          <line
            x1={0}
            x2={600}
            y1={92.5}
            y2={92.5}
            className="stroke-neutral-400"
          />
          <text
            x={0}
            y={106}
            className="fill-neutral-500 text-[11px] dark:fill-neutral-400"
          >
            0 s
          </text>
          <text
            x={600}
            y={106}
            textAnchor="end"
            className="fill-neutral-500 text-[11px] dark:fill-neutral-400"
          >
            {WINDOW} s
          </text>
        </svg>
        <p className="text-xs text-neutral-600 dark:text-neutral-400">
          One bar per step, from the first arrival; height grows with the
          sequences in the step. Green: prefill. Blue: decode. Amber: mixed
          (decodes plus a prompt chunk). Gaps: the instance is idle, waiting.
        </p>
      </div>
    </WidgetFrame>
  );
}
