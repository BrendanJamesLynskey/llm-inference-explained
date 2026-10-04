"use client";

/**
 * The live disaggregated simulator: Disaggregated_Inference_Sim's own
 * JavaScript engine (vendored at a pinned commit, bit-exact with the Python
 * package) on the recorded workloads. Every change reruns a colocated and a
 * disaggregated cluster on the same GPUs and the same requests. Presets
 * reproduce rows of the simulator's results.md, and say so.
 */
import { useId, useMemo, useState } from "react";

import { Segmented, Slider } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { FORMATS, type Metrics } from "@/lib/disagg/engine";
import {
  DECODE_DEVICES,
  DEFAULT_CONTROLS,
  PREFILL_DEVICES,
  PRESETS,
  SIM_LINKS,
  SIM_MODELS,
  colocatedDevice,
  configs,
  type Controls,
  type DecodeDevice,
  type PrefillDevice,
  type SimModel,
} from "@/lib/disagg/simulator";
import { WORKLOAD_RATES, type WorkloadRate } from "@/lib/disagg/workloads";

import { RunStatus, useRuns, type Run } from "./disagg/useRuns";

const { ms, ms2, pct } = FORMATS;

type Row = {
  label: string;
  key: keyof Metrics;
  show: (m: Metrics) => string;
  disaggOnly?: boolean;
};

const ROWS: Row[] = [
  { label: "TTFT p99", key: "ttft_p99", show: (m) => ms2(m.ttft_p99) },
  { label: "TPOT p99", key: "tpot_p99", show: (m) => ms(m.tpot_p99) },
  { label: "ITL p99", key: "itl_p99", show: (m) => ms(m.itl_p99) },
  {
    label: "Hand-off p99",
    key: "handoff_p99",
    show: (m) => ms2(m.handoff_p99),
    disaggOnly: true,
  },
  { label: "SLO met", key: "slo", show: (m) => pct(m.slo) },
  {
    label: "Goodput",
    key: "goodput",
    show: (m) => `${m.goodput.toFixed(2)} req/s`,
  },
  { label: "J / token", key: "j_tok", show: (m) => m.j_tok.toFixed(3) },
  {
    label: "KV link busy",
    key: "link_util",
    show: (m) => pct(m.link_util),
    disaggOnly: true,
  },
  { label: "Hot-spot", key: "hotspot", show: (m) => m.hotspot },
];

export function DisaggSimulatorWidget(): JSX.Element {
  const [c, setC] = useState<Controls>(DEFAULT_CONTROLS);
  const [presetId, setPresetId] = useState<string>("sweep-4");
  const preset = PRESETS.find((p) => p.id === presetId);
  const set = <K extends keyof Controls>(k: K, v: Controls[K]): void => {
    setC((old) => ({ ...old, [k]: v }));
    setPresetId("");
  };
  const cfgs = useMemo(() => configs(c), [c]);
  const state = useRuns(c.rate, cfgs);
  const selectId = useId();
  const n = SIM_MODELS[c.model].devices;

  return (
    <WidgetFrame
      testId="disagg-sim"
      title="Disaggregated serving, simulated live"
      caption="Disaggregated_Inference_Sim's JavaScript engine (vendored at a pinned commit; tested request by request against the Python package) on recorded Poisson workloads: 800 requests, prompts ~2,048 tokens, outputs ~256, seed 1. The colocated column runs the same number of instances on the same GPUs."
    >
      <div className="flex flex-col gap-1 text-sm">
        <label
          htmlFor={selectId}
          className="text-xs font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400"
        >
          Preset (a row of results.md)
        </label>
        <select
          id={selectId}
          value={presetId}
          onChange={(e) => {
            const p = PRESETS.find((q) => q.id === e.target.value);
            setPresetId(e.target.value);
            if (p) setC(p.controls);
          }}
          className="focus-ring max-w-full rounded border border-neutral-300 bg-white px-2 py-1 text-sm dark:border-neutral-700 dark:bg-neutral-950"
        >
          <option value="">Custom</option>
          {PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Segmented
          label="Model"
          value={c.model}
          options={Object.entries(SIM_MODELS).map(([k, v]) => ({
            value: k as SimModel,
            label: `${v.label} ×${v.devices}`,
          }))}
          onChange={(v) => set("model", v)}
        />
        <Segmented
          label="Offered load (req/s)"
          value={String(c.rate)}
          options={WORKLOAD_RATES.map((r) => ({
            value: String(r),
            label: String(r),
          }))}
          onChange={(v) => set("rate", Number(v) as WorkloadRate)}
        />
        <Segmented
          label="Prefill pool device"
          value={c.prefillDevice}
          options={Object.entries(PREFILL_DEVICES).map(([k, v]) => ({
            value: k as PrefillDevice,
            label: v,
          }))}
          onChange={(v) => set("prefillDevice", v)}
        />
        <Segmented
          label="Decode pool device"
          value={c.decodeDevice}
          options={Object.entries(DECODE_DEVICES).map(([k, v]) => ({
            value: k as DecodeDevice,
            label: v,
          }))}
          onChange={(v) => set("decodeDevice", v)}
        />
        <Slider
          label="Prefill instances"
          value={c.nPrefill}
          min={1}
          max={3}
          onChange={(v) => set("nPrefill", v)}
        />
        <Slider
          label="Decode instances"
          value={c.nDecode}
          min={1}
          max={3}
          onChange={(v) => set("nDecode", v)}
        />
        <Segmented
          label="KV link"
          value={c.link}
          options={SIM_LINKS.map((k) => ({ value: k, label: k }))}
          onChange={(v) => set("link", v)}
        />
        <Segmented
          label="Hand-off compression"
          value={
            c.compression === "none"
              ? "none"
              : `${c.compression}|${c.compressAt}`
          }
          options={[
            { value: "none", label: "none" },
            { value: "fp8|transit", label: "fp8 in transit" },
            { value: "fp8|endpoint", label: "fp8 at the GPU" },
            { value: "fp4-block|endpoint", label: "fp4 at the GPU" },
          ]}
          onChange={(v) => {
            const [comp, at] = v.split("|") as [
              Controls["compression"],
              Controls["compressAt"]?,
            ];
            setC((old) => ({
              ...old,
              compression: comp,
              compressAt: at ?? "transit",
            }));
            setPresetId("");
          }}
        />
        <Slider
          label="TTFT SLO"
          value={Math.round(c.ttftSlo * 1e3)}
          min={250}
          max={3000}
          step={250}
          onChange={(v) => set("ttftSlo", v / 1e3)}
          format={(v) => `${v} ms`}
        />
        <Slider
          label="TPOT SLO"
          value={Math.round(c.tpotSlo * 1e3)}
          min={10}
          max={100}
          step={5}
          onChange={(v) => set("tpotSlo", v / 1e3)}
          format={(v) => `${v} ms`}
        />
      </div>

      {state.status !== "ready" ? (
        <RunStatus state={state} />
      ) : (
        <ResultsTable
          runs={state.runs}
          colocated={`${c.nPrefill + c.nDecode} × ${n} ${DECODE_DEVICES[colocatedDevice(c)]}`}
          disagg={`${c.nPrefill}P × ${n} ${PREFILL_DEVICES[c.prefillDevice]} + ${c.nDecode}D × ${n} ${DECODE_DEVICES[c.decodeDevice]}`}
          preset={preset}
        />
      )}
    </WidgetFrame>
  );
}

function ResultsTable({
  runs,
  colocated,
  disagg,
  preset,
}: {
  runs: Record<"colocated" | "disagg", Run>;
  colocated: string;
  disagg: string;
  preset: (typeof PRESETS)[number] | undefined;
}): JSX.Element {
  const expected = new Map(
    (preset?.expected ?? []).map((e) => [`${e.column}.${e.key}`, e]),
  );
  const cell = (col: "colocated" | "disagg", r: Row) => {
    const m = runs[col].metrics;
    if (r.disaggOnly && col === "colocated")
      return (
        <td className="py-1 pr-2 text-neutral-600 dark:text-neutral-400">
          n/a
        </td>
      );
    const e = expected.get(`${col}.${r.key}`);
    const text = e ? FORMATS[e.fmt](m[e.key] as number) : r.show(m);
    return (
      <td className="py-1 pr-2" data-cell={`${col}.${r.key}`}>
        {text}
        {e && (
          <span
            className={`ml-1 ${text === e.text ? "text-emerald-700 dark:text-emerald-400" : "text-rose-700 dark:text-rose-400"}`}
            title={`results.md: ${e.text}`}
          >
            {text === e.text ? "✓" : `≠ ${e.text}`}
          </span>
        )}
      </td>
    );
  };
  const allMatch =
    preset !== undefined &&
    preset.expected.every(
      (e) => FORMATS[e.fmt](runs[e.column].metrics[e.key] as number) === e.text,
    );

  return (
    <>
      <div
        className="focus-ring mt-4 overflow-x-auto rounded"
        role="region"
        tabIndex={0}
        aria-label="Simulation results"
      >
        <table className="w-full text-left text-xs">
          <caption className="sr-only">
            Colocated and disaggregated results
          </caption>
          <thead>
            <tr className="text-neutral-600 dark:text-neutral-400">
              <th className="py-1 pr-2 font-normal">Metric</th>
              <th className="py-1 pr-2 font-normal">Colocated ({colocated})</th>
              <th className="py-1 pr-2 font-normal">
                Disaggregated ({disagg})
              </th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {ROWS.map((r) => (
              <tr
                key={r.key}
                className="border-t border-neutral-200 dark:border-neutral-800"
              >
                <th
                  scope="row"
                  className="py-1 pr-2 text-left font-sans font-normal"
                >
                  {r.label}
                </th>
                {cell("colocated", r)}
                {cell("disagg", r)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <StageBar run={runs.disagg} />
      {preset && (
        <p
          role="status"
          className="mt-3 break-words text-xs text-neutral-700 dark:text-neutral-300"
        >
          {allMatch ? "✓ Matches" : "≠ Differs from"} results.md §
          {preset.section}: <code className="break-all">{preset.line}</code>
        </p>
      )}
    </>
  );
}

const STAGES = [
  ["prefill_queue", "prefill queue", "bg-sky-300"],
  ["prefill", "prefill", "bg-indigo-500"],
  ["kv_wait", "KV wait", "bg-amber-400"],
  ["kv_transfer", "KV transfer", "bg-amber-600"],
  ["decode_queue", "decode queue", "bg-rose-400"],
  ["decode", "decode", "bg-emerald-500"],
] as const;

/** Where a disaggregated request's time goes (mean share of end-to-end). */
function StageBar({ run }: { run: Run }): JSX.Element {
  const shares = STAGES.map(([k]) => {
    let sum = 0;
    let e2e = 0;
    const done = run.result.reqs.filter((r) => r.finish !== null);
    for (const r of done.slice(Math.floor(done.length * 0.1))) {
      const s = {
        prefill_queue: r.prefillStart! - r.arrival,
        prefill: r.firstToken! - r.prefillStart!,
        kv_wait: r.kvStart !== null ? r.kvStart - r.firstToken! : 0,
        kv_transfer: r.kvStart !== null ? r.kvReady! - r.kvStart : 0,
        decode_queue: r.decodeStart! - (r.kvReady ?? r.firstToken!),
        decode: r.finish! - r.decodeStart!,
      };
      sum += s[k];
      e2e += r.finish! - r.arrival;
    }
    return e2e > 0 ? sum / e2e : 0;
  });
  return (
    <div className="mt-3">
      <p className="text-xs text-neutral-600 dark:text-neutral-400">
        Where a disaggregated request&apos;s time goes
      </p>
      <div className="mt-1 flex h-3 w-full overflow-hidden rounded" aria-hidden>
        {STAGES.map(([k, , cls], i) => (
          <div
            key={k}
            className={cls}
            style={{ width: `${100 * shares[i]!}%` }}
          />
        ))}
      </div>
      <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[0.7rem] text-neutral-700 dark:text-neutral-300">
        {STAGES.map(([k, label, cls], i) => (
          <li key={k} className="flex items-center gap-1">
            <span className={`inline-block size-2 rounded-sm ${cls}`} />
            {label} {(100 * shares[i]!).toFixed(1)}%
          </li>
        ))}
      </ul>
    </div>
  );
}
