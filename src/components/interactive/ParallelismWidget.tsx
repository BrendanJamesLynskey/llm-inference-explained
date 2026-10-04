"use client";

/**
 * Communication per decode step for tensor, pipeline and expert
 * parallelism, on the simulator's links, next to the compute step it adds to.
 */
import { useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  H100_SXM,
  LINKS,
  LLAMA3_70B,
  costModel,
} from "@/lib/inference/costModel";
import {
  bubbleFraction,
  epComm,
  ppComm,
  tpComm,
  type Comm,
} from "@/lib/inference/parallel";

type Strategy = "tp" | "pp" | "ep";
const LINK_KEYS = ["nvlink4", "pcie5", "ib-ndr", "eth-100g"] as const;

const us = (s: number): string =>
  s >= 1e-3 ? `${(s * 1e3).toFixed(2)} ms` : `${(s * 1e6).toFixed(0)} µs`;

export function ParallelismWidget(): JSX.Element {
  const [strategy, setStrategy] = useState<Strategy>("tp");
  const [n, setN] = useState(4);
  const [linkKey, setLinkKey] = useState<(typeof LINK_KEYS)[number]>("nvlink4");
  const [batch, setBatch] = useState(16);
  const [micro, setMicro] = useState(4);
  const link = LINKS[linkKey]!;
  const m = LLAMA3_70B;
  // The roofline step on n devices (the cost model's TP-as-one-big-device).
  const step = costModel(m, H100_SXM, { nDevices: n }).decode(
    2048 * batch,
    batch,
  );
  let comm: Comm;
  let note: string;
  if (strategy === "tp") {
    comm = tpComm(m, n, link, batch);
    note =
      "Two all-reduces per layer (after attention and after the MLP), 160 per step for 80 layers.";
  } else if (strategy === "pp") {
    comm = ppComm(m, n, link, batch);
    note = `One activation per stage boundary. With ${micro} micro-batch${micro > 1 ? "es" : ""} in flight, stages idle ${(bubbleFraction(n, micro) * 100).toFixed(0)}% of the time (the bubble).`;
  } else {
    // A hypothetical MoE with Llama-3-70B's width: top-2 routing in every layer.
    comm = epComm(m.d_model, m.n_layers, 2, n, link, batch);
    note =
      "Hypothetical: an MoE with Llama-3-70B's width, top-2 routing in all 80 layers, experts spread evenly. Each token goes to its 2 experts and back.";
  }
  const share = comm.time / (comm.time + step.time);

  return (
    <WidgetFrame
      testId="parallelism"
      title="What splitting a model across devices sends"
      caption="Llama-3-70B decode, context 2,048 per sequence, BF16 activations. Link bandwidths and latencies from the simulator's hardware.py. First-order estimates: no overlap with compute; real collective libraries use faster algorithms for small messages."
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Segmented
          label="Strategy"
          value={strategy}
          options={[
            { value: "tp", label: "tensor" },
            { value: "pp", label: "pipeline" },
            { value: "ep", label: "expert" },
          ]}
          onChange={setStrategy}
        />
        <Segmented
          label="Link"
          value={linkKey}
          options={LINK_KEYS.map((k) => ({ value: k, label: k }))}
          onChange={setLinkKey}
        />
        <Slider label="Devices" value={n} min={2} max={8} onChange={setN} />
        <Slider
          label="Batch (sequences)"
          value={batch}
          min={1}
          max={128}
          onChange={setBatch}
        />
        {strategy === "pp" && (
          <Slider
            label="Micro-batches in flight"
            value={micro}
            min={1}
            max={32}
            onChange={setMicro}
          />
        )}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Sent per device / step"
          value={`${(comm.bytes / 1e6).toFixed(2)} MB`}
        />
        <Stat label="Messages / step" value={String(comm.messages)} />
        <Stat label="Comm time / step" value={us(comm.time)} hint={link.name} />
        <Stat
          label="Compute step"
          value={us(step.time)}
          hint={`${n}× H100, ${step.bound}-bound`}
        />
      </div>
      <div
        className="mt-3 h-3 w-full overflow-hidden rounded bg-neutral-200 dark:bg-neutral-800"
        aria-label={`Communication is ${(share * 100).toFixed(0)}% of the step`}
      >
        <div
          className="h-full bg-rose-500"
          style={{ width: `${share * 100}%` }}
        />
      </div>
      <p className="mt-1 text-sm" data-testid="comm-share">
        Communication is <strong>{(share * 100).toFixed(0)}%</strong> of a
        decode step if nothing overlaps. {note}
      </p>
    </WidgetFrame>
  );
}
