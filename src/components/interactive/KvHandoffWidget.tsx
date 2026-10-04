"use client";

/**
 * The KV hand-off in closed form: how many bytes a prompt's cache is, how
 * long each link takes to carry it, what compression saves, and how much of
 * the transfer layer-wise streaming hides behind the prefill.
 */
import { scaleLinear } from "d3";
import { useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { handoff, type Compression } from "@/lib/disagg/handoff";
import { LINKS, LLAMA3_70B, LLAMA3_8B } from "@/lib/inference/costModel";

const LINK_KEYS = ["nvlink4", "ib-ndr", "cpo-optical", "eth-100g", "eth-25g"];
const PROMPTS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const W = 560;

const time = (s: number): string =>
  s >= 1 ? `${s.toFixed(2)} s` : `${(s * 1e3).toFixed(1)} ms`;
const mb = (b: number): string =>
  b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`;

export function KvHandoffWidget(): JSX.Element {
  const [model, setModel] = useState<"8b" | "70b">("8b");
  const [p, setP] = useState(3);
  const [link, setLink] = useState("eth-25g");
  const [comp, setComp] = useState<Compression>("none");
  const [mode, setMode] = useState<"after" | "layerwise">("after");

  const m = model === "8b" ? LLAMA3_8B : LLAMA3_70B;
  const devices = model === "8b" ? 1 : 4;
  const prompt = PROMPTS[p]!;
  const h = handoff({
    model: m,
    devices,
    prompt,
    link,
    compression: comp,
    layerwise: mode === "layerwise",
  });

  const end = h.prefillTime + h.exposedTime;
  const x = scaleLinear()
    .domain([0, Math.max(end, h.prefillTime + h.transferTime)])
    .range([70, W - 8]);
  const xferStart = mode === "layerwise" ? end - h.transferTime : h.prefillTime;

  return (
    <WidgetFrame
      testId="kv-handoff"
      title="Handing the KV cache from prefill to decode"
      caption={`${m.name} (${devices}× H100 per prefill instance), BF16 KV cache. Bytes per token, link bandwidths and latencies, and the prefill step are Disaggregated_Inference_Sim's (hardware.py); layer-wise streaming is this site's extension.`}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Segmented
          label="Model"
          value={model}
          options={[
            { value: "8b", label: "Llama-3-8B" },
            { value: "70b", label: "Llama-3-70B" },
          ]}
          onChange={setModel}
        />
        <Slider
          label="Prompt tokens"
          value={p}
          min={0}
          max={PROMPTS.length - 1}
          onChange={setP}
          format={(i) => (PROMPTS[i] ?? 0).toLocaleString("en-GB")}
        />
        <Segmented
          label="Link"
          value={link}
          options={LINK_KEYS.map((k) => ({ value: k, label: k }))}
          onChange={setLink}
        />
        <Segmented
          label="Compression"
          value={comp}
          options={[
            { value: "none", label: "none (BF16)" },
            { value: "fp8", label: "fp8" },
            { value: "fp4-block", label: "fp4-block" },
          ]}
          onChange={setComp}
        />
        <Segmented
          label="Send"
          value={mode}
          options={[
            { value: "after", label: "after prefill" },
            { value: "layerwise", label: "layer by layer" },
          ]}
          onChange={setMode}
        />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-3">
        <Stat
          label="KV to hand off"
          value={mb(h.bytes)}
          hint={`${prompt.toLocaleString("en-GB")} tokens`}
        />
        <Stat
          label="On the wire"
          value={mb(h.wireBytes)}
          hint={comp === "none" ? "uncompressed" : comp}
        />
        <Stat
          label="Transfer"
          value={time(h.transferTime)}
          hint={`${LINKS[link]!.name}, ${(LINKS[link]!.bandwidth / 1e9).toFixed(1)} GB/s`}
        />
        <Stat label="Prefill" value={time(h.prefillTime)} />
        <Stat
          label="Exposed after prefill"
          value={time(h.exposedTime)}
          hint="added before decode can start"
        />
        <Stat
          label="Link full at"
          value={`${h.linkSaturationRate.toFixed(1)} req/s`}
          hint="hand-offs alone"
        />
      </div>

      <svg
        viewBox={`0 0 ${W} 78`}
        className="mt-4 h-auto w-full"
        role="img"
        aria-label={`Timeline: prefill ${time(h.prefillTime)}, transfer ${time(h.transferTime)}, ${time(h.exposedTime)} exposed after the prefill`}
      >
        <text
          x={0}
          y={22}
          className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
        >
          prefill
        </text>
        <rect
          x={x(0)}
          y={10}
          width={Math.max(1, x(h.prefillTime) - x(0))}
          height={16}
          className="fill-indigo-500"
        />
        <text
          x={0}
          y={50}
          className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
        >
          KV transfer
        </text>
        <rect
          x={x(xferStart)}
          y={38}
          width={Math.max(1, x(xferStart + h.transferTime) - x(xferStart))}
          height={16}
          className="fill-amber-500"
        />
        <line
          x1={x(h.prefillTime)}
          x2={x(h.prefillTime)}
          y1={4}
          y2={60}
          strokeDasharray="3 3"
          className="stroke-neutral-500"
        />
        <text
          x={W / 2}
          y={74}
          textAnchor="middle"
          className="fill-neutral-600 text-[10px] dark:fill-neutral-400"
        >
          0 – {time(Math.max(end, h.prefillTime + h.transferTime))}; dashed
          line: first token
        </text>
      </svg>
    </WidgetFrame>
  );
}
