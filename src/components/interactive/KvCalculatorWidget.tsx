"use client";

/**
 * KV-cache size calculator: model preset × attention variant × precision ×
 * context × batch, against an 80 GB GPU's memory.
 */
import { useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  KV_PRESETS,
  kvBytes,
  kvValuesPerToken,
  type AttentionKind,
} from "@/lib/inference/kvMemory";

const KINDS: { value: AttentionKind; label: string }[] = [
  { value: "mha", label: "MHA" },
  { value: "gqa", label: "GQA" },
  { value: "mqa", label: "MQA" },
  { value: "mla", label: "MLA" },
];

/** Per-token sizes in binary units (128 KiB), as the chapter text quotes them. */
const fmtKiB = (b: number): string => `${(b / 1024).toFixed(1)} KiB`;

const fmtBytes = (b: number): string =>
  b >= 1e9
    ? `${(b / 1e9).toFixed(2)} GB`
    : b >= 1e6
      ? `${(b / 1e6).toFixed(1)} MB`
      : `${(b / 1e3).toFixed(1)} kB`;

export function KvCalculatorWidget(): JSX.Element {
  const [preset, setPreset] = useState<keyof typeof KV_PRESETS>("llama3-8b");
  const [kind, setKind] = useState<AttentionKind>("gqa");
  const [bytes, setBytes] = useState<"2" | "1">("2");
  const [ctxPow, setCtxPow] = useState(13); // 8192
  const [batch, setBatch] = useState(16);
  const shape = KV_PRESETS[preset]!;
  const b = Number(bytes);
  const ctx = 2 ** ctxPow;
  const perTok = kvBytes(shape, kind, b, 1);
  const total = kvBytes(shape, kind, b, ctx * batch);
  const gpus = total / 80e9;

  return (
    <WidgetFrame
      testId="kv-calculator"
      title="How big is the KV cache?"
      caption="Bytes = values per token × bytes per value × tokens. Llama shapes from Disaggregated_Inference_Sim's hardware.py; DeepSeek-V2 and the MLA sizes from its paper (arXiv:2405.04434)."
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Segmented
          label="Model"
          value={preset}
          options={[
            { value: "llama3-8b", label: "Llama-3-8B" },
            { value: "llama3-70b", label: "Llama-3-70B" },
            { value: "deepseek-v2", label: "DeepSeek-V2" },
          ]}
          onChange={setPreset}
        />
        <Segmented
          label="Attention"
          value={kind}
          options={KINDS}
          onChange={setKind}
        />
        <Segmented
          label="KV precision"
          value={bytes}
          options={[
            { value: "2", label: "BF16" },
            { value: "1", label: "FP8" },
          ]}
          onChange={setBytes}
        />
        <div className="grid gap-3">
          <Slider
            label="Context per sequence"
            value={ctxPow}
            min={9}
            max={17}
            onChange={setCtxPow}
            format={() => `${ctx.toLocaleString("en-GB")} tokens`}
          />
          <Slider
            label="Batch (sequences)"
            value={batch}
            min={1}
            max={128}
            onChange={setBatch}
          />
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat
          label="Values / token"
          value={kvValuesPerToken(shape, kind).toLocaleString("en-GB")}
        />
        <Stat label="Bytes / token" value={fmtKiB(perTok)} />
        <Stat label="Whole batch" value={fmtBytes(total)} />
        <Stat
          label="80 GB GPUs of KV"
          value={gpus.toFixed(2)}
          hint="cache alone, no weights"
        />
      </div>

      <table className="mt-4 w-full text-left text-xs">
        <caption className="sr-only">
          Bytes per token for each attention variant
        </caption>
        <thead>
          <tr className="text-neutral-500 dark:text-neutral-400">
            <th className="py-1 font-normal">Variant</th>
            <th className="py-1 font-normal">KV heads</th>
            <th className="py-1 font-normal">Bytes / token</th>
            <th className="py-1 font-normal">vs MHA</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {KINDS.map((k) => {
            const v = kvBytes(shape, k.value, b, 1);
            const mha = kvBytes(shape, "mha", b, 1);
            return (
              <tr
                key={k.value}
                className={`border-t border-neutral-200 dark:border-neutral-800 ${k.value === kind ? "font-semibold" : ""}`}
              >
                <td className="py-1">{k.label}</td>
                <td className="py-1">
                  {k.value === "mla"
                    ? `latent ${shape.mla.dC}+${shape.mla.dRope}`
                    : k.value === "mha"
                      ? shape.heads
                      : k.value === "gqa"
                        ? shape.kvHeads
                        : 1}
                </td>
                <td className="py-1">{fmtKiB(v)}</td>
                <td className="py-1">{((v / mha) * 100).toFixed(1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {preset !== "deepseek-v2" && (
        <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
          No Llama model uses MLA; its row applies DeepSeek-V2&rsquo;s latent
          sizes to this layer count, for comparison only.
        </p>
      )}
    </WidgetFrame>
  );
}
