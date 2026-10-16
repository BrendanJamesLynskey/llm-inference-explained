"use client";

/**
 * Chapter 2's mechanism: the KV memory of one instance over time. OPT-13B
 * on one A100-40GB (vLLM's setting, where memory binds), eighteen requests
 * arriving fast. Each cell is the KV of a fixed number of tokens, coloured
 * by the request that holds it: solid where it holds tokens, a lighter cell
 * for a block only partly filled, hatched where memory is reserved but
 * still empty. Reserved KV admits a request only when prompt + output fit;
 * paged KV hands out blocks as tokens arrive and, when none is left,
 * preempts the latest arrival (recompute it later, or swap it to host).
 */
import { useMemo, useState } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import {
  eventsUpTo,
  fmtSeconds,
  frameTime,
  kvCaption,
  kvView,
  variantOf,
} from "@/lib/tradeoffs/mech";

import { Hatch, Loading, Provenance, reqColour, useMech } from "./common";

const FRAMES = 90;

export default function KvWidget({
  equation,
}: {
  equation?: React.ReactNode;
}): JSX.Element {
  const f = useMech("paged");
  const [key, setKey] = useState("oracle");
  const [ref, width] = useWidth(640);
  const stepper = useStepper(FRAMES, { stepMs: 260, resetKey: key });
  const v = f ? variantOf(f, key) : null;
  const t = v ? frameTime(v, stepper.step, FRAMES) : 0;
  const k = useMemo(() => (v ? kvView(v, t) : null), [v, t]);
  if (!f || !v || !k) return <Loading what="KV memory animation" />;
  const prevT = stepper.step > 0 ? frameTime(v, stepper.step - 1, FRAMES) : -1;
  const fresh = eventsUpTo(v, t).filter(
    (e) => e[0] > prevT && e[1].startsWith("preempt"),
  );
  const W = width;
  const cell = W < 560 ? 14 : 16;
  const gap = 2;
  const cols = Math.max(8, Math.floor((W - 8) / (cell + gap)));
  const rows = Math.ceil(k.cells.length / cols);
  const gridH = rows * (cell + gap);
  const top = 22;
  const H = top + gridH + 64;
  const queue = k.step?.queue ?? [];
  const swapped = k.step?.swapped ?? [];
  return (
    <AnimationPanel
      title="KV memory, cell by cell"
      summary={
        <>
          OPT-13B on one A100-40GB, eighteen requests arriving fast: memory
          binds. Each cell is {k.cellTokens} tokens of KV, coloured by the
          request holding it: solid when it holds tokens, light when partly
          filled, hatched when reserved but still empty. Pick reserved or paged
          KV.
        </>
      }
      stepper={stepper}
      stepLabel="frame"
      countFrom={0}
      caption={kvCaption(k)}
      testId="mech-kv"
      equation={equation}
      hl={fresh.length ? "c" : key === "oracle" ? "w" : "p"}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="The instance's KV memory as cells, coloured by the request holding them"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <defs>
              {Array.from(new Set(k.cells.map((c) => c?.rid ?? -1)))
                .filter((r) => r >= 0)
                .map((r) => (
                  <Hatch key={r} id={`kv-h-${r}`} colour={reqColour(r)} />
                ))}
              <Hatch id="kv-wait" />
            </defs>
            <text x={4} y={14} fontSize={12} fill="currentColor">
              KV memory: {(100 * (k.allocTokens / k.capTokens)).toFixed(0)}%
              allocated, {(100 * (k.heldTokens / k.capTokens)).toFixed(0)}%
              holding tokens
            </text>
            {k.cells.map((c, i) => {
              const x = 4 + (i % cols) * (cell + gap);
              const y = top + Math.floor(i / cols) * (cell + gap);
              if (!c)
                return (
                  <rect
                    key={i}
                    x={x}
                    y={y}
                    width={cell}
                    height={cell}
                    fill="none"
                    stroke="currentColor"
                    strokeOpacity={0.18}
                  />
                );
              return (
                <rect
                  key={i}
                  x={x}
                  y={y}
                  width={cell}
                  height={cell}
                  fill={
                    c.state === "reserved"
                      ? `url(#kv-h-${c.rid})`
                      : reqColour(c.rid)
                  }
                  opacity={c.state === "partial" ? 0.55 : 1}
                  stroke={reqColour(c.rid)}
                  strokeWidth={1}
                >
                  <title>{`request ${c.rid}: ${c.state}`}</title>
                </rect>
              );
            })}
            <text x={4} y={top + gridH + 18} fontSize={12} fill="currentColor">
              Waiting ({queue.length}):
            </text>
            {queue.slice(0, Math.floor((W - 110) / 16)).map((rid, i) => (
              <rect
                key={rid}
                x={100 + i * 16}
                y={top + gridH + 7}
                width={12}
                height={12}
                fill="url(#kv-wait)"
                stroke={reqColour(rid)}
                strokeWidth={2}
              >
                <title>{`request ${rid} waiting`}</title>
              </rect>
            ))}
            <text x={4} y={top + gridH + 42} fontSize={12} fill="currentColor">
              Swapped to host ({swapped.length}){fresh.length ? " · " : ""}
              {fresh.length
                ? `just preempted: ${fresh.map((e) => `r${e[2]}`).join(", ")}`
                : ""}
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="TTFT p99" value={fmtSeconds(v.summary.ttft_p99)} />
          <Stat label="TPOT p99" value={fmtSeconds(v.summary.tpot_p99)} />
          <Stat label="Preemptions" value={String(v.summary.preemptions)} />
          <Stat label="All done at" value={fmtSeconds(v.horizon)} />
        </div>
      }
      params={
        <div className="sm:col-span-2">
          <Segmented
            label="KV memory"
            value={key}
            options={f.variants.map((x) => ({ value: x.key, label: x.label }))}
            onChange={setKey}
          />
          <Provenance f={f} />
        </div>
      }
    />
  );
}
