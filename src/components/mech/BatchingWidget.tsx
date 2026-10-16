"use client";

/**
 * Chapter 1's mechanism: which rows share each forward pass under each
 * batching policy. Four requests are decoding when two long prompts arrive
 * (Mistral-7B on one A100); every step the simulator ran is a frame: the
 * pass itself (decode rows and prompt chunks, against the token budget),
 * where it sits on the time axis, and every request's tokens so far, so a
 * stalled decode row shows as a gap.
 */
import { useMemo, useState } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import {
  batchCaption,
  batchHl,
  batchView,
  fmtSeconds,
  hasPrompt,
  instSteps,
  parseLabel,
  variantOf,
} from "@/lib/tradeoffs/mech";
import { OKABE_ITO } from "@/lib/viz/palette";

import { Hatch, Loading, Provenance, reqColour, useMech } from "./common";

export default function BatchingWidget({
  equation,
}: {
  equation?: React.ReactNode;
}): JSX.Element {
  const f = useMech("batching");
  const [key, setKey] = useState("prefill-priority");
  const [ref, width] = useWidth(640);
  const v = f ? variantOf(f, key) : null;
  const steps = useMemo(() => (v ? instSteps(v, 0) : []), [v]);
  const stepper = useStepper(Math.max(1, steps.length), {
    stepMs: 650,
    resetKey: key,
  });
  if (!f || !v) return <Loading what="batching animation" />;
  const b = batchView(v, Math.min(stepper.step, steps.length - 1));
  const chunked = v.cfg.batchPolicy === "chunked";
  const maxTok = Math.max(
    ...steps.map((s) => parseLabel(s.label).tok ?? 0),
    chunked ? b.budget : 0,
  );
  const W = width;
  const pad = 8;
  const inner = W - 2 * pad;
  // the pass: decode rows get a minimum width so one token stays visible
  const minW = 6;
  const nDec = b.decodeRows;
  const decW = nDec * minW;
  const scale = (inner - (chunked ? 0 : decW)) / Math.max(1, maxTok);
  let x = pad;
  const passY = 26;
  const passH = 30;
  const budgetX = pad + b.budget * scale + (chunked ? 0 : decW);
  // the time axis
  const T = v.horizon;
  const tx = (t: number) => pad + (t / T) * inner;
  const axisY = 96;
  const lanesY = axisY + 34;
  const laneH = 16;
  const nReq = v.reqs.length;
  const H = lanesY + nReq * laneH + (W >= 560 ? 24 : 36);
  // every request's token times up to this step
  const ticks: { rid: number; t: number; first: boolean }[] = [];
  for (let i = 0; i <= b.k; i++) {
    const s = steps[i]!;
    for (const [rid, p, d] of s.work)
      if (d > 0) ticks.push({ rid, t: s.t0 + s.dt, first: p > 0 });
  }
  return (
    <AnimationPanel
      title="Who shares each forward pass"
      summary={
        <>
          Four requests are decoding when two long prompts arrive (Mistral-7B,
          one A100). Each frame is one step the simulator ran: the pass (decode
          rows, one token each, and prompt tokens), the step on the time axis,
          and every request&apos;s tokens so far. Pick a policy.
        </>
      }
      stepper={stepper}
      stepLabel="step"
      caption={batchCaption(b)}
      testId="mech-batching"
      equation={equation}
      hl={batchHl(b)}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="The current forward pass, the step timeline and each request's tokens"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <defs>
              <Hatch id="bt-hatch" />
            </defs>
            <text x={pad} y={16} fontSize={12} fill="currentColor">
              This pass: {b.decodeRows} decode, {b.prefillTokens} prompt tokens
            </text>
            <rect
              x={pad}
              y={passY}
              width={inner}
              height={passH}
              fill="none"
              stroke="currentColor"
              strokeOpacity={0.25}
            />
            {b.slices.map((s, i) => {
              const w =
                s.kind === "decode" ? minW : Math.max(2, s.tokens * scale);
              const r = (
                <rect
                  key={i}
                  x={x}
                  y={passY}
                  width={Math.max(1, w - 1)}
                  height={passH}
                  fill={s.rid < 0 ? OKABE_ITO.orange : reqColour(s.rid)}
                  stroke={s.kind === "prefill" ? "currentColor" : "none"}
                  strokeWidth={s.kind === "prefill" ? 1 : 0}
                  opacity={s.kind === "prefill" ? 0.75 : 1}
                >
                  <title>
                    {s.kind === "decode"
                      ? `request ${s.rid}: one decode token`
                      : `request ${s.rid}: ${s.tokens} prompt tokens`}
                  </title>
                </rect>
              );
              x += w;
              return r;
            })}
            {chunked && (
              <>
                <line
                  x1={budgetX}
                  x2={budgetX}
                  y1={passY - 4}
                  y2={passY + passH + 4}
                  stroke={OKABE_ITO.vermillion}
                  strokeWidth={2}
                />
                <text
                  x={Math.min(budgetX, W - pad)}
                  y={passY + passH + 16}
                  fontSize={11}
                  textAnchor="end"
                  fill="currentColor"
                >
                  budget τ = {b.budget}
                </text>
              </>
            )}
            {/* the steps on the time axis */}
            <text x={pad} y={axisY - 8} fontSize={12} fill="currentColor">
              Steps over time ({fmtSeconds(T)})
            </text>
            {steps.map((s, i) => (
              <rect
                key={i}
                x={tx(s.t0)}
                y={axisY}
                width={Math.max(1, tx(s.t0 + s.dt) - tx(s.t0) - 0.5)}
                height={14}
                fill={
                  i === b.k
                    ? OKABE_ITO.blue
                    : hasPrompt(s)
                      ? OKABE_ITO.orange
                      : "#a3a3a3"
                }
                opacity={i > b.k ? 0.25 : 0.9}
              />
            ))}
            {/* each request's tokens */}
            {v.reqs.map((_, rid) => (
              <g key={rid}>
                <text
                  x={pad}
                  y={lanesY + rid * laneH + 11}
                  fontSize={11}
                  fill="currentColor"
                >
                  r{rid}
                </text>
                <line
                  x1={pad + 22}
                  x2={W - pad}
                  y1={lanesY + rid * laneH + 7}
                  y2={lanesY + rid * laneH + 7}
                  stroke="currentColor"
                  strokeOpacity={0.12}
                />
              </g>
            ))}
            {ticks.map((k, i) => (
              <rect
                key={i}
                x={Math.max(pad + 22, tx(k.t) - 1)}
                y={lanesY + k.rid * laneH + 2}
                width={k.first ? 4 : 2}
                height={10}
                fill={reqColour(k.rid)}
              />
            ))}
            {W >= 560 ? (
              <text x={pad} y={H - 6} fontSize={11} fill="currentColor">
                grey: decode-only steps · orange: steps with prompt tokens ·
                ticks: tokens out
              </text>
            ) : (
              <>
                <text x={pad} y={H - 18} fontSize={11} fill="currentColor">
                  grey: decode-only steps · orange: with prompt tokens
                </text>
                <text x={pad} y={H - 4} fontSize={11} fill="currentColor">
                  ticks: tokens out (thick: first token)
                </text>
              </>
            )}
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="TTFT p99" value={fmtSeconds(v.summary.ttft_p99)} />
          <Stat label="ITL p99" value={fmtSeconds(v.summary.itl_p99)} />
          <Stat label="TPOT p99" value={fmtSeconds(v.summary.tpot_p99)} />
          <Stat label="All done at" value={fmtSeconds(v.horizon)} />
        </div>
      }
      params={
        <div className="sm:col-span-2">
          <Segmented
            label="Batching policy"
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
