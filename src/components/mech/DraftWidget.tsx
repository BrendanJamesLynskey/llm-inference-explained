"use client";

/**
 * Chapter 10's mechanism: draft tokens accepted or rejected, verify pass by
 * verify pass. Two requests decode on one H100 (Llama-3-8B with an MTP
 * draft head). Each pass drafts gamma tokens per row, the target checks
 * them all in one pass, and a row keeps the accepted run plus one token of
 * the target's own; the acceptance draws are the simulator's (per request,
 * seeded), so the frames are its records, not a re-enactment.
 */
import { useMemo, useState } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import { engine } from "@/lib/tradeoffs/engine";
import {
  fmtSeconds,
  gammaOf,
  instSteps,
  specCaption,
  specView,
  variantOf,
} from "@/lib/tradeoffs/mech";
import { OKABE_ITO, PHASE_COLOUR } from "@/lib/viz/palette";

import { Hatch, Loading, Provenance, reqColour, useMech } from "./common";

export default function DraftWidget({
  equation,
}: {
  equation?: React.ReactNode;
}): JSX.Element {
  const f = useMech("speculative");
  const [key, setKey] = useState("g3a07");
  const [ref, width] = useWidth(640);
  const v = f ? variantOf(f, key) : null;
  const steps = useMemo(() => (v ? instSteps(v, 0) : []), [v]);
  const stepper = useStepper(Math.max(1, steps.length), {
    stepMs: 900,
    resetKey: key,
  });
  if (!f || !v) return <Loading what="speculative decoding animation" />;
  const s = specView(v, Math.min(stepper.step, steps.length - 1));
  const g = gammaOf(v);
  const spec = v.cfg.speculative as
    | { alpha: number; gamma: number }
    | undefined;
  const W = width;
  const cell = W < 560 ? 22 : 28;
  const gap = 4;
  const rowY = (i: number) => 26 + i * (cell + 18);
  const nRows = v.reqs.length;
  const progY = rowY(nRows) + 6;
  const barW = W - 70;
  const T = v.horizon;
  const tlY = progY + nRows * 18 + 22;
  const H = tlY + 40;
  const outputs = v.reqs.map((r) => Number(r[8]));
  return (
    <AnimationPanel
      title="Drafts accepted and rejected, pass by pass"
      summary={
        <>
          Two requests decode on one H100 (Llama-3-8B, an MTP draft head). Each
          verify pass checks gamma drafted tokens per row; a row keeps the
          accepted run (green) plus one token of the target&apos;s own (blue).
          The first rejected draft is crossed out and the rest are discarded
          (grey). The draws are the simulator&apos;s.
        </>
      }
      stepper={stepper}
      stepLabel="pass"
      caption={specCaption(v, s)}
      testId="mech-spec"
      equation={equation}
      hl={s.rows.some((r) => r.rejected > 0) ? "a" : "n"}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Each row's drafted tokens in this verify pass, accepted or rejected, and each request's progress"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <defs>
              <Hatch id="sp-rej" colour={OKABE_ITO.vermillion} />
            </defs>
            <text x={4} y={14} fontSize={12} fill="currentColor">
              {s.kind === "prefill"
                ? "A prefill pass (no drafts)"
                : g
                  ? `This pass: ${g} drafts per row, then the target's token`
                  : "No speculation: one token per row per pass"}
            </text>
            {v.reqs.map((_, rid) => {
              const r = s.rows.find((x) => x.rid === rid);
              const y = rowY(rid);
              const cells: JSX.Element[] = [];
              if (r) {
                const slots = g || 0;
                for (let j = 0; j < slots; j++) {
                  const state =
                    j < r.accepted ? "ok" : j === r.accepted ? "rej" : "skip";
                  cells.push(
                    <g key={j}>
                      <rect
                        x={40 + j * (cell + gap)}
                        y={y}
                        width={cell}
                        height={cell}
                        fill={
                          state === "ok"
                            ? OKABE_ITO.green
                            : state === "rej"
                              ? "url(#sp-rej)"
                              : "none"
                        }
                        stroke={
                          state === "rej"
                            ? OKABE_ITO.vermillion
                            : "currentColor"
                        }
                        strokeOpacity={state === "skip" ? 0.3 : 1}
                      />
                      <text
                        x={40 + j * (cell + gap) + cell / 2}
                        y={y + cell / 2 + 4}
                        fontSize={12}
                        textAnchor="middle"
                        fill={state === "skip" ? "currentColor" : "#000"}
                        opacity={state === "skip" ? 0.4 : 1}
                      >
                        {state === "ok" ? "✓" : state === "rej" ? "✗" : "·"}
                      </text>
                    </g>,
                  );
                }
                cells.push(
                  <g key="t">
                    <rect
                      x={40 + slots * (cell + gap) + 6}
                      y={y}
                      width={cell + 8}
                      height={cell}
                      fill={PHASE_COLOUR.decode}
                    />
                    <text
                      x={40 + slots * (cell + gap) + 10 + cell / 2}
                      y={y + cell / 2 + 4}
                      fontSize={12}
                      textAnchor="middle"
                      fill="#fff"
                    >
                      +1
                    </text>
                  </g>,
                );
              }
              return (
                <g key={rid}>
                  <text
                    x={4}
                    y={y + cell / 2 + 4}
                    fontSize={11}
                    fill="currentColor"
                  >
                    r{rid}
                  </text>
                  {r ? (
                    cells
                  ) : (
                    <text
                      x={40}
                      y={y + cell / 2 + 4}
                      fontSize={11}
                      fill="currentColor"
                      opacity={0.6}
                    >
                      {s.kind === "prefill"
                        ? "(prefill)"
                        : "(not in this pass)"}
                    </text>
                  )}
                  {r && (
                    <text
                      x={40 + (g || 0) * (cell + gap) + cell + 24}
                      y={y + cell / 2 + 4}
                      fontSize={11}
                      fill="currentColor"
                    >
                      = {r.kept}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={4} y={progY + 4} fontSize={11} fill="currentColor">
              Tokens out so far
            </text>
            {v.reqs.map((_, rid) => (
              <g key={rid}>
                <rect
                  x={60}
                  y={progY + 10 + rid * 18}
                  width={barW}
                  height={12}
                  fill="none"
                  stroke="currentColor"
                  strokeOpacity={0.2}
                />
                <rect
                  x={60}
                  y={progY + 10 + rid * 18}
                  width={
                    (barW * Math.min(s.out[rid]!, outputs[rid]!)) /
                    outputs[rid]!
                  }
                  height={12}
                  fill={reqColour(rid)}
                />
                <text
                  x={4}
                  y={progY + 20 + rid * 18}
                  fontSize={11}
                  fill="currentColor"
                >
                  r{rid} {s.out[rid]}
                </text>
              </g>
            ))}
            <text x={4} y={tlY - 4} fontSize={11} fill="currentColor">
              Passes over time ({fmtSeconds(T)}): draft part hatched
            </text>
            {steps.map((st, i) => {
              const x = 4 + (st.t0 / T) * (W - 8);
              const w = Math.max(1, (st.dt / T) * (W - 8) - 0.5);
              const dw = (st.draft / st.dt) * w;
              return (
                <g key={i} opacity={i > s.k ? 0.2 : 1}>
                  <rect
                    x={x}
                    y={tlY}
                    width={w}
                    height={16}
                    fill={i === s.k ? OKABE_ITO.blue : "#a3a3a3"}
                  />
                  {dw > 0 && (
                    <rect
                      x={x}
                      y={tlY}
                      width={dw}
                      height={16}
                      fill="url(#sp-rej)"
                      opacity={0.5}
                    />
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="Tokens per verify"
            value={g ? v.summary.tokens_per_verify.toFixed(2) : "1"}
          />
          <Stat
            label="Equation (1)"
            value={
              spec
                ? engine.expectedTokens(spec.alpha, spec.gamma).toFixed(2)
                : "1"
            }
          />
          <Stat label="TPOT p99" value={fmtSeconds(v.summary.tpot_p99)} />
          <Stat label="All done at" value={fmtSeconds(v.horizon)} />
        </div>
      }
      params={
        <div className="sm:col-span-2">
          <Segmented
            label="Drafts per pass and acceptance rate"
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
