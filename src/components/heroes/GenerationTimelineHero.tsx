"use client";

/**
 * Chapter 1's hero: one request on one H100, in time order. The prefill
 * pass reads the whole prompt at once and is compute-bound; then every
 * output token is its own short, memory-bound decode step. Each bar is a
 * pass the cost model priced (`generationTimeline`); its width is its time,
 * its colour its phase, and under it a meter shows FLOPs against bytes.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import {
  TIMELINE_DEFAULT,
  generationTimeline,
  timelineCaption,
} from "@/lib/inference/heroes";
import { PHASE_COLOUR } from "@/lib/viz/palette";

const PROMPTS = ["512", "2048", "8192"] as const;

const msText = (s: number): string => `${(s * 1e3).toFixed(1)} ms`;

export default function GenerationTimelineHero({
  equation,
}: {
  equation?: ReactNode;
}): JSX.Element {
  const [prompt, setPrompt] = useState<(typeof PROMPTS)[number]>("2048");
  const steps = useMemo(
    () => generationTimeline({ ...TIMELINE_DEFAULT, prompt: Number(prompt) }),
    [prompt],
  );
  const st = useStepper(steps.length, { stepMs: 650, resetKey: prompt });
  const [ref, width] = useWidth(640);
  const k = st.step;
  const s = steps[k]!;
  const W = width;
  const phone = W < 520;
  const end = steps[steps.length - 1]!;
  const T = end.t0 + end.dt;
  const left = 8;
  const span = W - 16;
  const x = (t: number) => left + (t / T) * span;
  const barY = 34;
  const barH = phone ? 34 : 40;
  // the current pass: its arithmetic time against its memory time
  const meterY = barY + barH + 46;
  const tMax = Math.max(s.tCompute, s.tMemory);
  const mw = (v: number) => Math.max(2, (v / tMax) * (span - 96));
  const tokensY = meterY + 62;
  const tokW = Math.min(26, (span - 4) / steps.length);
  const H = tokensY + tokW + 26;
  const hl = s.kind === "prefill" ? "p" : "bw";
  return (
    <AnimationPanel
      testId="hero-timeline"
      title="One request: a long prefill, then one short step per token"
      summary={
        <>
          Llama-3-8B on one H100 with a {Number(prompt).toLocaleString("en-GB")}
          -token prompt and {TIMELINE_DEFAULT.output} output tokens. Every bar
          is a forward pass the simulator&apos;s cost model priced: orange the
          prefill, blue each decode step, width its time.
        </>
      }
      stepper={st}
      stepLabel="pass"
      countFrom={0}
      caption={timelineCaption(steps, k)}
      equation={equation}
      hl={hl}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="A timeline of one request's forward passes: one long prefill bar, then a row of short decode bars; below, the FLOPs and bytes of the current pass, and the output tokens produced so far"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <text x={left} y={16} fontSize={12} fill="currentColor">
              time → {k === 0 ? "" : `${msText(s.t0 + s.dt)} of ${msText(T)}`}
            </text>
            {steps.map((u) => (
              <rect
                key={u.k}
                data-testid={`pass-${u.k}`}
                data-on={u.k === k ? "true" : undefined}
                x={x(u.t0)}
                y={barY}
                width={Math.max(1.5, x(u.t0 + u.dt) - x(u.t0) - 1)}
                height={barH}
                rx={2}
                fill={
                  u.kind === "prefill"
                    ? PHASE_COLOUR.prefill
                    : PHASE_COLOUR.decode
                }
                opacity={u.k > k ? 0.15 : u.k === k ? 1 : 0.55}
                stroke={u.k === k ? "currentColor" : "none"}
                strokeWidth={1.5}
              />
            ))}
            <text
              x={x(steps[0]!.dt / 2)}
              y={barY + barH / 2 + 4}
              fontSize={12}
              textAnchor="middle"
              fill="#000"
            >
              prefill
            </text>
            <text
              x={x(steps[0]!.dt)}
              y={barY + barH + 16}
              fontSize={11}
              fill="currentColor"
            >
              ▲ first token ({msText(steps[0]!.dt)})
            </text>
            {/* the current pass: arithmetic time against memory time */}
            <text x={left} y={meterY - 8} fontSize={11} fill="currentColor">
              this pass: the longer bar sets its time ({s.bound}-bound)
            </text>
            <rect
              x={left + 92}
              y={meterY}
              width={mw(s.tCompute)}
              height={12}
              fill={PHASE_COLOUR.prefill}
            />
            <text x={left} y={meterY + 10} fontSize={11} fill="currentColor">
              arithmetic
            </text>
            <rect
              x={left + 92}
              y={meterY + 18}
              width={mw(s.tMemory)}
              height={12}
              fill={PHASE_COLOUR.decode}
            />
            <text x={left} y={meterY + 28} fontSize={11} fill="currentColor">
              memory
            </text>
            <text x={left} y={tokensY - 6} fontSize={11} fill="currentColor">
              output tokens: {s.out}
            </text>
            {steps.map((u) => (
              <rect
                key={`tok-${u.k}`}
                x={left + u.k * tokW}
                y={tokensY}
                width={tokW - 3}
                height={tokW - 3}
                rx={3}
                fill={u.k <= k ? PHASE_COLOUR.decode : "none"}
                stroke="#a3a3a3"
              />
            ))}
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Prefill (TTFT)" value={msText(steps[0]!.dt)} />
          <Stat label="Each decode step" value={msText(steps[1]!.dt)} />
          <Stat
            label="Prefill FLOP/byte"
            value={Math.round(steps[0]!.flops / steps[0]!.bytes).toLocaleString(
              "en-GB",
            )}
          />
          <Stat
            label="Decode FLOP/byte"
            value={(steps[1]!.flops / steps[1]!.bytes).toFixed(1)}
          />
        </div>
      }
      params={
        <Segmented
          label="Prompt tokens"
          value={prompt}
          options={PROMPTS.map((p) => ({
            value: p,
            label: Number(p).toLocaleString("en-GB"),
          }))}
          onChange={setPrompt}
        />
      }
    />
  );
}
