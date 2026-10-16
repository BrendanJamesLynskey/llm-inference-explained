"use client";

/**
 * Chapter 2's hero: the KV cache filling, step by step, on the trained toy
 * model (or its random init). Prefill writes the prompt's keys and values
 * into every block at once; then each decode step's new token asks every
 * cached key (the head rows above each key grid light up by attention
 * weight), its own key and value join the cache as one new column, and
 * the next token is picked. Every number is the model's (`kvHeroStates`).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import { kvHeroCaption, kvHeroStates } from "@/lib/inference/kvHero";
import {
  TOY_CONFIG,
  toyLabel,
  toyWeights,
  type ToyKind,
} from "@/lib/inference/toyModel";
import { OKABE_ITO } from "@/lib/viz/palette";

import { useToyWeights } from "../interactive/useToyWeights";

export const KV_HERO_PROMPT = "said";
export const KV_HERO_ROUNDS = 4;

/** A value as a fill: orange positive, blue negative, strength by size. */
function cellFill(v: number, max: number): { fill: string; opacity: number } {
  const a = Math.min(1, Math.abs(v) / (max || 1));
  return {
    fill: v >= 0 ? OKABE_ITO.orange : OKABE_ITO.blue,
    opacity: 0.08 + 0.9 * a,
  };
}

export default function KvFillHero({
  equation,
}: {
  equation?: ReactNode;
}): JSX.Element {
  const [kind, setKind] = useState<ToyKind>("trained");
  const { ready, error } = useToyWeights(kind);
  if (error)
    return (
      <p role="alert" className="my-8 text-sm text-red-700 dark:text-red-300">
        Could not load the trained weights ({error}).{" "}
        <button
          type="button"
          className="focus-ring underline"
          onClick={() => setKind("random")}
        >
          Use random init
        </button>
      </p>
    );
  if (!ready)
    return (
      <p
        data-pending-widget
        className="my-8 min-h-96 text-sm text-neutral-600 dark:text-neutral-400"
      >
        Loading the trained model…
      </p>
    );
  return <Panel kind={kind} setKind={setKind} equation={equation} />;
}

function Panel({
  kind,
  setKind,
  equation,
}: {
  kind: ToyKind;
  setKind: (k: ToyKind) => void;
  equation?: ReactNode;
}): JSX.Element {
  const states = useMemo(
    () => kvHeroStates(KV_HERO_PROMPT, KV_HERO_ROUNDS, toyWeights(kind)),
    [kind],
  );
  const st = useStepper(states.length, { stepMs: 1000, resetKey: kind });
  const [ref, width] = useWidth(640);
  const s = states[st.step]!;
  const last = states[states.length - 1]!;
  const S = last.ids.length; // positions drawn
  const D = TOY_CONFIG.d_model;
  const H = TOY_CONFIG.n_heads;
  const W = width;
  const phone = W < 520;
  // layout: tokens on top; per block a K grid and a V grid side by side
  // (stacked on a phone), each with the head rows of attention above K
  const gridW = phone ? W - 16 : (W - 40) / 4;
  const cell = Math.min(18, (gridW - 24) / S);
  const ch = Math.max(5, Math.min(9, cell * 0.5));
  const tokY = 8;
  const tokH = Math.min(30, cell * 1.6);
  const blockY0 = tokY + tokH + 28;
  const attnH = H * (cell * 0.7);
  const gridH = attnH + 16 + D * ch;
  const blockH = gridH + 22;
  const maxAbs = useMemo(() => {
    let m = 0;
    for (const K of last.K)
      for (const r of K) for (const v of r) m = Math.max(m, Math.abs(v));
    for (const V of last.V)
      for (const r of V) for (const v of r) m = Math.max(m, Math.abs(v));
    return m;
  }, [last]);
  const grids: { b: number; which: "K" | "V"; gx: number; gy: number }[] = [];
  for (let b = 0; b < TOY_CONFIG.n_blocks; b++)
    for (const which of ["K", "V"] as const) {
      const i = b * 2 + (which === "K" ? 0 : 1);
      grids.push(
        phone
          ? { b, which, gx: 8, gy: blockY0 + i * blockH }
          : { b, which, gx: 8 + i * (gridW + 8), gy: blockY0 },
      );
    }
  const SVG_H = phone ? blockY0 + 4 * blockH + 4 : blockY0 + blockH + 4;
  const fresh = (pos: number) =>
    (s.phase === "append" || s.phase === "pick" || s.phase === "prefill") &&
    (s.phase === "prefill" ? true : pos === s.pos);
  const hl =
    s.phase === "attend"
      ? "a"
      : s.phase === "append"
        ? "c"
        : s.phase === "pick"
          ? "p"
          : "c";
  return (
    <AnimationPanel
      testId="hero-kv"
      title="The KV cache filling, one column per step"
      summary={
        <>
          The toy decoder (
          {kind === "trained" ? "trained on one book" : "random weights"}, 2
          blocks, 2 heads, width 16) continues “{KV_HERO_PROMPT}”. Each block
          keeps a key and a value per token; the rows above the keys show how
          much the newest token&apos;s query reads each cached key, per head.
        </>
      }
      stepper={st}
      stepLabel="step"
      countFrom={0}
      caption={kvHeroCaption(s)}
      equation={equation}
      hl={hl}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={SVG_H}
            viewBox={`0 0 ${W} ${SVG_H}`}
            role="img"
            aria-label="The tokens so far; for each of the two blocks, a grid of cached keys and a grid of cached values, one column per token, with attention rows above the keys showing the newest token's weights per head"
            className="text-neutral-800 dark:text-neutral-200"
          >
            {Array.from({ length: S }, (_, i) => {
              const have = i < s.ids.length;
              const now = i === s.pos && s.phase !== "prefill";
              return (
                <g key={`t${i}`} data-testid={`kv-token-${i}`}>
                  <rect
                    x={8 + 24 + i * cell}
                    y={tokY}
                    width={cell - 2}
                    height={tokH}
                    rx={3}
                    fill={now ? OKABE_ITO.sky : "none"}
                    fillOpacity={now ? 0.35 : 0}
                    stroke={have ? "currentColor" : "#a3a3a3"}
                    strokeDasharray={have ? undefined : "2 2"}
                  />
                  {have && (
                    <text
                      x={8 + 24 + i * cell + (cell - 2) / 2}
                      y={tokY + tokH / 2 + 4}
                      fontSize={11}
                      textAnchor="middle"
                      fill="currentColor"
                    >
                      {toyLabel(s.ids[i]!)}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={8} y={tokY + tokH + 16} fontSize={11} fill="currentColor">
              cached: {s.cached} of {S} positions
            </text>
            {grids.map(({ b, which, gx, gy }) => {
              const M = which === "K" ? s.K[b]! : s.V[b]!;
              const ox = gx + 24;
              const gy0 = gy + attnH + 16;
              return (
                <g key={`${b}${which}`} data-testid={`kv-${b}-${which}`}>
                  <text x={gx} y={gy + 10} fontSize={11} fill="currentColor">
                    block {b + 1} {which === "K" ? "keys" : "values"}
                  </text>
                  {which === "K" &&
                    s.attn &&
                    s.attn[b]!.map((row, h) =>
                      row.map((w, j) => (
                        <rect
                          key={`a${h}-${j}`}
                          data-testid={`kv-attn-${b}-${h}-${j}`}
                          x={ox + j * cell}
                          y={gy + 14 + h * cell * 0.7}
                          width={cell - 2}
                          height={cell * 0.7 - 2}
                          fill={OKABE_ITO.green}
                          opacity={0.08 + 0.92 * w}
                        >
                          <title>{`head ${h} → position ${j}: ${w.toFixed(3)}`}</title>
                        </rect>
                      )),
                    )}
                  {M.map((row, j) => (
                    <g
                      key={`c${j}`}
                      data-testid={`kv-col-${b}-${which}-${j}`}
                      data-fresh={fresh(j) ? "true" : undefined}
                    >
                      {row.map((v, r) => {
                        const f = cellFill(v, maxAbs);
                        return (
                          <rect
                            key={r}
                            x={ox + j * cell}
                            y={gy0 + r * ch}
                            width={cell - 2}
                            height={ch - 1}
                            fill={f.fill}
                            opacity={f.opacity}
                          />
                        );
                      })}
                      {fresh(j) && s.phase !== "prefill" && (
                        <rect
                          x={ox + j * cell - 1}
                          y={gy0 - 1}
                          width={cell}
                          height={D * ch + 1}
                          fill="none"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        />
                      )}
                    </g>
                  ))}
                  {Array.from({ length: S - M.length }, (_, e) => (
                    <rect
                      key={`e${e}`}
                      x={ox + (M.length + e) * cell}
                      y={gy0}
                      width={cell - 2}
                      height={D * ch - 1}
                      fill="none"
                      stroke="#a3a3a3"
                      strokeDasharray="2 2"
                    />
                  ))}
                </g>
              );
            })}
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Cached positions" value={String(s.cached)} />
          <Stat
            label="Cache numbers"
            value={String(2 * TOY_CONFIG.n_blocks * s.cached * D)}
            hint="2 × blocks × positions × width"
          />
          <Stat
            label="Text"
            value={`“${s.ids.map((i) => toyLabel(i)).join("")}”`}
          />
          <Stat label="Weights" value={kind} />
        </div>
      }
      params={
        <Segmented
          label="Weights"
          value={kind}
          options={[
            { value: "trained", label: "trained" },
            { value: "random", label: "random init" },
          ]}
          onChange={setKind}
        />
      }
    />
  );
}
