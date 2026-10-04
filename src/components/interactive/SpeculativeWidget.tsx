"use client";

/**
 * Speculative decoding: the closed-form expected tokens per target pass and
 * speed-up, and a Monte Carlo run of the real draft-and-verify rule.
 */
import { useMemo, useState } from "react";

import { ActionButton, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  acceptanceRate,
  bestGamma,
  expectedTokens,
  simulateSpeculative,
  speedup,
} from "@/lib/inference/speculative";
import { softmax } from "@/lib/transformer/softmax";

const LOGITS = [2.2, 1.6, 1.1, 0.7, 0.2, -0.3, -0.8, -1.2];
const G_MAX = 12;

export function SpeculativeWidget(): JSX.Element {
  const [alpha, setAlpha] = useState(0.8);
  const [gamma, setGamma] = useState(4);
  const [cost, setCost] = useState(0.05);
  const [mismatch, setMismatch] = useState(0.6);
  const [seed, setSeed] = useState(1);

  const curve = Array.from({ length: G_MAX }, (_, i) =>
    speedup(alpha, i + 1, cost),
  );
  const top = Math.max(...curve, 1);
  const best = bestGamma(alpha, cost, G_MAX);

  // Toy target p and draft q: the draft has the target's logits, perturbed.
  const p = useMemo(() => softmax(LOGITS), []);
  const q = useMemo(
    () =>
      softmax(LOGITS.map((l, i) => l * (1 - mismatch) + (i % 3) * mismatch)),
    [mismatch],
  );
  const a = acceptanceRate(p, q);
  const pMax = Math.max(...p) * 1.15;
  const sim = useMemo(
    () => simulateSpeculative(p, q, gamma, 4000, seed),
    [p, q, gamma, seed],
  );

  return (
    <WidgetFrame
      testId="speculative"
      title="Draft γ tokens, verify them in one pass"
      caption="Top: Leviathan et al.'s closed forms (i.i.d. acceptance α; one draft step costs c of a target step). Bottom: 4,000 rounds of the real accept/reject rule on an 8-token toy vocabulary."
    >
      <div className="grid gap-3 sm:grid-cols-3">
        <Slider
          label="Acceptance α"
          value={alpha}
          min={0}
          max={0.99}
          step={0.01}
          onChange={setAlpha}
          format={(v) => v.toFixed(2)}
        />
        <Slider
          label="Draft length γ"
          value={gamma}
          min={1}
          max={G_MAX}
          onChange={setGamma}
        />
        <Slider
          label="Draft cost c"
          value={cost}
          min={0}
          max={0.5}
          step={0.01}
          onChange={setCost}
          format={(v) => v.toFixed(2)}
        />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Tokens / target pass"
          value={expectedTokens(alpha, gamma).toFixed(2)}
        />
        <Stat
          label="Speed-up"
          value={`${speedup(alpha, gamma, cost).toFixed(2)}×`}
        />
        <Stat
          plain
          label="Best γ here"
          value={String(best)}
          hint={`${speedup(alpha, best, cost).toFixed(2)}×`}
        />
        <Stat label="Tokens if all accepted" value={String(gamma + 1)} />
      </div>
      <svg
        viewBox="0 0 480 120"
        className="mt-3 h-auto w-full"
        role="img"
        aria-label={`Speed-up against draft length for alpha ${alpha.toFixed(2)} and cost ${cost.toFixed(2)}; best gamma ${best}`}
      >
        <line
          x1={30}
          x2={470}
          y1={100 - (1 / top) * 85}
          y2={100 - (1 / top) * 85}
          strokeDasharray="3 3"
          className="stroke-neutral-400"
        />
        <text
          x={472}
          y={100 - (1 / top) * 85 + 3}
          className="fill-neutral-500 text-[9px] dark:fill-neutral-400"
          textAnchor="end"
        >
          1×
        </text>
        {curve.map((v, i) => {
          const h = (v / top) * 85;
          return (
            <g key={i}>
              <rect
                x={34 + i * 36}
                y={100 - h}
                width={28}
                height={h}
                className={
                  i + 1 === gamma
                    ? "fill-indigo-500"
                    : "fill-indigo-200 dark:fill-indigo-900"
                }
              />
              <text
                x={48 + i * 36}
                y={112}
                textAnchor="middle"
                className="fill-neutral-500 text-[9px] dark:fill-neutral-400"
              >
                {i + 1}
              </text>
            </g>
          );
        })}
        <text
          x={2}
          y={112}
          className="fill-neutral-500 text-[9px] dark:fill-neutral-400"
        >
          γ
        </text>
      </svg>

      <div className="mt-6 grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
        <Slider
          label="Draft–target mismatch"
          value={mismatch}
          min={0}
          max={1}
          step={0.05}
          onChange={setMismatch}
          format={(v) => v.toFixed(2)}
        />
        <ActionButton variant="secondary" onClick={() => setSeed((s) => s + 1)}>
          Re-run (seed {seed})
        </ActionButton>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat plain label="α = Σ min(p, q)" value={a.toFixed(3)} />
        <Stat label="Measured acceptance" value={sim.acceptance.toFixed(3)} />
        <Stat
          label="Tokens / pass"
          value={sim.tokensPerPass.toFixed(2)}
          hint={`formula ${expectedTokens(a, gamma).toFixed(2)}`}
        />
        <Stat plain label="γ" value={String(gamma)} />
      </div>
      <div
        className="mt-3"
        aria-label="Emitted token frequencies against the target distribution"
      >
        {p.map((pi, i) => (
          <div key={i} className="flex items-center gap-2 text-[0.7rem]">
            <span className="w-10 shrink-0 font-mono text-neutral-500 dark:text-neutral-400">
              tok {i}
            </span>
            <span className="min-w-0 flex-1">
              <span
                className="block h-1.5 rounded bg-neutral-400 dark:bg-neutral-500"
                style={{ width: `${Math.min(100, (pi / pMax) * 100)}%` }}
              />
              <span
                className="mt-0.5 block h-1.5 rounded bg-indigo-500"
                style={{
                  width: `${Math.min(100, (sim.histogram[i]! / pMax) * 100)}%`,
                }}
              />
            </span>
            <span className="w-24 shrink-0 text-right font-mono text-neutral-600 dark:text-neutral-400">
              {(pi * 100).toFixed(1)} / {(sim.histogram[i]! * 100).toFixed(1)}%
            </span>
          </div>
        ))}
        <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          Grey: the target distribution p. Blue: what speculative decoding
          actually emitted. They match whatever the draft is: only the speed
          changes.
        </p>
      </div>
    </WidgetFrame>
  );
}
