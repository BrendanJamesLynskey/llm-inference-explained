/**
 * Static SVG for the landing page: cumulative FLOPs of generating with and
 * without the KV cache. Server Component (no client JavaScript); the data
 * is computed on the server by `lib/inference/heroPreview.ts`.
 */
import type { HeroPreview } from "@/lib/inference/heroPreview";

const W = 280;
const H = 180;
const PAD = { l: 8, r: 8, t: 12, b: 24 };

export function HeroFlopsSvg({ data }: { data: HeroPreview }): JSX.Element {
  const n = data.cached.length;
  const max = data.uncached[n - 1]!;
  const x = (i: number): number => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number): number => H - PAD.b - (v / max) * (H - PAD.t - PAD.b);
  const line = (vs: number[]): string =>
    vs
      .map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
      .join("");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label={`Cumulative FLOPs over ${n} generated tokens: ${data.uncached[n - 1]!.toLocaleString("en-GB")} recomputing every step, ${data.cached[n - 1]!.toLocaleString("en-GB")} with the KV cache.`}
    >
      <line
        x1={PAD.l}
        x2={W - PAD.r}
        y1={H - PAD.b}
        y2={H - PAD.b}
        className="stroke-neutral-300 dark:stroke-neutral-700"
      />
      <path
        d={line(data.uncached)}
        fill="none"
        strokeWidth={2.5}
        className="stroke-neutral-500"
      />
      <path
        d={line(data.cached)}
        fill="none"
        strokeWidth={2.5}
        className="stroke-indigo-500 dark:stroke-indigo-400"
      />
      <text
        x={x(Math.round((n - 1) * 0.72))}
        y={PAD.t + 4}
        textAnchor="end"
        className="fill-neutral-600 text-[11px] dark:fill-neutral-400"
      >
        recompute every step
      </text>
      <text
        x={x(n - 1)}
        y={y(data.cached[n - 1]!) - 6}
        textAnchor="end"
        className="fill-indigo-700 text-[11px] dark:fill-indigo-300"
      >
        with the KV cache
      </text>
      <text
        x={PAD.l}
        y={H - 6}
        className="fill-neutral-500 text-[10px] dark:fill-neutral-400"
      >
        tokens generated →
      </text>
    </svg>
  );
}
