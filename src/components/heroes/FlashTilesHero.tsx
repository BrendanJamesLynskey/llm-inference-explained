"use client";

/**
 * Chapter 6's hero: FlashAttention's forward pass, tile by tile, on 16
 * queries and keys. The score matrix is drawn as the tiles of it that have
 * been computed; each tile lives only in on-chip memory (it is computed,
 * folded into the rows' running max m and sum l, and dropped), so the N×N
 * scores never reach HBM. When a block of query rows has seen every key
 * block, its output rows are normalised and written. Every state comes
 * from `flashTileStates` (the site's tested online softmax, `foldTile`).
 */
import { useMemo, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Stat } from "@/components/ui/Controls";
import {
  FLASH_HERO,
  flashCaption,
  flashHeroInputs,
  flashTileStates,
  hbmStandard,
} from "@/lib/inference/attentionKernels";
import { LEVEL_COLOUR, OKABE_ITO } from "@/lib/viz/palette";

export default function FlashTilesHero({
  equation,
}: {
  equation?: ReactNode;
}): JSX.Element {
  const { n, d, br, bc } = FLASH_HERO;
  const states = useMemo(() => {
    const { Q, K, V } = flashHeroInputs();
    return flashTileStates(Q, K, V, br, bc);
  }, [br, bc]);
  const st = useStepper(states.length, { stepMs: 900 });
  const [ref, width] = useWidth(640);
  const s = states[st.step]!;
  const W = width;
  const phone = W < 520;
  const side = Math.min(phone ? W - 120 : 300, 300);
  const cell = side / n;
  const ox = 26;
  const oy = 28;
  const statX = phone ? ox : ox + side + 24;
  const statY = phone ? oy + side + 34 : oy;
  const barW = phone ? W - 2 * ox - 60 : W - statX - 70;
  const H = phone
    ? statY + n * 13 + 40
    : Math.max(oy + side + 30, statY + n * 13 + 40);
  const lMax = Math.max(...s.l, 1);
  const naiveHbm = hbmStandard(n, d);
  return (
    <AnimationPanel
      testId="hero-flash"
      title="FlashAttention, one tile at a time"
      summary={
        <>
          {n} queries against {n} keys of width {d}, in tiles of {br} queries by{" "}
          {bc} keys. Each tile&apos;s scores are computed in on-chip memory
          (green), folded into every row&apos;s running max m and sum l, and
          dropped; only the output rows (purple) are written to HBM.
        </>
      }
      stepper={st}
      stepLabel="tile"
      caption={flashCaption(s, n)}
      equation={equation}
      hl={s.cols[1] === n ? "c" : "a"}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="The 16 by 16 score matrix drawn as tiles: computed tiles are faint, the current tile is outlined; beside it, each query row's running sum as a bar, and the rows already written"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <text x={ox} y={14} fontSize={11} fill="currentColor">
              scores Q·Kᵀ: keys across, queries down (never stored)
            </text>
            {states.map((u) => {
              const done = u.t < s.t;
              const now = u.t === s.t;
              if (!done && !now) return null;
              return (
                <rect
                  key={u.t}
                  data-testid={`tile-${u.qi}-${u.kj}`}
                  data-now={now ? "true" : undefined}
                  x={ox + u.cols[0] * cell}
                  y={oy + u.rows[0] * cell}
                  width={(u.cols[1] - u.cols[0]) * cell - 1}
                  height={(u.rows[1] - u.rows[0]) * cell - 1}
                  fill={LEVEL_COLOUR.smem}
                  opacity={now ? 0.85 : 0.18}
                  stroke={now ? "currentColor" : "none"}
                  strokeWidth={2}
                />
              );
            })}
            <rect
              x={ox}
              y={oy}
              width={side}
              height={side}
              fill="none"
              stroke="#a3a3a3"
            />
            <text x={statX} y={statY - 8} fontSize={11} fill="currentColor">
              each row: running sum l, written to HBM
            </text>
            {s.l.map((l, i) => {
              const started = s.m[i] !== -Infinity;
              const written = s.written.includes(i);
              return (
                <g
                  key={i}
                  data-testid={`row-${i}`}
                  data-written={written ? "true" : undefined}
                >
                  <text
                    x={statX}
                    y={statY + i * 13 + 10}
                    fontSize={11}
                    fill="currentColor"
                  >
                    {i}
                  </text>
                  <rect
                    x={statX + 18}
                    y={statY + i * 13 + 2}
                    width={started ? Math.max(1, (l / lMax) * barW) : 0}
                    height={9}
                    fill={written ? LEVEL_COLOUR.hbm : OKABE_ITO.sky}
                  />
                </g>
              );
            })}
            <text
              x={statX}
              y={statY + n * 13 + 22}
              fontSize={11}
              fill="currentColor"
            >
              HBM traffic: {s.hbm.toLocaleString("en-GB")} numbers (the textbook
              way: {naiveHbm.toLocaleString("en-GB")})
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Tile" value={`${s.t + 1} of ${states.length}`} />
          <Stat label="Rows written" value={`${s.written.length} of ${n}`} />
          <Stat label="HBM numbers" value={s.hbm.toLocaleString("en-GB")} />
          <Stat
            label="Textbook HBM"
            value={naiveHbm.toLocaleString("en-GB")}
            hint="4Nd + 4N²"
          />
        </div>
      }
    />
  );
}
