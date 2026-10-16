"use client";

/**
 * The landing page's hero: the same requests on the same eight H100s,
 * colocated and disaggregated, as lanes the serving simulator computed
 * (hatched: queued; orange: prefill; sky: the KV hand-off; blue: decode,
 * with its stalls cut out in red hatching). A clock sweeps across both.
 * The page renders the first frame on the server from the vendored engine;
 * once the hero is on screen the browser runs the same engine on all 200
 * requests in a Web Worker and says whether it got the same answer.
 */
import { useEffect, useMemo, useState } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Hatch } from "@/components/mech/common";
import { Stat } from "@/components/ui/Controls";
import type { Metrics } from "@/lib/disagg/engine";
import {
  HERO_LANES,
  HERO_URL,
  heroCaption,
  heroHorizon,
  heroLanes,
  heroTime,
  laneFrame,
  type HeroFile,
  type Lane,
} from "@/lib/disagg/hero";
import { OKABE_ITO, PHASE_COLOUR } from "@/lib/viz/palette";

import { simulateAll } from "../interactive/disagg/useRuns";

const FRAMES = 60;

export type TeaserData = {
  colocated: { lanes: Lane[]; metrics: Metrics };
  disagg: { lanes: Lane[]; metrics: Metrics };
  requests: number;
  commit: string;
};

type Check = { state: "idle" | "running" | "same" | "different"; ms?: number };

function sameLanes(a: Lane[], b: Lane[]): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export default function ServingTeaserHero({
  data,
}: {
  data: TeaserData;
}): JSX.Element {
  const st = useStepper(FRAMES, { stepMs: 160, smooth: false });
  const [ref, width] = useWidth(600);
  const [check, setCheck] = useState<Check>({ state: "idle" });
  const T = useMemo(
    () => heroHorizon(data.colocated.lanes, data.disagg.lanes),
    [data],
  );
  const t = heroTime(st.step, FRAMES, T);

  // re-run the engine in the browser once, when the hero is first shown
  useEffect(() => {
    if (check.state !== "idle") return;
    const el = document.querySelector('[data-testid="hero-teaser"]');
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((es) => {
      if (!es.some((e) => e.isIntersecting)) return;
      io.disconnect();
      setCheck({ state: "running" });
      const t0 = performance.now();
      fetch(HERO_URL)
        .then((r) => r.json() as Promise<HeroFile>)
        .then((f) =>
          simulateAll(
            f.rows,
            Object.fromEntries(f.modes.map((m) => [m.key, m.cfg])) as Record<
              "colocated" | "disagg",
              (typeof f.modes)[number]["cfg"]
            >,
          ),
        )
        .then((runs) => {
          const ok =
            sameLanes(
              heroLanes(runs.colocated.result.reqs),
              data.colocated.lanes,
            ) &&
            sameLanes(heroLanes(runs.disagg.result.reqs), data.disagg.lanes) &&
            runs.colocated.metrics.itl_p99 === data.colocated.metrics.itl_p99 &&
            runs.disagg.metrics.itl_p99 === data.disagg.metrics.itl_p99;
          setCheck({
            state: ok ? "same" : "different",
            ms: performance.now() - t0,
          });
        })
        .catch(() => setCheck({ state: "different" }));
    });
    io.observe(el);
    return () => io.disconnect();
  }, [check.state, data]);

  const W = width;
  const label = W < 480 ? 0 : 70;
  const laneH = W < 480 ? 9 : 11;
  const left = 6 + label;
  const span = W - left - 8;
  const x = (s: number) => left + (s / T) * span;
  const panelH = 18 + HERO_LANES * (laneH + 3) + 8;
  const narrow = W < 480;
  const H = 2 * panelH + (narrow ? 44 : 30);
  const modes = [
    {
      key: "colocated",
      title: "Colocated: 2 × (4 H100s), each doing both",
      d: data.colocated,
    },
    {
      key: "disagg",
      title: "Disaggregated: 4 H100s prefill, 4 decode",
      d: data.disagg,
    },
  ] as const;
  const frames = modes.map((m) => laneFrame(m.d.lanes, t));
  const ms = (s: number) => `${(s * 1e3).toFixed(s < 0.1 ? 1 : 0)} ms`;
  return (
    <AnimationPanel
      testId="hero-teaser"
      title="The same requests, colocated or split"
      summary={
        <>
          Llama-3-70B on eight H100s, the first {HERO_LANES} of {data.requests}{" "}
          requests, simulated by Disaggregated_Inference_Sim (commit{" "}
          <span className="font-mono">{data.commit.slice(0, 7)}</span>
          ).
        </>
      }
      stepper={st}
      stepLabel="frame"
      countFrom={0}
      caption={heroCaption(frames[0]!, frames[1]!)}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Two timelines of eight requests each, colocated above and disaggregated below: queueing, prefill, KV hand-off and decode bars, with decode stalls marked, and a clock line sweeping across"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <defs>
              <Hatch id="teaser-queue" />
              <Hatch id="teaser-stall" colour={OKABE_ITO.vermillion} />
            </defs>
            {modes.map((m, mi) => {
              const y0 = mi * (panelH + 14);
              return (
                <g key={m.key} data-testid={`teaser-${m.key}`}>
                  <text x={6} y={y0 + 12} fontSize={11} fill="currentColor">
                    {m.title}
                  </text>
                  {m.d.lanes.map((l, i) => {
                    const y = y0 + 18 + i * (laneH + 3);
                    const seg = (
                      a: number,
                      b: number,
                      fill: string,
                      key: string,
                    ) => {
                      const a1 = Math.min(a, t);
                      const b1 = Math.min(b, t);
                      if (b1 <= a1) return null;
                      return (
                        <rect
                          key={key}
                          x={x(a1)}
                          y={y}
                          width={Math.max(0.8, x(b1) - x(a1))}
                          height={laneH}
                          fill={fill}
                        />
                      );
                    };
                    return (
                      <g
                        key={i}
                        data-testid={`teaser-${m.key}-${i}`}
                        data-phase={frames[mi]!.phases[i]}
                      >
                        {label > 0 && (
                          <text
                            x={6}
                            y={y + laneH - 1}
                            fontSize={10}
                            fill="currentColor"
                          >
                            req {i + 1}
                          </text>
                        )}
                        {seg(
                          l.arrival,
                          l.prefillStart,
                          "url(#teaser-queue)",
                          "q",
                        )}
                        {seg(
                          l.prefillStart,
                          l.firstToken,
                          PHASE_COLOUR.prefill,
                          "p",
                        )}
                        {l.kvStart !== null &&
                          l.kvReady !== null &&
                          seg(l.kvStart, l.kvReady, PHASE_COLOUR.handoff, "h")}
                        {seg(l.decodeStart, l.finish, PHASE_COLOUR.decode, "d")}
                        {l.stalls.map(([a, b], k) =>
                          seg(a, b, "url(#teaser-stall)", `s${k}`),
                        )}
                      </g>
                    );
                  })}
                </g>
              );
            })}
            <line
              x1={x(t)}
              x2={x(t)}
              y1={14}
              y2={H - 14}
              stroke="currentColor"
              strokeWidth={1}
              strokeDasharray="2 2"
            />
            <text
              x={6}
              y={narrow ? H - 18 : H - 2}
              fontSize={11}
              fill="currentColor"
            >
              t = {t.toFixed(2)} s · hatched red: a decode stall
              {narrow ? "" : " (a gap over 3× the request’s median)"}
            </text>
            {narrow && (
              <text x={6} y={H - 3} fontSize={11} fill="currentColor">
                (a gap over 3× the request&apos;s median)
              </text>
            )}
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="ITL p99, colocated"
            value={ms(data.colocated.metrics.itl_p99)}
          />
          <Stat
            label="ITL p99, split"
            value={ms(data.disagg.metrics.itl_p99)}
          />
          <Stat
            label="TTFT p99, colocated"
            value={ms(data.colocated.metrics.ttft_p99)}
          />
          <Stat
            label="TTFT p99, split"
            value={ms(data.disagg.metrics.ttft_p99)}
          />
        </div>
      }
      params={
        <p
          data-testid="teaser-check"
          data-check={check.state}
          className="text-xs text-neutral-600 dark:text-neutral-400"
          aria-live="polite"
        >
          {check.state === "same"
            ? `Re-run in your browser just now: all ${data.requests} requests, both layouts, the same timestamps (${Math.round(check.ms ?? 0)} ms).`
            : check.state === "different"
              ? "Re-running in your browser did not give the same answer."
              : check.state === "running"
                ? `Re-running all ${data.requests} requests in your browser…`
                : `When this is on screen, your browser re-runs all ${data.requests} requests with the same engine.`}
        </p>
      }
    />
  );
}
