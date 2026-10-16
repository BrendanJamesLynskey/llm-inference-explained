"use client";

/**
 * Chapters 4 to 7's mechanism: the pools at work. One lane per instance
 * with every forward pass the simulator ran (prefill orange, decode blue,
 * mixed purple), the KV link with every hand-off (sky), and each request's
 * life cut into its phases (queued, prefill, hand-off, decode), with a
 * clock sweeping simulated time. The same component draws colocated
 * against disaggregated serving (chapter 4), different hardware per pool
 * (5), links and compression (6) and CED (7), each from its own recorded
 * scenario.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import {
  boxes,
  engineReq,
  fmtSeconds,
  frameTime,
  poolsCaption,
  poolsView,
  transfers,
  variantOf,
  type Scenario,
  type StepKind,
} from "@/lib/tradeoffs/mech";
import { segments } from "@/lib/tradeoffs/timeline";
import { OKABE_ITO, PHASE_COLOUR } from "@/lib/viz/palette";

import { Hatch, Loading, Provenance, useMech } from "./common";

const FRAMES = 90;

const KIND_COLOUR: Record<StepKind, string> = {
  prefill: PHASE_COLOUR.prefill,
  decode: PHASE_COLOUR.decode,
  mixed: OKABE_ITO.purple,
  ced: OKABE_ITO.green,
};

export default function PoolWidget({
  scenario,
  initial,
  title,
  summary,
  testId,
  equation,
  stats,
}: {
  scenario: Scenario;
  initial: string;
  title: string;
  summary: ReactNode;
  testId: string;
  equation?: ReactNode;
  /** Which recorded numbers to show under the picture. */
  stats: (
    | "ttft_p99"
    | "tpot_p99"
    | "itl_p99"
    | "j_per_tok"
    | "handoff"
    | "horizon"
  )[];
}): JSX.Element {
  const f = useMech(scenario);
  const [key, setKey] = useState(initial);
  const [ref, width] = useWidth(640);
  const stepper = useStepper(FRAMES, { stepMs: 280, resetKey: key });
  const v = f ? variantOf(f, key) : null;
  const t = v ? frameTime(v, stepper.step, FRAMES) : 0;
  const view = useMemo(() => (v ? poolsView(v, t) : null), [v, t]);
  const bx = useMemo(() => (v ? boxes(v) : []), [v]);
  const segs = useMemo(
    () => (v ? v.reqs.map((r) => segments(engineReq(r))) : []),
    [v],
  );
  if (!f || !v || !view) return <Loading what="pool animation" />;
  const W = width;
  const pad = 4;
  const label = W < 560 ? 64 : 92;
  const x0 = pad + label;
  const inner = W - x0 - pad;
  const T = v.horizon;
  const tx = (s: number) => x0 + (Math.min(s, T) / T) * inner;
  const laneH = 20;
  const disagg = v.cfg.mode === "disagg";
  const tr = transfers(v);
  const nLanes = v.insts.length + (disagg ? 1 : 0);
  const reqTop = 22 + nLanes * (laneH + 4) + 18;
  const rH = Math.max(7, Math.min(12, 160 / v.reqs.length));
  const H = reqTop + v.reqs.length * (rH + 2) + 26;
  const mb = v.handoff_bytes.filter((b): b is number => b !== null);
  const meanMb = mb.length
    ? mb.reduce((a, b) => a + b, 0) / mb.length / 1e6
    : 0;
  const statFor = (k: (typeof stats)[number]) => {
    if (k === "handoff")
      return (
        <Stat
          key={k}
          label="Hand-off on the link"
          value={`${meanMb.toFixed(0)} MB`}
        />
      );
    if (k === "horizon")
      return <Stat key={k} label="All done at" value={fmtSeconds(v.horizon)} />;
    if (k === "j_per_tok")
      return (
        <Stat
          key={k}
          label="J / token"
          value={v.summary.j_per_tok.toFixed(2)}
        />
      );
    const name = {
      ttft_p99: "TTFT p99",
      tpot_p99: "TPOT p99",
      itl_p99: "ITL p99",
    }[k];
    return <Stat key={k} label={name} value={fmtSeconds(v.summary[k])} />;
  };
  return (
    <AnimationPanel
      title={title}
      summary={summary}
      stepper={stepper}
      stepLabel="frame"
      countFrom={0}
      caption={poolsCaption(v, view)}
      testId={testId}
      equation={equation}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Each instance's forward passes over time, the KV link's hand-offs and each request's phases"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <defs>
              <Hatch id={`${testId}-q`} />
            </defs>
            <text x={pad} y={14} fontSize={12} fill="currentColor">
              {v.label} · t = {fmtSeconds(t)}
            </text>
            {v.insts.map((name, i) => {
              const y = 22 + i * (laneH + 4);
              return (
                <g key={name}>
                  <text x={pad} y={y + 14} fontSize={11} fill="currentColor">
                    {W < 560
                      ? name
                          .replace("colocated-", "coloc ")
                          .replace("prefill-", "prefill ")
                          .replace("decode-", "decode ")
                      : name}
                  </text>
                  {bx[i]!.map((b, j) => (
                    <rect
                      key={j}
                      x={tx(b.t0)}
                      y={y}
                      width={Math.max(0.6, tx(b.t1) - tx(b.t0) - 0.3)}
                      height={laneH}
                      fill={KIND_COLOUR[b.kind]}
                      opacity={b.t0 > t ? 0.12 : b.t1 > t ? 1 : 0.7}
                    />
                  ))}
                </g>
              );
            })}
            {disagg && (
              <g>
                <text
                  x={pad}
                  y={22 + v.insts.length * (laneH + 4) + 14}
                  fontSize={11}
                  fill="currentColor"
                >
                  KV link
                </text>
                {tr.map(([rid, a, b]) => (
                  <rect
                    key={rid}
                    x={tx(a)}
                    y={22 + v.insts.length * (laneH + 4) + 3}
                    width={Math.max(1, tx(b) - tx(a))}
                    height={laneH - 6}
                    fill={PHASE_COLOUR.handoff}
                    stroke={a <= t && t < b ? OKABE_ITO.blue : "none"}
                    opacity={a > t ? 0.15 : 0.9}
                  >
                    <title>{`request ${rid}: hand-off ${fmtSeconds(b - a)}`}</title>
                  </rect>
                ))}
              </g>
            )}
            <text x={pad} y={reqTop - 6} fontSize={11} fill="currentColor">
              Requests: queued · prefill · hand-off · decode
            </text>
            {segs.map((ss, rid) =>
              ss.map((s, j) =>
                s.t0 > t ? null : (
                  <rect
                    key={`${rid}-${j}`}
                    x={tx(s.t0)}
                    y={reqTop + rid * (rH + 2)}
                    width={Math.max(0.6, tx(Math.min(s.t1, t)) - tx(s.t0))}
                    height={rH}
                    fill={
                      s.phase === "queue"
                        ? `url(#${testId}-q)`
                        : PHASE_COLOUR[s.phase]
                    }
                    stroke={s.phase === "queue" ? PHASE_COLOUR.queue : "none"}
                    strokeWidth={0.5}
                  />
                ),
              ),
            )}
            <line
              x1={tx(t)}
              x2={tx(t)}
              y1={20}
              y2={H - 22}
              stroke={OKABE_ITO.vermillion}
              strokeWidth={1.5}
            />
            <text x={pad} y={H - 6} fontSize={11} fill="currentColor">
              0
            </text>
            <text
              x={W - pad}
              y={H - 6}
              fontSize={11}
              textAnchor="end"
              fill="currentColor"
            >
              {fmtSeconds(T)}
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {stats.map(statFor)}
        </div>
      }
      params={
        <div className="sm:col-span-2">
          <Segmented
            label="Configuration"
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
