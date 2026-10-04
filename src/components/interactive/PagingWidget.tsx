"use client";

/**
 * A paged KV cache, step by step: a pool of blocks, each sequence's block
 * table, a shared prompt prefix, and preemption when the pool runs dry.
 */
import { useMemo, useState } from "react";

import { ActionButton, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { runScenario, type ScenarioRequest } from "@/lib/inference/paging";

const BLOCK = 4;
const MAX_LEN = 64;
const STEPS = 90;
const REQS: ScenarioRequest[] = [
  { id: 0, arrive: 0, prompt: 13, output: 30 },
  { id: 1, arrive: 2, prompt: 6, output: 40 },
  { id: 2, arrive: 4, prompt: 13, output: 18, sharesWith: 0 },
  { id: 3, arrive: 8, prompt: 22, output: 24 },
  { id: 4, arrive: 12, prompt: 9, output: 35 },
  { id: 5, arrive: 20, prompt: 17, output: 20 },
];

const COLOURS = [
  "bg-emerald-400 dark:bg-emerald-600",
  "bg-sky-400 dark:bg-sky-600",
  "bg-violet-400 dark:bg-violet-600",
  "bg-rose-400 dark:bg-rose-600",
  "bg-amber-400 dark:bg-amber-600",
  "bg-teal-400 dark:bg-teal-600",
];

export function PagingWidget(): JSX.Element {
  const [blocks, setBlocks] = useState(28);
  const [step, setStep] = useState(0);
  const frames = useMemo(
    () => runScenario(REQS, blocks, BLOCK, MAX_LEN, STEPS),
    [blocks],
  );
  const f = frames[Math.min(step, frames.length - 1)]!;
  const lastEvent = [...frames.slice(0, step + 1)]
    .reverse()
    .find((x) => x.event)?.event;

  return (
    <WidgetFrame
      testId="paging"
      title="Blocks, block tables and preemption"
      caption={`Six requests share a pool of ${BLOCK}-token blocks. Each step every running sequence appends one token. Request 2 has the same prompt as request 0, so it reuses its full blocks. When a sequence needs a block and none is free, the latest-arrived running request is preempted and recomputed later.`}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Slider
          label="Step"
          value={step}
          min={0}
          max={STEPS - 1}
          onChange={setStep}
        />
        <Slider
          label="Pool size"
          value={blocks}
          min={14}
          max={48}
          onChange={(v) => {
            setBlocks(v);
          }}
          format={(v) => `${v} blocks (${v * BLOCK} tokens)`}
        />
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <ActionButton
          variant="secondary"
          onClick={() => setStep((s) => Math.max(0, s - 1))}
        >
          ← step
        </ActionButton>
        <ActionButton
          variant="secondary"
          onClick={() => setStep((s) => Math.min(STEPS - 1, s + 1))}
        >
          step →
        </ActionButton>
      </div>

      <div
        className="mt-4 grid grid-cols-8 gap-1 sm:grid-cols-12"
        aria-label={`Block pool at step ${f.step}`}
      >
        {f.owners.map((o, i) => (
          <div
            key={i}
            title={
              o === -1
                ? `block ${i}: free`
                : o === -2
                  ? `block ${i}: shared`
                  : `block ${i}: request ${o}`
            }
            className={`flex h-7 items-center justify-center rounded font-mono text-[0.65rem] ${
              o === -1
                ? "bg-neutral-200 text-neutral-500 dark:bg-neutral-800"
                : o === -2
                  ? "bg-neutral-700 text-white dark:bg-neutral-300 dark:text-neutral-900"
                  : `${COLOURS[o % COLOURS.length]} text-neutral-950`
            }`}
          >
            {o === -1 ? "" : o === -2 ? "S" : o}
          </div>
        ))}
      </div>
      <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
        Numbers: the request owning each block. S: a block shared by several
        sequences (reference count &gt; 1). Grey: free.
      </p>

      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Running"
          value={f.running.map((r) => `#${r}`).join(" ") || "–"}
        />
        <Stat
          label="Waiting"
          value={f.waiting.map((r) => `#${r}`).join(" ") || "–"}
        />
        <Stat
          label="Finished"
          value={f.finished.map((r) => `#${r}`).join(" ") || "–"}
        />
        <Stat label="Blocks used" value={`${f.stats.usedBlocks} / ${blocks}`} />
        <Stat
          label="Paged slack"
          value={`${f.stats.slack} slots`}
          hint="last-block gaps"
        />
        <Stat
          label="Contiguous waste"
          value={`${f.contiguousWaste} slots`}
          hint={`${MAX_LEN}-token slabs`}
        />
        <Stat
          label="Shared tokens"
          value={String(f.stats.sharedTokens)}
          hint="stored once"
        />
        <Stat
          label="Preempted, to recompute"
          value={f.preempted.map((r) => `#${r}`).join(" ") || "–"}
        />
      </div>
      <p className="mt-2 min-h-5 text-sm" role="status">
        {lastEvent ? `Latest event: ${lastEvent}` : ""}
      </p>
    </WidgetFrame>
  );
}
