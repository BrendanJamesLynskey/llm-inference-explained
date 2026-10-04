"use client";

/**
 * The KV cache filling up, step by step, next to proof that it is exact:
 * every step also recomputes the whole sequence without a cache and
 * compares the two logit vectors element by element.
 */
import { useRef, useState } from "react";

import { ActionButton, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  TOY_CONFIG,
  TOY_MAX_LEN,
  toyEncode,
  toyLabel,
  toyWeights,
} from "@/lib/inference/toyModel";
import {
  decodeStep,
  decodeStepFlops,
  prefill,
  prefillFlops,
  type KVCache,
} from "@/lib/transformer/kvcache";
import { forwardTyped } from "@/lib/transformer/model";
import { argmax } from "@/lib/transformer/sampling";

type Row = { pos: number; cached: number; uncached: number; maxDiff: number };

const PROMPT = "the kv";

function maxAbsDiff(a: number[], b: number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

export function KvCacheWidget(): JSX.Element {
  const [ids, setIds] = useState<number[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [promptLen, setPromptLen] = useState(0);
  const cache = useRef<KVCache | null>(null);

  function step(): void {
    const w = toyWeights();
    if (!cache.current) {
      const p = toyEncode(PROMPT);
      const out = prefill(p, TOY_CONFIG, w);
      cache.current = out.cache;
      const ref = forwardTyped(
        p,
        { ...TOY_CONFIG, seq_len: p.length },
        w,
      ).logits;
      const l = out.logits[p.length - 1]!;
      setPromptLen(p.length);
      setIds([...p, argmax(l)]);
      setRows([
        {
          pos: p.length,
          cached: prefillFlops(p.length, TOY_CONFIG),
          uncached: prefillFlops(p.length, TOY_CONFIG),
          maxDiff: maxAbsDiff(l, ref[p.length - 1]!),
        },
      ]);
      return;
    }
    if (ids.length >= TOY_MAX_LEN) return;
    const c = cache.current;
    const ctx = c.length;
    const l = decodeStep(ids[ids.length - 1]!, c, TOY_CONFIG, w);
    // The uncached alternative: run the whole sequence again.
    const ref = forwardTyped(ids, { ...TOY_CONFIG, seq_len: ids.length }, w)
      .logits[ids.length - 1]!;
    setIds([...ids, argmax(l)]);
    setRows((r) => [
      ...r,
      {
        pos: ids.length,
        cached: decodeStepFlops(ctx, TOY_CONFIG),
        uncached: prefillFlops(ids.length, TOY_CONFIG),
        maxDiff: maxAbsDiff(l, ref),
      },
    ]);
  }

  function reset(): void {
    cache.current = null;
    setIds([]);
    setRows([]);
  }

  const cachedLen = cache.current?.length ?? 0;
  const totC = rows.reduce((a, r) => a + r.cached, 0);
  const totU = rows.reduce((a, r) => a + r.uncached, 0);
  const worst = rows.reduce((a, r) => Math.max(a, r.maxDiff), 0);
  const last = rows[rows.length - 1];
  const maxBar = Math.max(1, ...rows.map((r) => r.uncached));
  const layers = Array.from({ length: TOY_CONFIG.n_blocks }, (_, b) => b);

  return (
    <WidgetFrame
      testId="kv-cache"
      title="Watch the cache fill, and check it against recomputation"
      caption="Each click runs one step of the same untrained toy model with greedy decoding. The cache gains one column per layer; a full uncached forward pass runs alongside and the logits are compared exactly."
    >
      <div className="flex flex-wrap gap-2">
        <ActionButton onClick={step} disabled={ids.length >= TOY_MAX_LEN}>
          {cache.current ? "Decode one token" : `Prefill “${PROMPT}”`}
        </ActionButton>
        <ActionButton variant="secondary" onClick={reset}>
          Reset
        </ActionButton>
      </div>

      <div
        className="focus-ring mt-4 overflow-x-auto rounded"
        role="region"
        tabIndex={0}
        aria-label="KV cache contents"
      >
        <table className="border-separate border-spacing-0.5 font-mono text-[0.65rem]">
          <caption className="sr-only">
            KV cache: one row per layer and K or V, one column per cached
            position
          </caption>
          <thead>
            <tr>
              <th className="pr-2 text-left font-normal text-neutral-500 dark:text-neutral-400">
                pos
              </th>
              {Array.from({ length: TOY_MAX_LEN }, (_, i) => (
                <th
                  key={i}
                  className="w-4 font-normal text-neutral-500 dark:text-neutral-400"
                >
                  {i < ids.length ? toyLabel(ids[i]!) : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {layers.flatMap((b) =>
              (["K", "V"] as const).map((kv) => (
                <tr key={`${b}${kv}`}>
                  <th className="whitespace-nowrap pr-2 text-left font-normal text-neutral-600 dark:text-neutral-400">
                    L{b} {kv}
                  </th>
                  {Array.from({ length: TOY_MAX_LEN }, (_, i) => {
                    const filled = i < cachedLen;
                    const fresh = filled && i === cachedLen - 1;
                    const prompt = i < promptLen;
                    return (
                      <td
                        key={i}
                        className={`size-4 rounded-sm ${
                          !filled
                            ? "bg-neutral-200 dark:bg-neutral-800"
                            : fresh
                              ? "bg-amber-400"
                              : prompt
                                ? "bg-emerald-400 dark:bg-emerald-600"
                                : "bg-indigo-400 dark:bg-indigo-500"
                        }`}
                      />
                    );
                  })}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
        Green: written by prefill. Blue: written by decode steps. Amber: the row
        written this step. Grey: empty.
      </p>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Cached positions" value={String(cachedLen)} />
        <Stat
          label="This step: cached"
          value={last ? last.cached.toLocaleString("en-GB") : "0"}
          hint="FLOPs"
        />
        <Stat
          label="This step: uncached"
          value={last ? last.uncached.toLocaleString("en-GB") : "0"}
          hint="FLOPs"
        />
        <Stat
          label="Max |Δ logit|"
          value={rows.length ? worst.toExponential(1) : "–"}
          hint={
            rows.length
              ? worst === 0
                ? "bit-identical"
                : "differs"
              : undefined
          }
        />
      </div>
      {rows.length > 1 && (
        <p className="mt-2 text-sm" role="status">
          So far the cache has done{" "}
          <strong>{((totC / totU) * 100).toFixed(1)}%</strong> of the FLOPs that
          recomputing every step would have done.
        </p>
      )}

      {rows.length > 0 && (
        <div className="mt-3 space-y-1" aria-label="FLOPs per step">
          {rows.map((r) => (
            <div key={r.pos} className="flex items-center gap-2 text-[0.65rem]">
              <span className="w-8 shrink-0 font-mono text-neutral-500 dark:text-neutral-400">
                #{r.pos}
              </span>
              <span className="min-w-0 flex-1">
                <span
                  className="block h-1.5 rounded bg-neutral-400 dark:bg-neutral-600"
                  style={{ width: `${(r.uncached / maxBar) * 100}%` }}
                  title="uncached"
                />
                <span
                  className="mt-0.5 block h-1.5 rounded bg-indigo-500"
                  style={{ width: `${(r.cached / maxBar) * 100}%` }}
                  title="cached"
                />
              </span>
            </div>
          ))}
          <p className="text-xs text-neutral-600 dark:text-neutral-400">
            Grey: FLOPs to recompute the sequence. Blue: FLOPs with the cache.
          </p>
        </div>
      )}
    </WidgetFrame>
  );
}
