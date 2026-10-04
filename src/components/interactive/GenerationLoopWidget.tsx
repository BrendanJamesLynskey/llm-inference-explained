"use client";

/**
 * The generation loop, live: prefill the prompt in one pass, then decode one
 * token per step until a stop condition. Runs the vendored transformer
 * (untrained, seed 42) with the KV cache in the browser.
 */
import { useMemo, useRef, useState } from "react";

import { ActionButton, Segmented, Stat } from "@/components/ui/Controls";
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
  prefill,
  type FlopCounter,
  type KVCache,
} from "@/lib/transformer/kvcache";
import { mulberry32 } from "@/lib/transformer/random";
import { sample, type SampleMode } from "@/lib/transformer/sampling";
import { softmax } from "@/lib/transformer/softmax";

type Mode = "greedy" | "temperature";
type Tok = { id: number; phase: "prompt" | "generated" };
type Step = {
  n: number;
  phase: "prefill" | "decode";
  tokensIn: number;
  cached: number;
  flops: number;
  emitted: string;
};

const STOP_ID = toyEncode(".")[0]!;

export function GenerationLoopWidget(): JSX.Element {
  const [prompt, setPrompt] = useState("the cat sat");
  const [mode, setMode] = useState<Mode>("temperature");
  const [toks, setToks] = useState<Tok[]>([]);
  const [steps, setSteps] = useState<Step[]>([]);
  const [logits, setLogits] = useState<number[] | null>(null);
  const [stopped, setStopped] = useState<string | null>(null);
  const cache = useRef<KVCache | null>(null);
  const rng = useRef(mulberry32(7));

  const sampleMode: SampleMode =
    mode === "greedy"
      ? { kind: "greedy" }
      : { kind: "temperature", temperature: 0.8 };

  function reset(): void {
    cache.current = null;
    rng.current = mulberry32(7);
    setToks([]);
    setSteps([]);
    setLogits(null);
    setStopped(null);
  }

  /** Pick the next token from `l`, record the step, check stop conditions. */
  function emit(l: number[], step: Omit<Step, "emitted">, all: Tok[]): void {
    const id = sample(l, sampleMode, rng.current);
    const next = [...all, { id, phase: "generated" as const }];
    setToks(next);
    setLogits(l);
    setSteps((s) => [...s, { ...step, emitted: toyLabel(id) }]);
    if (id === STOP_ID) setStopped("stop token '.' generated");
    else if (next.length >= TOY_MAX_LEN)
      setStopped(`length limit (${TOY_MAX_LEN})`);
  }

  function runPrefill(): void {
    const ids = toyEncode(prompt || "a");
    const c: FlopCounter = { flops: 0 };
    const out = prefill(ids, TOY_CONFIG, toyWeights(), c);
    cache.current = out.cache;
    const all = ids.map((id) => ({ id, phase: "prompt" as const }));
    emit(
      out.logits[ids.length - 1]!,
      {
        n: 1,
        phase: "prefill",
        tokensIn: ids.length,
        cached: out.cache.length,
        flops: c.flops,
      },
      all,
    );
  }

  function runDecode(): void {
    const c = cache.current;
    if (!c || stopped) return;
    const last = toks[toks.length - 1]!;
    const fc: FlopCounter = { flops: 0 };
    const l = decodeStep(last.id, c, TOY_CONFIG, toyWeights(), fc);
    emit(
      l,
      {
        n: steps.length + 1,
        phase: "decode",
        tokensIn: 1,
        cached: c.length,
        flops: fc.flops,
      },
      toks,
    );
  }

  const top = useMemo(() => {
    if (!logits) return [];
    const p = softmax(logits);
    return p
      .map((v, id) => ({ id, v }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 5);
  }, [logits]);

  const started = cache.current !== null;

  return (
    <WidgetFrame
      testId="generation-loop"
      title="Prefill once, then one token per step"
      caption="A real 2-block decoder (d_model 16, untrained random weights, so the text is gibberish) running in your browser. Prefill processes the whole prompt in one pass; every decode step processes one token."
    >
      <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Prompt
          </span>
          <input
            type="text"
            value={prompt}
            maxLength={16}
            disabled={started}
            onChange={(e) => setPrompt(e.target.value)}
            aria-label="Generation loop prompt"
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-mono text-sm disabled:opacity-60 dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
        <Segmented
          label="Sampling"
          value={mode}
          options={[
            { value: "greedy", label: "greedy" },
            { value: "temperature", label: "τ = 0.8" },
          ]}
          onChange={setMode}
        />
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <ActionButton onClick={runPrefill} disabled={started}>
          1 · Prefill the prompt
        </ActionButton>
        <ActionButton
          onClick={runDecode}
          disabled={!started || stopped !== null}
        >
          2 · Decode one token
        </ActionButton>
        <ActionButton variant="secondary" onClick={reset}>
          Reset
        </ActionButton>
      </div>

      <div
        aria-label="Sequence so far"
        className="mt-4 flex min-h-10 flex-wrap gap-1"
      >
        {(toks.length
          ? toks
          : toyEncode(prompt).map((id) => ({ id, phase: "prompt" as const }))
        ).map((t, i) => (
          <span
            key={i}
            className={`inline-flex h-7 min-w-7 items-center justify-center rounded font-mono text-sm ${
              t.phase === "prompt"
                ? "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-100"
                : "bg-indigo-100 text-indigo-900 dark:bg-indigo-900/50 dark:text-indigo-100"
            } ${!started ? "opacity-50" : ""}`}
          >
            {toyLabel(t.id)}
          </span>
        ))}
      </div>
      <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
        <span className="font-medium text-emerald-700 dark:text-emerald-300">
          Green
        </span>
        : prompt (prefill).{" "}
        <span className="font-medium text-indigo-700 dark:text-indigo-300">
          Blue
        </span>
        : generated (decode). Stops on &lsquo;.&rsquo; or at {TOY_MAX_LEN}{" "}
        tokens.
        {stopped && (
          <strong
            role="status"
            className="ml-1 text-neutral-800 dark:text-neutral-200"
          >
            Stopped: {stopped}.
          </strong>
        )}
      </p>

      {top.length > 0 && (
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <p className="text-sm font-semibold">
              Next-token probabilities (last step)
            </p>
            <ol className="mt-2 space-y-1">
              {top.map((r) => (
                <li key={r.id} className="flex items-center gap-2 text-xs">
                  <span className="w-6 font-mono">{toyLabel(r.id)}</span>
                  <span className="min-w-0 flex-1">
                    <span
                      className="block h-2 rounded bg-indigo-500"
                      style={{
                        width: `${Math.max(1, (r.v / top[0]!.v) * 100)}%`,
                      }}
                    />
                  </span>
                  <span className="font-mono text-neutral-600 dark:text-neutral-400">
                    {(r.v * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ol>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Steps" value={String(steps.length)} />
            <Stat
              label="Cached positions"
              value={String(cache.current?.length ?? 0)}
            />
            <Stat
              label="Prefill FLOPs"
              value={(steps[0]?.flops ?? 0).toLocaleString("en-GB")}
            />
            <Stat
              label="Last decode FLOPs"
              value={(steps.length > 1
                ? steps[steps.length - 1]!.flops
                : 0
              ).toLocaleString("en-GB")}
            />
          </div>
        </div>
      )}

      {steps.length > 0 && (
        <div
          className="focus-ring mt-4 max-h-56 overflow-auto rounded"
          role="region"
          tabIndex={0}
          aria-label="Step log"
        >
          <table className="w-full text-left font-mono text-xs">
            <caption className="sr-only">Step log</caption>
            <thead>
              <tr className="text-neutral-500 dark:text-neutral-400">
                <th className="py-1 pr-2 font-normal">step</th>
                <th className="py-1 pr-2 font-normal">phase</th>
                <th className="py-1 pr-2 font-normal">tokens in</th>
                <th className="py-1 pr-2 font-normal">KV after</th>
                <th className="py-1 pr-2 font-normal">FLOPs</th>
                <th className="py-1 font-normal">emitted</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((s) => (
                <tr
                  key={s.n}
                  className="border-t border-neutral-200 dark:border-neutral-800"
                >
                  <td className="py-1 pr-2">{s.n}</td>
                  <td className="py-1 pr-2">{s.phase}</td>
                  <td className="py-1 pr-2">{s.tokensIn}</td>
                  <td className="py-1 pr-2">{s.cached}</td>
                  <td className="py-1 pr-2">
                    {s.flops.toLocaleString("en-GB")}
                  </td>
                  <td className="py-1">{s.emitted}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </WidgetFrame>
  );
}
