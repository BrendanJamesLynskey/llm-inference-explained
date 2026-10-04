"use client";

/**
 * Load a recorded workload and run engine configurations on it in a Web
 * Worker (off the main thread; inline as a fallback). A newer request
 * supersedes an older one. Shared by the chapter 11–14 interactives.
 */
import { useEffect, useState } from "react";

import {
  runPlain,
  type PlainRun,
  type Row,
  type SimConfig,
} from "@/lib/disagg/engine";
import { loadWorkload, type WorkloadRate } from "@/lib/disagg/workloads";

export type Run = PlainRun;

// One shared worker runs every simulation off the main thread, in order.
let worker: Worker | null | undefined;
let nextId = 0;
const waiting = new Map<
  number,
  { resolve: (r: Record<string, Run>) => void; reject: (e: Error) => void }
>();

function getWorker(): Worker | null {
  if (worker !== undefined) return worker;
  try {
    worker = new Worker(new URL("./sim.worker.ts", import.meta.url));
    worker.onmessage = (
      e: MessageEvent<{
        id: number;
        runs?: Record<string, Run>;
        error?: string;
      }>,
    ) => {
      const w = waiting.get(e.data.id);
      if (!w) return;
      waiting.delete(e.data.id);
      if (e.data.runs) w.resolve(e.data.runs);
      else w.reject(new Error(e.data.error ?? "simulation failed"));
    };
  } catch {
    worker = null; // no workers: run on the main thread below
  }
  return worker;
}

function simulateAll<K extends string>(
  rows: Row[],
  cfgs: Record<K, SimConfig>,
): Promise<Record<K, Run>> {
  const w = getWorker();
  if (w) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      waiting.set(id, {
        resolve: resolve as (r: Record<string, Run>) => void,
        reject,
      });
      w.postMessage({ id, rows, cfgs });
    });
  }
  return new Promise((resolve, reject) =>
    setTimeout(() => {
      try {
        const out = {} as Record<K, Run>;
        for (const k of Object.keys(cfgs) as K[])
          out[k] = runPlain(cfgs[k], rows);
        resolve(out);
      } catch (e) {
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    }, 0),
  );
}

export type RunsState<K extends string> =
  | { status: "running" }
  | { status: "error"; message: string }
  | { status: "ready"; runs: Record<K, Run> };

export function useRuns<K extends string>(
  rate: WorkloadRate,
  cfgs: Record<K, SimConfig>,
): RunsState<K> {
  const [state, setState] = useState<RunsState<K>>({ status: "running" });
  const key = `${rate}|${JSON.stringify(cfgs)}`;
  useEffect(() => {
    let cancelled = false;
    setState({ status: "running" });
    loadWorkload(rate)
      .then((rows) => simulateAll(rows, cfgs))
      .then((runs) => {
        if (!cancelled) setState({ status: "ready", runs });
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setState({
            status: "error",
            message: e instanceof Error ? e.message : String(e),
          });
      });
    return () => {
      cancelled = true;
    };
    // `key` captures rate and every configuration value
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}

/** A placeholder while the engine runs (or the error, if it failed). */
export function RunStatus({
  state,
}: {
  state: { status: "running" } | { status: "error"; message: string };
}): JSX.Element {
  return state.status === "error" ? (
    <p role="alert" className="mt-4 text-sm text-rose-700 dark:text-rose-400">
      The simulation failed: {state.message}
    </p>
  ) : (
    <p
      role="status"
      className="mt-4 text-sm text-neutral-600 dark:text-neutral-400"
    >
      Running the simulator…
    </p>
  );
}
