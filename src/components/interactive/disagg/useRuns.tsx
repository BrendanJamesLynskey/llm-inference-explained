"use client";

/**
 * Load a recorded workload and run engine configurations on it, off the
 * render path: the runs start after the browser has painted, and a newer
 * request cancels an older one. Shared by the chapter 11–14 interactives.
 */
import { useEffect, useState } from "react";

import {
  engine,
  metricsOf,
  type Metrics,
  type SimConfig,
  type SimResult,
} from "@/lib/disagg/engine";
import { loadWorkload, type WorkloadRate } from "@/lib/disagg/workloads";

export type Run = { metrics: Metrics; result: SimResult };

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
      .then(
        (rows) =>
          new Promise<Record<K, Run>>((resolve, reject) =>
            setTimeout(() => {
              try {
                const out = {} as Record<K, Run>;
                for (const k of Object.keys(cfgs) as K[]) {
                  const result = engine.simulate(cfgs[k], rows);
                  out[k] = { result, metrics: metricsOf(result) };
                }
                resolve(out);
              } catch (e) {
                reject(e);
              }
            }, 0),
          ),
      )
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
      className="mt-4 animate-pulse text-sm text-neutral-600 dark:text-neutral-400"
    >
      Running the simulator…
    </p>
  );
}
