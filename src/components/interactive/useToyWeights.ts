"use client";

/**
 * Fetch the trained toy weights once per page (they are not bundled) and
 * report when they are there. Random init needs nothing.
 */
import { useEffect, useState } from "react";

import {
  TRAINED_WEIGHTS_URL,
  hasTrainedWeights,
  provideTrainedWeights,
  type ToyKind,
  type TrainedWeightsFile,
} from "@/lib/inference/toyModel";

let inflight: Promise<void> | null = null;

function load(): Promise<void> {
  inflight ??= fetch(TRAINED_WEIGHTS_URL)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<TrainedWeightsFile>;
    })
    .then(provideTrainedWeights)
    .catch((e: unknown) => {
      inflight = null; // let a later mount retry
      throw e;
    });
  return inflight;
}

export function useToyWeights(kind: ToyKind): {
  ready: boolean;
  error: string | null;
} {
  const [have, setHave] = useState(hasTrainedWeights());
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (kind !== "trained" || have) return;
    let live = true;
    load()
      .then(() => live && setHave(true))
      .catch(
        (e: unknown) =>
          live && setError(e instanceof Error ? e.message : "failed"),
      );
    return () => {
      live = false;
    };
  }, [kind, have]);
  return { ready: kind === "random" || have, error };
}
