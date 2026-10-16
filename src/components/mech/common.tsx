"use client";

/**
 * Shared pieces of the chapters' mechanism animations: loading a recorded
 * scenario (fetched after the page, so pages stay light), the variant
 * picker, the colour of each request, and the loading placeholder.
 */
import { useEffect, useState } from "react";

import { OKABE_ITO } from "@/lib/viz/palette";
import type { MFile, Scenario } from "@/lib/tradeoffs/mech";
import { mechUrl } from "@/lib/tradeoffs/mech";

/** Fetch one recorded scenario (or the cost-model tables). */
export function useJson<T>(url: string): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    let live = true;
    fetch(url)
      .then((r) => r.json() as Promise<T>)
      .then((d) => {
        if (live) setData(d);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [url]);
  return data;
}

export function useMech(s: Scenario): MFile | null {
  return useJson<MFile>(mechUrl(s));
}

export function Loading({ what }: { what: string }): JSX.Element {
  return (
    <p
      data-pending-widget
      className="my-8 min-h-96 text-sm text-neutral-600 dark:text-neutral-400"
    >
      Loading the {what} (recorded by the simulator)…
    </p>
  );
}

/**
 * One colour per request, cycling through the family palette (yellow is
 * drawn with a dark outline wherever it is used).
 */
const CYCLE = [
  OKABE_ITO.blue,
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.purple,
  OKABE_ITO.sky,
  OKABE_ITO.vermillion,
  OKABE_ITO.yellow,
];
export const reqColour = (rid: number): string =>
  CYCLE[((rid % CYCLE.length) + CYCLE.length) % CYCLE.length]!;

/** The hatch pattern for "waiting" or "reserved but empty" (never colour alone). */
export function Hatch({
  id,
  colour = "#a3a3a3",
}: {
  id: string;
  colour?: string;
}): JSX.Element {
  return (
    <pattern
      id={id}
      width="6"
      height="6"
      patternUnits="userSpaceOnUse"
      patternTransform="rotate(45)"
    >
      <rect width="6" height="6" fill="none" />
      <line x1="0" y1="0" x2="0" y2="6" stroke={colour} strokeWidth="2" />
    </pattern>
  );
}

/** "Recorded by the simulator" line under a widget, with its commit. */
export function Provenance({ f }: { f: MFile }): JSX.Element {
  return (
    <p className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
      Every frame is a state Disaggregated_Inference_Sim recorded (commit{" "}
      <span className="font-mono">{f.commit.slice(0, 7)}</span>, scenario{" "}
      <span className="font-mono">{f.scenario}</span>); the site&apos;s own
      engine reproduces its timestamps bit for bit.
    </p>
  );
}
