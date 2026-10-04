"use client";

/**
 * Code-split wrappers for the interactives. Each widget becomes its own
 * chunk, loaded only on the chapter that uses it, so no chapter ships the
 * others' code. They still render on the server (all their computation is
 * deterministic), so the page doesn't jump when the JavaScript arrives.
 */
import dynamic from "next/dynamic";

function Loading(): JSX.Element {
  return (
    <div className="my-8 h-48 animate-pulse rounded-lg border border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900" />
  );
}

export const GenerationLoopWidget = dynamic(
  () => import("./GenerationLoopWidget").then((m) => m.GenerationLoopWidget),
  { loading: Loading },
);
export const KvCacheWidget = dynamic(
  () => import("./KvCacheWidget").then((m) => m.KvCacheWidget),
  { loading: Loading },
);
export const KvCalculatorWidget = dynamic(
  () => import("./KvCalculatorWidget").then((m) => m.KvCalculatorWidget),
  { loading: Loading },
);
export const RooflineWidget = dynamic(
  () => import("./RooflineWidget").then((m) => m.RooflineWidget),
  { loading: Loading },
);
export const BatchingWidget = dynamic(
  () => import("./BatchingWidget").then((m) => m.BatchingWidget),
  { loading: Loading },
);
export const PagingWidget = dynamic(
  () => import("./PagingWidget").then((m) => m.PagingWidget),
  { loading: Loading },
);
export const FlashAttentionWidget = dynamic(
  () => import("./FlashAttentionWidget").then((m) => m.FlashAttentionWidget),
  { loading: Loading },
);
export const SpeculativeWidget = dynamic(
  () => import("./SpeculativeWidget").then((m) => m.SpeculativeWidget),
  { loading: Loading },
);
export const QuantisationWidget = dynamic(
  () => import("./QuantisationWidget").then((m) => m.QuantisationWidget),
  { loading: Loading },
);
export const ParallelismWidget = dynamic(
  () => import("./ParallelismWidget").then((m) => m.ParallelismWidget),
  { loading: Loading },
);
export const MetricsWidget = dynamic(
  () => import("./MetricsWidget").then((m) => m.MetricsWidget),
  { loading: Loading },
);
