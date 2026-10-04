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
export const KvHandoffWidget = dynamic(
  () => import("./KvHandoffWidget").then((m) => m.KvHandoffWidget),
  { loading: Loading },
);

// The disaggregation interactives run Disaggregated_Inference_Sim's engine on
// workloads fetched at run time, so they render in the browser only: the
// engine (and each workload) loads with the chapter that needs it. Their
// server-rendered placeholder is marked, so `pnpm smoke` can find it.
function Pending({ name }: { name: string }): JSX.Element {
  return (
    <div
      data-pending-widget={name}
      className="my-8 flex h-48 items-center justify-center rounded-lg border border-neutral-200 bg-neutral-50 text-sm text-neutral-600 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-400"
    >
      Loading the simulator…
    </div>
  );
}
export const InterferenceWidget = dynamic(
  () => import("./InterferenceWidget").then((m) => m.InterferenceWidget),
  { loading: () => <Pending name="interference" />, ssr: false },
);
export const DisaggSimulatorWidget = dynamic(
  () => import("./DisaggSimulatorWidget").then((m) => m.DisaggSimulatorWidget),
  { loading: () => <Pending name="disagg-sim" />, ssr: false },
);
export const TradeoffWidget = dynamic(
  () => import("./TradeoffWidget").then((m) => m.TradeoffWidget),
  { loading: () => <Pending name="tradeoffs" />, ssr: false },
);
