"use client";

/**
 * Code-split loaders for the chapter heroes (brief 41). They render on the
 * server too (their models are deterministic), so nothing jumps when the
 * JavaScript arrives; the KV hero waits for its fetched weights.
 */
import dynamic from "next/dynamic";

function Loading(): JSX.Element {
  return (
    <div className="my-8 h-96 animate-pulse rounded-lg border border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900" />
  );
}

export const LazyGenerationTimeline = dynamic(
  () => import("./GenerationTimelineHero"),
  { loading: Loading },
);
export const LazyKvFill = dynamic(() => import("./KvFillHero"), {
  ssr: false,
  loading: Loading,
});
export const LazyRooflineClimb = dynamic(() => import("./RooflineClimbHero"), {
  loading: Loading,
});
export const LazyFlashTiles = dynamic(() => import("./FlashTilesHero"), {
  loading: Loading,
});
