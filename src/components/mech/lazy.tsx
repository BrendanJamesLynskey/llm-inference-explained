"use client";

/**
 * Code-split loaders for the mechanism animations vendored from Inference
 * Trade-offs Explained (brief 41; see src/lib/tradeoffs/VENDORED.json).
 * Each loads its own chunk after the page, then fetches its recorded
 * scenario, so a chapter pays only for the animation it shows.
 */
import dynamic from "next/dynamic";

function Loading(): JSX.Element {
  return (
    <p
      data-pending-widget
      className="my-8 min-h-96 text-sm text-neutral-600 dark:text-neutral-400"
    >
      Loading the animation (recorded by the simulator)…
    </p>
  );
}

export const MechBatching = dynamic(() => import("./BatchingWidget"), {
  ssr: false,
  loading: Loading,
});
export const MechKv = dynamic(() => import("./KvWidget"), {
  ssr: false,
  loading: Loading,
});
export const MechPool = dynamic(() => import("./PoolWidget"), {
  ssr: false,
  loading: Loading,
});
export const MechRing = dynamic(() => import("./RingWidget"), {
  ssr: false,
  loading: Loading,
});
export const MechGpipe = dynamic(() => import("./GpipeWidget"), {
  ssr: false,
  loading: Loading,
});
export const MechDraft = dynamic(() => import("./DraftWidget"), {
  ssr: false,
  loading: Loading,
});
