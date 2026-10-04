/**
 * MDX components map (as in transformer-explainer). The interactives come
 * through `lazy.tsx`, so each chapter loads only its own widgets' code.
 */
import type { MDXRemoteProps } from "next-mdx-remote/rsc";

import { Layer } from "@/components/interactive/Layer";
import {
  BatchingWidget,
  DisaggSimulatorWidget,
  FlashAttentionWidget,
  GenerationLoopWidget,
  InterferenceWidget,
  KvCacheWidget,
  KvCalculatorWidget,
  KvHandoffWidget,
  MetricsWidget,
  PagingWidget,
  ParallelismWidget,
  QuantisationWidget,
  RooflineWidget,
  SpeculativeWidget,
  TradeoffWidget,
} from "@/components/interactive/lazy";
import { Callout } from "@/components/ui/Callout";
import { MdxTable } from "@/components/ui/MdxTable";

export const mdxComponents: NonNullable<MDXRemoteProps["components"]> = {
  table: MdxTable,
  Layer,
  Callout,
  GenerationLoopWidget,
  KvCacheWidget,
  KvCalculatorWidget,
  RooflineWidget,
  BatchingWidget,
  PagingWidget,
  FlashAttentionWidget,
  SpeculativeWidget,
  QuantisationWidget,
  ParallelismWidget,
  MetricsWidget,
  InterferenceWidget,
  KvHandoffWidget,
  DisaggSimulatorWidget,
  TradeoffWidget,
};
