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
import {
  BatchingMechanism,
  DraftMechanism,
  HandoffMechanism,
  PagedKvMechanism,
  PipelineMechanism,
  PoolsMechanism,
  RingMechanism,
} from "@/components/mech/Mechanisms";
import {
  FlashTiles,
  GenerationTimeline,
  KvFill,
  RooflineClimb,
} from "@/components/heroes/Heroes";
import { CheckYourself } from "@/components/learn/CheckYourself";
import { Keep, Remember } from "@/components/learn/Remember";
import { Check } from "@/components/mdx/Check";
import { V } from "@/components/mdx/V";
import { Callout } from "@/components/ui/Callout";
import { MdxTable } from "@/components/ui/MdxTable";

export const mdxComponents: NonNullable<MDXRemoteProps["components"]> = {
  table: MdxTable,
  Layer,
  Callout,
  CheckYourself,
  Check,
  Remember,
  Keep,
  V,
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
  // brief 41: the Trade-offs site's mechanism animations
  BatchingMechanism,
  PagedKvMechanism,
  DraftMechanism,
  RingMechanism,
  PipelineMechanism,
  PoolsMechanism,
  HandoffMechanism,
  // brief 41: the chapter heroes
  GenerationTimeline,
  KvFill,
  RooflineClimb,
  FlashTiles,
};
