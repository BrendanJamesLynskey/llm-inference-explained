/**
 * The mechanism animations this site borrows from Inference Trade-offs
 * Explained (brief 41), each with its equation, ready for the chapters'
 * MDX. Every frame is a state Disaggregated_Inference_Sim recorded at the
 * Trade-offs site's vendored commit; tests/unit/mech/mech.test.ts checks
 * the recordings against the vendored engine and the frames against the
 * records. Server Components: the equations render with KaTeX on the
 * server, the animations load in the browser.
 */
import {
  BatchEq,
  GpipeEq,
  HandoffEq,
  KvEq,
  RingEq,
  SpecEq,
  TimelineEq,
} from "./equations";
import {
  MechBatching,
  MechDraft,
  MechGpipe,
  MechKv,
  MechPool,
  MechRing,
} from "./lazy";

/** Chapter 4: three batching policies on the same six requests. */
export function BatchingMechanism(): JSX.Element {
  return <MechBatching equation={<BatchEq />} />;
}

/** Chapter 5: reserved against paged KV memory. */
export function PagedKvMechanism(): JSX.Element {
  return <MechKv equation={<KvEq />} />;
}

/** Chapter 7: draft and verify, accepted and rejected tokens. */
export function DraftMechanism(): JSX.Element {
  return <MechDraft equation={<SpecEq />} />;
}

/** Chapter 9: the ring all-reduce of tensor parallelism. */
export function RingMechanism(): JSX.Element {
  return <MechRing equation={<RingEq />} />;
}

/** Chapter 9: a GPipe schedule and its bubble. */
export function PipelineMechanism(): JSX.Element {
  return <MechGpipe equation={<GpipeEq />} />;
}

/** Chapter 11: the same eight GPUs, colocated or split. */
export function PoolsMechanism(): JSX.Element {
  return (
    <MechPool
      scenario="pools"
      initial="colocated"
      testId="mech-pools"
      title="The same eight GPUs, colocated or split"
      summary={
        <>
          Llama-3-70B on eight H100s: two colocated instances of four, or one
          prefill and one decode instance of four, serving the same fourteen
          requests (long prompts, short outputs). Each lane is an instance;
          orange passes are prefills, blue decodes; the KV link carries each
          prompt&apos;s cache from one pool to the other.
        </>
      }
      equation={<TimelineEq />}
      stats={["ttft_p99", "tpot_p99", "itl_p99", "j_per_tok"]}
    />
  );
}

/** Chapter 12: every prompt's KV cache crosses the link. */
export function HandoffMechanism(): JSX.Element {
  return (
    <MechPool
      scenario="handoff"
      initial="eth-25g"
      testId="mech-handoff"
      title="Every prompt's KV cache crosses the link"
      summary={
        <>
          One prefill and one decode H100 (Llama-3-8B), fourteen requests at ten
          per second with long prompts. Watch the KV link lane: on a slow link
          the hand-offs queue, and decode waits for them. Compress the cache on
          the prefill GPU before it leaves.
        </>
      }
      equation={<HandoffEq />}
      stats={["tpot_p99", "itl_p99", "handoff", "horizon"]}
    />
  );
}
