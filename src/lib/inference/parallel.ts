/*
 * src/lib/inference/parallel.ts
 *
 * Operation: communication cost per decode step of tensor, pipeline and
 *            expert parallelism, on the links of hardware.py.
 * Shapes:    tpComm / ppComm / epComm(model, n, link, tokens) → { bytes, time }
 * Intuition: splitting a model across devices adds messages. Tensor
 *            parallelism all-reduces activations twice per layer, so it wants
 *            NVLink-class links inside a node; pipeline parallelism sends one
 *            activation per stage boundary but idles stages; expert
 *            parallelism sends each token to the devices holding its experts.
 * MDX:       /learn/09-parallelism.
 *
 * Message patterns after Megatron-LM (arXiv:1909.08053: two all-reduces per
 * layer in the forward pass), GPipe (arXiv:1811.06965: bubble fraction) and
 * GShard (arXiv:2006.16668: all-to-all dispatch and combine). The ring
 * all-reduce cost 2(n−1)/n · S / B is the standard bandwidth term; latency
 * here is the link's per-message latency, 2(n−1) messages. These are
 * first-order estimates, not measurements: real collectives overlap,
 * use trees, and run several links at once.
 */
import type { Link, ModelSpec } from "./costModel";

export type Comm = {
  /** Bytes each device sends per decode step. */
  bytes: number;
  /** Seconds per decode step on the link (not overlapped with compute). */
  time: number;
  /** Messages per device per step. */
  messages: number;
};

const ZERO: Comm = { bytes: 0, time: 0, messages: 0 };

/** Ring all-reduce of `size` bytes over `n` devices. */
export function ringAllReduce(size: number, n: number, link: Link): Comm {
  if (n <= 1) return ZERO;
  const bytes = (2 * (n - 1) * size) / n;
  const messages = 2 * (n - 1);
  return {
    bytes,
    time: bytes / link.bandwidth + messages * link.latency,
    messages,
  };
}

/**
 * Tensor parallelism over `n` devices: two all-reduces of the activations
 * ([tokens, d_model] at `actBytes` bytes) per layer, one after attention
 * and one after the MLP.
 */
export function tpComm(
  m: ModelSpec,
  n: number,
  link: Link,
  tokens: number,
  actBytes = 2,
): Comm {
  if (n <= 1) return ZERO;
  const one = ringAllReduce(tokens * m.d_model * actBytes, n, link);
  const k = 2 * m.n_layers;
  return {
    bytes: k * one.bytes,
    time: k * one.time,
    messages: k * one.messages,
  };
}

/**
 * Pipeline parallelism over `n` stages: each boundary passes the
 * activations [tokens, d_model] once per step. Sequential stages: the
 * per-step time adds every boundary's transfer.
 */
export function ppComm(
  m: ModelSpec,
  n: number,
  link: Link,
  tokens: number,
  actBytes = 2,
): Comm {
  if (n <= 1) return ZERO;
  const size = tokens * m.d_model * actBytes;
  return {
    bytes: size,
    time: (n - 1) * (size / link.bandwidth + link.latency),
    messages: 1,
  };
}

/**
 * GPipe's pipeline bubble: with `n` stages and `micro` micro-batches in
 * flight, the idle fraction is (n − 1) / (micro + n − 1).
 */
export function bubbleFraction(n: number, micro: number): number {
  return (n - 1) / (micro + n - 1);
}

/**
 * Expert parallelism over `n` devices for an MoE layer with top-`k`
 * routing: each token's activation goes to k experts and comes back
 * (dispatch + combine all-to-alls); with experts spread evenly a fraction
 * (n − 1)/n of those copies leave the device. `moeLayers` layers per step.
 */
export function epComm(
  dModel: number,
  moeLayers: number,
  topK: number,
  n: number,
  link: Link,
  tokens: number,
  actBytes = 2,
): Comm {
  if (n <= 1) return ZERO;
  const perLayer = (2 * tokens * topK * dModel * actBytes * (n - 1)) / n;
  // Two all-to-alls per layer; each device exchanges with n − 1 peers.
  const messages = 2 * (n - 1) * moeLayers;
  return {
    bytes: perLayer * moeLayers,
    time: (perLayer * moeLayers) / link.bandwidth + messages * link.latency,
    messages,
  };
}
