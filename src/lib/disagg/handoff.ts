/*
 * src/lib/disagg/handoff.ts
 *
 * Operation: the size and time of a KV-cache hand-off from a prefill pool to
 *            a decode pool, in closed form.
 * Shapes:    handoff({ model, prompt, link, ratio, layerwise, … }) → Handoff
 * Intuition: a disaggregated request must ship its whole KV cache, prompt
 *            × bytes per token, before decoding can start. Time = link
 *            latency + bytes ÷ bandwidth. Streaming it layer by layer while
 *            prefill is still running hides most of that time, as long as
 *            the link keeps up with the prefill.
 * MDX:       /learn/12-moving-the-kv-cache.
 *
 * The bytes, bandwidths and the prefill step are Disaggregated_Inference_Sim's
 * (`hardware.py`: handoff_bytes, LINKS, CostModel.prefill). Layer-wise
 * streaming is this site's extension: the simulator ships the cache after the
 * whole prefill (its README, "Modelling assumptions").
 */
import {
  H100_SXM,
  LINKS,
  costModel,
  kvBytesPerToken,
  type Accelerator,
  type ModelSpec,
} from "@/lib/inference/costModel";

/** Compression of the hand-off: bytes in ÷ bytes out (hardware.KV_PRESETS). */
export const COMPRESSION = {
  none: 1,
  fp8: 2,
  /** 4-bit values plus one 8-bit scale per 16: 64/17. */
  "fp4-block": 64 / 17,
} as const;
export type Compression = keyof typeof COMPRESSION;

export type HandoffInput = {
  model: ModelSpec;
  /** Devices per prefill instance (tensor-parallel group). */
  devices: number;
  device?: Accelerator;
  prompt: number;
  link: keyof typeof LINKS;
  compression: Compression;
  /** Stream each layer's KV as soon as that layer's prefill finishes. */
  layerwise: boolean;
};

export type Handoff = {
  /** KV bytes of the prompt (BF16). */
  bytes: number;
  /** Bytes on the wire after compression. */
  wireBytes: number;
  prefillTime: number;
  /** Latency + wire bytes ÷ bandwidth: the whole transfer on its own. */
  transferTime: number;
  /** Transfer time left after the prefill ends (all of it, unless streamed). */
  exposedTime: number;
  /** Requests per second at which this hand-off alone fills the link. */
  linkSaturationRate: number;
};

export function handoff(x: HandoffInput): Handoff {
  const link = LINKS[x.link]!;
  const bytes = x.prompt * kvBytesPerToken(x.model);
  const wireBytes = bytes / COMPRESSION[x.compression];
  const wire = wireBytes / link.bandwidth;
  const transferTime = link.latency + wire;
  const prefillTime = costModel(x.model, x.device ?? H100_SXM, {
    nDevices: x.devices,
  }).prefill([x.prompt]).time;
  const L = x.model.n_layers;
  // Layer l's KV leaves when layer l's prefill finishes (every layer takes
  // prefill/L) and the link is free; each layer's slice takes wire/L. The last
  // slice lands max(prefill + wire/L, prefill/L + wire) after the start.
  const exposedTime = x.layerwise
    ? link.latency + Math.max(wire / L, wire - (prefillTime * (L - 1)) / L)
    : transferTime;
  return {
    bytes,
    wireBytes,
    prefillTime,
    transferTime,
    exposedTime,
    linkSaturationRate: link.bandwidth / wireBytes,
  };
}
