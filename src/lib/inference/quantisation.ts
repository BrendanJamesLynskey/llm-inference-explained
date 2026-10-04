/*
 * src/lib/inference/quantisation.ts
 *
 * Operation: what weight and KV-cache formats do to the bytes a decode step
 *            reads, and so to its time, using the roofline cost model.
 * Shapes:    decodeBreakdown(model, device, n, fmt, ctx, batch) →
 *            { weightBytes, kvBytes, step, tokensPerSecond }
 * Intuition: decode is memory-bound, so its time is roughly bytes / bandwidth.
 *            Halving the bytes per weight nearly halves the step, as long as
 *            the step stays memory-bound and the kernels dequantise for free.
 * MDX:       /learn/08-quantisation.
 *
 * The cost model keeps the BF16 FLOP rate: weight-only formats dequantise to
 * BF16 before the multiply. W8A8 on FP8/INT8 tensor cores would also raise
 * the compute roof; that is not modelled here.
 */
import {
  costModel,
  kvBytesPerToken,
  weightBytesTotal,
  type Accelerator,
  type ModelSpec,
} from "./costModel";

export type Format = {
  key: string;
  label: string;
  /** Bytes per value. */
  bytes: number;
};

export const WEIGHT_FORMATS: Format[] = [
  { key: "bf16", label: "BF16 (16-bit)", bytes: 2 },
  { key: "int8", label: "INT8 / FP8 (8-bit)", bytes: 1 },
  { key: "int4", label: "INT4 (4-bit)", bytes: 0.5 },
];

export const KV_FORMATS: Format[] = [
  { key: "bf16", label: "BF16 (16-bit)", bytes: 2 },
  { key: "fp8", label: "FP8 / INT8 (8-bit)", bytes: 1 },
  { key: "int4", label: "INT4 (4-bit)", bytes: 0.5 },
];

/** A model with its weights and KV cache stored in the given formats. */
export function withFormats(
  m: ModelSpec,
  weightBytes: number,
  kvBytes: number,
): ModelSpec {
  return { ...m, weight_bytes: weightBytes, kv_bytes: kvBytes };
}

export type DecodeBreakdown = {
  /** Weight bytes one decode step reads (layers + LM head + looked-up rows). */
  weightBytes: number;
  /** KV-cache bytes one decode step reads and writes. */
  kvBytes: number;
  time: number;
  bound: string;
  /** Output tokens per second over the whole batch. */
  tokensPerSecond: number;
  /** Bytes the weights occupy. */
  residentWeightBytes: number;
};

/** One decode step of `batch` sequences, each with `ctx` cached tokens. */
export function decodeBreakdown(
  m: ModelSpec,
  device: Accelerator,
  nDevices: number,
  ctxPerSeq: number,
  batch: number,
): DecodeBreakdown {
  const cm = costModel(m, device, { nDevices });
  const s = cm.decode(ctxPerSeq * batch, batch);
  const kv = (ctxPerSeq * batch + batch) * kvBytesPerToken(m);
  return {
    weightBytes: s.bytes - kv,
    kvBytes: kv,
    time: s.time,
    bound: s.bound,
    tokensPerSecond: batch / s.time,
    residentWeightBytes: weightBytesTotal(m),
  };
}
