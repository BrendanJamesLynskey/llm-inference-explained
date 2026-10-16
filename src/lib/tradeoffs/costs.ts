/**
 * Chapters 8 and 9's animations call the vendored engine's own cost model
 * live, so a parameter change re-runs the model: the step times of tensor-
 * and pipeline-parallel instances, and of each storage and compute format.
 * The unit tests check every number here against the simulator's Python
 * cost model (public/tradeoffs/mechanisms/cost_model.json, written by
 * scripts/mechanisms_reference.py), bit for bit.
 */
import { engine, type StepCost } from "./engine";

/** The model every sweep point serves. */
export const MODEL = "llama3-70b";
const OVERHEAD = 0.5e-3;

function cm(
  device: string,
  n: number,
  opts: Record<string, unknown>,
  formats?: { weights: string; kv: string },
) {
  const base = engine.MODELS[MODEL]!;
  const m = formats
    ? engine.derive({
        ...base,
        wb: engine.QUANT_FORMATS[formats.weights],
        kb: engine.QUANT_FORMATS[formats.kv],
      })
    : engine.derive(base);
  return engine.costModel(
    m,
    engine.deviceFor(device, {}),
    n,
    OVERHEAD,
    null,
    false,
    undefined,
    false,
    opts,
  );
}

export type TpSteps = {
  tp: number;
  decodeB1: StepCost;
  decodeB64: StepCost;
  prefill8192: StepCost;
  kvTokens: number;
};

/** Llama-3-70B on `tp` H100s with tensor parallelism (all-reduces priced). */
export function tpSteps(tp: number): TpSteps {
  const c = cm("h100", tp, { parallel: { tp, pp: 1 } });
  return {
    tp,
    decodeB1: c.decodeSum(2048, 1),
    decodeB64: c.decodeSum(64 * 2048, 64),
    prefill8192: c.prefill([8192]),
    kvTokens: c.kvCap ?? 0,
  };
}

/** A pipelined instance: prefill of four 4,096-token prompts and a 64-row decode step. */
export function ppSteps(
  tp: number,
  pp: number,
  mb: number | null,
): { prefill: StepCost; decode: StepCost } {
  const par: Record<string, number> = { tp, pp };
  if (mb !== null) par.microbatches = mb;
  const c = cm("h100", tp * pp, { parallel: par });
  return {
    prefill: c.prefill([4096, 4096, 4096, 4096]),
    decode: c.decodeSum(64 * 2048, 64),
  };
}

export type FormatSteps = {
  /** KV bytes per token (all layers), in the KV format. */
  kvTok: number;
  decodeB1: StepCost;
  decodeB64: StepCost;
  prefill8192: StepCost;
  kvTokens: number;
  speedup: number;
};

/** Four GPUs, no all-reduce (to isolate the format), weights / KV / matmul formats. */
export function formatSteps(
  device: string,
  weights: string,
  kv: string,
  compute: string,
): FormatSteps {
  const speedup =
    compute === "bf16" ? 1.0 : engine.DEVICES[device]!.native[compute]!;
  const c = cm(device, 4, compute === "bf16" ? {} : { speedup }, {
    weights,
    kv,
  });
  const m = engine.derive({
    ...engine.MODELS[MODEL]!,
    wb: engine.QUANT_FORMATS[weights],
    kb: engine.QUANT_FORMATS[kv],
  });
  return {
    kvTok: Number(m.kvTok),
    decodeB1: c.decodeSum(2048, 1),
    decodeB64: c.decodeSum(64 * 2048, 64),
    prefill8192: c.prefill([8192]),
    kvTokens: c.kvCap ?? 0,
    speedup,
  };
}

/** Bytes per stored value of a format (block scales included). */
export function bytesPer(format: string): number {
  return engine.QUANT_FORMATS[format]!;
}

/** Whether a device has matmul units for a format (else weight-only). */
export function native(device: string, format: string): boolean {
  return format === "bf16" || format in engine.DEVICES[device]!.native;
}
