/*
 * src/lib/inference/costModel.ts
 *
 * Operation: the roofline step-time model of Disaggregated_Inference_Sim
 *            (`src/disagg_sim/hardware.py`), transformer path only, ported
 *            to TypeScript with the same float-operation order.
 * Shapes:    prefill([prompt lengths]) / decode(totalContext, batch) →
 *            { flops, bytes, time, bound, intensity }
 * Intuition: a step takes max(FLOPs / FLOP rate, bytes / byte rate) plus a
 *            fixed overhead. Decode reads every weight to make one token per
 *            sequence, so it is memory-bound until the batch is large.
 * MDX:       /learn/03-roofline, /learn/08-quantisation, /learn/04-batching.
 *
 * Every coefficient below is copied from hardware.py at the commit recorded
 * in tests/unit/fixtures/cost_model.json; tests/unit/inference/costModel.test.ts
 * checks the port against fixtures written by the Python package and against
 * the published table in its examples/results.md §1.
 */

const GB = 1e9;
const TB = 1e12;

/** A decoder-only transformer, described by its shape (hardware.py `ModelSpec`). */
export type ModelSpec = {
  name: string;
  n_layers: number;
  d_model: number;
  n_heads: number;
  n_kv_heads: number;
  d_ff: number;
  vocab: number;
  /** Bytes per weight (BF16 = 2). */
  weight_bytes: number;
  /** Bytes per cached K or V value (BF16 = 2). */
  kv_bytes: number;
};

/** One device (hardware.py `Accelerator`): datasheet peaks, derated. */
export type Accelerator = {
  name: string;
  /** Dense BF16 FLOP/s. */
  peak_flops: number;
  /** HBM bytes/s. */
  mem_bw: number;
  /** HBM bytes. */
  mem_capacity: number;
  /** Achievable fraction of peak FLOP/s (MFU). */
  flops_eff: number;
  /** Achievable fraction of peak bandwidth (MBU). */
  bw_eff: number;
  /** Board power limit, W. */
  tdp_w: number;
  /** Static power, W (illustrative). */
  idle_w: number;
  /** Dynamic energy per FLOP, pJ (illustrative). */
  pj_per_flop: number;
  /** Dynamic energy per HBM byte, pJ (illustrative). */
  pj_per_byte: number;
};

export const LLAMA3_8B: ModelSpec = {
  name: "Llama-3-8B",
  n_layers: 32,
  d_model: 4096,
  n_heads: 32,
  n_kv_heads: 8,
  d_ff: 14336,
  vocab: 128256,
  weight_bytes: 2.0,
  kv_bytes: 2.0,
};

export const LLAMA3_70B: ModelSpec = {
  name: "Llama-3-70B",
  n_layers: 80,
  d_model: 8192,
  n_heads: 64,
  n_kv_heads: 8,
  d_ff: 28672,
  vocab: 128256,
  weight_bytes: 2.0,
  kv_bytes: 2.0,
};

const DEVICE_DEFAULTS = {
  flops_eff: 0.55,
  bw_eff: 0.8,
  tdp_w: 700.0,
  idle_w: 100.0,
  pj_per_flop: 1.0,
  pj_per_byte: 60.0,
};

export const H100_SXM: Accelerator = {
  ...DEVICE_DEFAULTS,
  name: "H100-SXM",
  peak_flops: 989 * TB,
  mem_bw: 3.35 * TB,
  mem_capacity: 80 * GB,
};

export const A100_SXM: Accelerator = {
  ...DEVICE_DEFAULTS,
  name: "A100-SXM",
  peak_flops: 312 * TB,
  mem_bw: 2.039 * TB,
  mem_capacity: 80 * GB,
  tdp_w: 400.0,
  idle_w: 60.0,
  pj_per_flop: 1.6,
  pj_per_byte: 70.0,
};

export const MODELS = { "llama3-8b": LLAMA3_8B, "llama3-70b": LLAMA3_70B };
export const DEVICES = { h100: H100_SXM, a100: A100_SXM };

// ─── model arithmetic ──────────────────────────────────────────────────

export function headDim(m: ModelSpec): number {
  return Math.floor(m.d_model / m.n_heads);
}

/** Wq, Wo (d×d) + Wk, Wv (d×kv, GQA-narrow) + SwiGLU MLP (3·d·d_ff). */
export function paramsPerLayer(m: ModelSpec): number {
  const d = m.d_model;
  const kv = m.n_kv_heads * headDim(m);
  const attn = 2 * d * d + 2 * d * kv;
  const mlp = 3 * d * m.d_ff;
  return attn + mlp;
}

export function layerParams(m: ModelSpec): number {
  return m.n_layers * paramsPerLayer(m);
}

/** Total parameters: layers + untied input embedding and LM head. */
export function params(m: ModelSpec): number {
  return layerParams(m) + 2 * m.vocab * m.d_model;
}

/** Parameters that take part in a matmul per token (embedding is a lookup). */
export function matmulParams(m: ModelSpec): number {
  return layerParams(m) + m.vocab * m.d_model;
}

/** Bytes the weights occupy in memory. */
export function weightBytesTotal(m: ModelSpec): number {
  return params(m) * m.weight_bytes;
}

/**
 * Weight traffic of one forward pass over `tokens` tokens: every layer and
 * the LM head in full, plus the embedding rows looked up.
 */
export function weightBytesRead(m: ModelSpec, tokens: number): number {
  return (
    matmulParams(m) * m.weight_bytes + tokens * (m.d_model * m.weight_bytes)
  );
}

/** K and V for every layer, for one token: 2·L·kv_heads·head_dim·bytes. */
export function kvBytesPerToken(m: ModelSpec): number {
  return 2 * m.n_layers * m.n_kv_heads * headDim(m) * m.kv_bytes;
}

// ─── roofline timing ───────────────────────────────────────────────────

export type StepCost = {
  flops: number;
  bytes: number;
  /** Seconds, including the fixed per-step overhead. */
  time: number;
  bound: "compute" | "memory" | "power";
  /** FLOP per byte. */
  intensity: number;
};

export type CostModelOptions = {
  nDevices?: number;
  /** Scheduler + kernel-launch time per step, seconds (default 0.5 ms). */
  stepOverhead?: number;
};

function cbrt(x: number): number {
  // hardware.py `_cbrt`: math.copysign(abs(x) ** (1 / 3), x). Kept as a
  // power, not Math.cbrt, so the two languages round the same way.
  return Math.sign(x) * Math.abs(x) ** (1 / 3);
}

/**
 * Roofline timing for one forward pass of a batch on one instance
 * (`n` devices with tensor parallelism, treated as one bigger device; the
 * all-reduce cost is ignored here, see /learn/09-parallelism). The device's
 * TDP is enforced, as in hardware.py's default (`enforce_tdp=True`).
 */
export function costModel(
  model: ModelSpec,
  device: Accelerator,
  opts: CostModelOptions = {},
) {
  const n = opts.nDevices ?? 1;
  const overhead = opts.stepOverhead ?? 0.5e-3;
  const flopsRate = device.peak_flops * device.flops_eff * n;
  const byteRate = device.mem_bw * device.bw_eff * n;
  const jPerFlop = device.pj_per_flop * 1e-12;
  const jPerByte = device.pj_per_byte * 1e-12;
  const budget = (device.tdp_w - device.idle_w) * n;
  const sMin = 0.4;

  /** hardware.py `step_time_raw` with dvfs off and no extra power cap. */
  function stepTimeRaw(
    flops: number,
    nbytes: number,
  ): { t: number; bound: StepCost["bound"] } {
    const tc = flops / flopsRate;
    const tm = nbytes / byteRate;
    let ec = flops * jPerFlop;
    const em = nbytes * jPerByte;
    let s = 1.0;
    let bound: StepCost["bound"] = tc >= tm ? "compute" : "memory";
    if ((ec * s * s + em) / Math.max(tc / s, tm) > budget) {
      bound = "power";
      const x = (budget * tm - em) / ec;
      if (x > 0 && Math.sqrt(x) * tm >= tc) {
        s = Math.min(s, Math.sqrt(x));
      } else {
        const p = em / ec;
        const q = (-budget * tc) / ec;
        const r = Math.sqrt((q * q) / 4 + (p * p * p) / 27);
        s = cbrt(-q / 2 + r) + cbrt(-q / 2 - r);
      }
      s = Math.max(s, sMin);
    }
    let t = Math.max(tc / s, tm);
    ec = ec * s * s;
    if ((ec + em) / t > budget) t = (ec + em) / budget;
    return { t, bound };
  }

  function time(flops: number, nbytes: number): StepCost {
    const { t, bound } = stepTimeRaw(flops, nbytes);
    return {
      flops,
      bytes: nbytes,
      time: t + overhead,
      bound,
      intensity: flops / nbytes,
    };
  }

  return {
    model,
    device,
    nDevices: n,
    flopsRate,
    byteRate,
    /** FLOP/byte where compute and memory time balance (derated peaks). */
    ridgePoint:
      (device.peak_flops * device.flops_eff) / (device.mem_bw * device.bw_eff),

    /** One prefill step over whole prompts (LM head on every prompt token). */
    prefill(promptLens: number[]): StepCost {
      const m = model;
      const tokens = promptLens.reduce((a, b) => a + b, 0);
      let flops = 2 * matmulParams(m) * tokens;
      // causal attention: QKᵀ and AV, each 2·d·c FLOPs at position c.
      let attn = 0;
      for (const s of promptLens)
        attn += 2 * m.n_layers * m.d_model * s * (s + 1);
      flops += attn;
      const nbytes = weightBytesRead(m, tokens) + tokens * kvBytesPerToken(m);
      return time(flops, nbytes);
    },

    /**
     * One decode step: `batch` sequences each make one token. `ctx` is the
     * *total* cached context over the batch; each new token also attends to
     * itself, hence `ctx + batch` positions.
     */
    decode(ctx: number, batch: number): StepCost {
      const m = model;
      const flops =
        2 * matmulParams(m) * batch +
        4 * m.n_layers * m.d_model * (ctx + batch);
      const nbytes =
        weightBytesRead(m, batch) + (ctx + batch) * kvBytesPerToken(m);
      return time(flops, nbytes);
    },

    /**
     * A mixed step (chunked prefill): prompt chunks plus `batch` decoding
     * sequences in one forward pass. **This site's extension** of the
     * simulator's closed form, built from the same terms: a chunk covering
     * prompt positions (a, b] costs its tokens' matmuls plus causal
     * attention 2·L·d·(b(b+1) − a(a+1)), and reads the KV of positions 1..b.
     * With one chunk (0, P] and no decodes it equals `prefill([P])`; with no
     * chunks it equals `decode(ctx, batch)` (both tested).
     */
    mixed(chunks: [number, number][], ctx: number, batch: number): StepCost {
      const m = model;
      let tokens = batch;
      for (const [a, b] of chunks) tokens += b - a;
      let flops = 2 * matmulParams(m) * tokens;
      let kvTokens = 0;
      for (const [a, b] of chunks) {
        flops += 2 * m.n_layers * m.d_model * (b * (b + 1) - a * (a + 1));
        kvTokens += b;
      }
      if (batch > 0) flops += 4 * m.n_layers * m.d_model * (ctx + batch);
      const nbytes =
        weightBytesRead(m, tokens) +
        kvTokens * kvBytesPerToken(m) +
        (batch > 0 ? (ctx + batch) * kvBytesPerToken(m) : 0);
      return time(flops, nbytes);
    },

    /** KV tokens that fit beside the weights at 90% HBM use. */
    kvCapacityTokens(): number {
      const free = device.mem_capacity * n * 0.9 - weightBytesTotal(model);
      if (free <= 0) return 0;
      return Math.floor(free / kvBytesPerToken(model));
    },
  };
}

export type CostModel = ReturnType<typeof costModel>;

/** A link between devices (hardware.py `Link`, from its `LINKS` table). */
export type Link = {
  name: string;
  /** Bytes/s per channel. */
  bandwidth: number;
  /** Seconds per transfer. */
  latency: number;
};

export const LINKS: Record<string, Link> = {
  nvlink4: {
    name: "NVLink 4 (one direction)",
    bandwidth: 450 * GB,
    latency: 5e-6,
  },
  "ib-ndr": { name: "InfiniBand NDR 400G", bandwidth: 50 * GB, latency: 10e-6 },
  pcie5: { name: "PCIe Gen5 x16", bandwidth: 64 * GB, latency: 5e-6 },
  "eth-100g": { name: "100 GbE", bandwidth: 12.5 * GB, latency: 20e-6 },
  "eth-25g": { name: "25 GbE", bandwidth: 3.125 * GB, latency: 20e-6 },
  "cpo-optical": {
    name: "Co-packaged optics (illustrative)",
    bandwidth: 200 * GB,
    latency: 5e-6,
  },
};
