/*
 * src/lib/disagg/simulator.ts
 *
 * Operation: turn the live simulator's controls into the engine's two
 *            configurations (colocated and disaggregated, on the same
 *            number of instances), and the presets that reproduce rows of
 *            Disaggregated_Inference_Sim's examples/results.md.
 * MDX:       /learn/13-live-simulator (and the presets in 11, 12 and 14).
 */
import type { Format, Metrics, SimConfig } from "./engine";
import type { WorkloadRate } from "./workloads";

export const SIM_MODELS = {
  "llama3-70b": { label: "Llama-3-70B", devices: 4 },
  "llama3-8b": { label: "Llama-3-8B", devices: 1 },
  /** FOptInf's speculative FFT-mixing variant, last-token LM head. */
  "llama3-8b-hyena-circ": {
    label: "Hyena-2 + circulant (speculative)",
    devices: 1,
  },
} as const;
export type SimModel = keyof typeof SIM_MODELS;

/** Prefill devices: two GPUs and the illustrative optical transform engine. */
export const PREFILL_DEVICES = {
  h100: "H100",
  a100: "A100",
  "optical-default": "Optical FFT (defaults)",
  "optical-optimistic": "Optical FFT (optimistic)",
} as const;
export type PrefillDevice = keyof typeof PREFILL_DEVICES;
export const DECODE_DEVICES = { h100: "H100", a100: "A100" } as const;
export type DecodeDevice = keyof typeof DECODE_DEVICES;

export const SIM_LINKS = [
  "nvlink4",
  "ib-ndr",
  "cpo-optical",
  "eth-100g",
  "eth-25g",
] as const;
export type SimLink = (typeof SIM_LINKS)[number];

export type Controls = {
  model: SimModel;
  prefillDevice: PrefillDevice;
  decodeDevice: DecodeDevice;
  nPrefill: number;
  nDecode: number;
  link: SimLink;
  compression: "none" | "fp8" | "fp4-block";
  compressAt: "transit" | "endpoint";
  rate: WorkloadRate;
  /** Seconds. */
  ttftSlo: number;
  /** Seconds per token. */
  tpotSlo: number;
};

export const DEFAULT_CONTROLS: Controls = {
  model: "llama3-70b",
  prefillDevice: "h100",
  decodeDevice: "h100",
  nPrefill: 1,
  nDecode: 1,
  link: "ib-ndr",
  compression: "none",
  compressAt: "transit",
  rate: 4,
  ttftSlo: 1.0,
  tpotSlo: 0.025,
};

/** The optical engine settings of results.md §11 ("optimistic"). */
const OPTIMISTIC = { enob: 11, maskRate: 20000.0, overlap: true };

/** The colocated pool's GPU: the shared one if both pools use it, else H100. */
export function colocatedDevice(c: Controls): DecodeDevice {
  return c.prefillDevice === c.decodeDevice ? c.decodeDevice : "h100";
}

export function configs(c: Controls): {
  colocated: SimConfig;
  disagg: SimConfig;
} {
  const base: SimConfig = {
    model: c.model,
    device: "h100",
    devicesPerInstance: SIM_MODELS[c.model].devices,
    mode: "disagg",
    nPrefill: c.nPrefill,
    nDecode: c.nDecode,
    nColocated: c.nPrefill + c.nDecode,
    link: c.link,
    ttftSlo: c.ttftSlo,
    tpotSlo: c.tpotSlo,
    ...(c.model === "llama3-8b-hyena-circ" ? { lmHead: "last" as const } : {}),
  };
  const optical = c.prefillDevice.startsWith("optical");
  const disagg: SimConfig = {
    ...base,
    prefillDevice: optical ? "optical-fft" : c.prefillDevice,
    decodeDevice: c.decodeDevice,
    ...(c.prefillDevice === "optical-optimistic" ? { engine: OPTIMISTIC } : {}),
    ...(c.compression !== "none"
      ? { kvCompress: c.compression, kvCompressAt: c.compressAt }
      : {}),
  };
  const colocated: SimConfig = {
    ...base,
    mode: "colocated",
    device: colocatedDevice(c),
  };
  return { colocated, disagg };
}

export type Expected = {
  column: "colocated" | "disagg";
  key: keyof Metrics;
  fmt: Format;
  text: string;
};

export type Preset = {
  id: string;
  label: string;
  /** results.md section and the line the values come from. */
  section: string;
  line: string;
  controls: Controls;
  expected: Expected[];
};

const L8 = {
  ...DEFAULT_CONTROLS,
  model: "llama3-8b",
  rate: 8,
} satisfies Controls;

/**
 * Each preset reproduces one results.md row; `expected` holds the cells as
 * results.md prints them (tests/unit/disagg/simulator.test.ts checks them
 * against the fixture from scripts/disagg_reference.py, and the e2e test
 * checks the widget shows them).
 */
export const PRESETS: Preset[] = [
  {
    id: "sweep-4",
    label: "Defaults, 4 req/s (§4)",
    section: "4",
    line: "| 4 | 25.9 ms | 97.8% | 15.2 ms | 854.2 ms | 99.7% |",
    controls: DEFAULT_CONTROLS,
    expected: [
      { column: "colocated", key: "tpot_p99", fmt: "ms", text: "25.9 ms" },
      { column: "colocated", key: "slo", fmt: "pct", text: "97.8%" },
      { column: "disagg", key: "tpot_p99", fmt: "ms", text: "15.2 ms" },
      { column: "disagg", key: "ttft_p99", fmt: "ms", text: "854.2 ms" },
      { column: "disagg", key: "slo", fmt: "pct", text: "99.7%" },
    ],
  },
  {
    id: "sweep-8",
    label: "Defaults, 8 req/s (§4)",
    section: "4",
    line: "| 8 | 46.8 ms | 14.0% | 16.2 ms | 6,085 ms | 21.0% |",
    controls: { ...DEFAULT_CONTROLS, rate: 8 },
    expected: [
      { column: "colocated", key: "tpot_p99", fmt: "ms", text: "46.8 ms" },
      { column: "colocated", key: "slo", fmt: "pct", text: "14.0%" },
      { column: "disagg", key: "tpot_p99", fmt: "ms", text: "16.2 ms" },
      { column: "disagg", key: "ttft_p99", fmt: "ms", text: "6,085 ms" },
      { column: "disagg", key: "slo", fmt: "pct", text: "21.0%" },
    ],
  },
  {
    id: "hetero-ha",
    label: "H100 prefill + A100 decode (§10)",
    section: "10",
    line: "| H100 prefill + A100 decode | 353.6 ms | 16.8 ms | 7.69 | 100.0% | 0.299 | 558 W | $687 | 2,714 |",
    controls: { ...L8, decodeDevice: "a100" },
    expected: [
      { column: "disagg", key: "ttft_p99", fmt: "ms2", text: "353.6 ms" },
      { column: "disagg", key: "tpot_p99", fmt: "ms", text: "16.8 ms" },
      { column: "disagg", key: "slo", fmt: "pct", text: "100.0%" },
      { column: "disagg", key: "j_tok", fmt: "f3", text: "0.299" },
    ],
  },
  {
    id: "hetero-2ah",
    label: "2× A100 prefill + H100 decode (§10)",
    section: "10",
    line: "| 2x A100 prefill + H100 decode | 1,093.5 ms | 8.9 ms | 7.46 | 96.9% | 0.412 | 793 W | $1,035 | 1,859 |",
    controls: { ...L8, prefillDevice: "a100", nPrefill: 2 },
    expected: [
      { column: "disagg", key: "ttft_p99", fmt: "ms2", text: "1,093.5 ms" },
      { column: "disagg", key: "tpot_p99", fmt: "ms", text: "8.9 ms" },
      { column: "disagg", key: "slo", fmt: "pct", text: "96.9%" },
      { column: "disagg", key: "j_tok", fmt: "f3", text: "0.412" },
    ],
  },
  {
    id: "circ-optical",
    label: "Optical prefill pool, circulant model (§11)",
    section: "11",
    line: "| optical-fft prefill (optimistic) + H100 decode | 90.5 ms | 13.5 ms | 7.69 | 100.0% | 0.194 | 375 W | 100.0% | 5.15 | 2,772 |",
    controls: {
      ...L8,
      model: "llama3-8b-hyena-circ",
      prefillDevice: "optical-optimistic",
    },
    expected: [
      { column: "disagg", key: "ttft_p99", fmt: "ms2", text: "90.5 ms" },
      { column: "disagg", key: "tpot_p99", fmt: "ms", text: "13.5 ms" },
      { column: "disagg", key: "slo", fmt: "pct", text: "100.0%" },
      { column: "disagg", key: "j_tok", fmt: "f3", text: "0.194" },
    ],
  },
  {
    id: "eth25-14-none",
    label: "25 GbE link at 14 req/s (§15)",
    section: "15",
    line: "| 14 req/s | none | 94.0% | 9,657.8 ms | 9,657.8 ms | 77.2 ms | 77.2 ms | 41.0% | 41.0% | 0.268 | 0.268 | 0.0% |",
    controls: { ...L8, link: "eth-25g", rate: 14 },
    expected: [
      { column: "disagg", key: "link_util", fmt: "pct", text: "94.0%" },
      { column: "disagg", key: "handoff_p99", fmt: "ms2", text: "9,657.8 ms" },
      { column: "disagg", key: "tpot_p99", fmt: "ms", text: "77.2 ms" },
      { column: "disagg", key: "slo", fmt: "pct", text: "41.0%" },
    ],
  },
  {
    id: "eth25-14-fp8",
    label: "… with fp8 hand-off compression (§15)",
    section: "15",
    line: "| 14 req/s | fp8 | 53.7% | 166.6 ms | 166.6 ms | 11.5 ms | 11.6 ms | 100.0% | 100.0% | 0.253 | 0.253 | 0.0% |",
    controls: { ...L8, link: "eth-25g", rate: 14, compression: "fp8" },
    expected: [
      { column: "disagg", key: "link_util", fmt: "pct", text: "53.7%" },
      { column: "disagg", key: "handoff_p99", fmt: "ms2", text: "166.6 ms" },
      { column: "disagg", key: "tpot_p99", fmt: "ms", text: "11.5 ms" },
      { column: "disagg", key: "slo", fmt: "pct", text: "100.0%" },
    ],
  },
];
