/**
 * The family's visual language (explained_sites_visual_standard.md §3):
 * Okabe and Ito's colour-blind-safe palette, the same in light and dark
 * mode. The memory-level colours are the family's (GPU Kernels Explained);
 * this site adds one colour per lever family, better / worse, and one per
 * phase of a request. Colours are fills and strokes, never text on white
 * (sky and yellow text fail contrast). "Active" is a highlight, "done" is
 * muted, and waiting is grey plus a hatch pattern, never colour alone.
 *
 * Okabe, M. and Ito, K. (2008), "Color Universal Design (CUD): how to make
 * figures and presentations that are friendly to colorblind people",
 * https://jfly.uni-koeln.de/color/
 */

export const OKABE_ITO = {
  black: "#000000",
  orange: "#E69F00",
  sky: "#56B4E9",
  green: "#009E73",
  yellow: "#F0E442",
  blue: "#0072B2",
  vermillion: "#D55E00",
  purple: "#CC79A7",
} as const;

/** One colour per memory level, the same on every site in the family. */
export const LEVEL_COLOUR = {
  reg: OKABE_ITO.orange,
  smem: OKABE_ITO.green,
  l2: OKABE_ITO.sky,
  hbm: OKABE_ITO.purple,
} as const;

/**
 * One colour per lever family (this site), in the explorer, the matrix
 * legends and the what-if. The baseline and the combined rows are drawn in
 * the text colour (`currentColor`), told apart by their marks. Yellow is
 * always drawn with a dark outline (it fails contrast on white alone).
 */
export const FAMILY_COLOUR: Record<string, string> = {
  baseline: "currentColor",
  batching: OKABE_ITO.orange,
  "kv-memory": OKABE_ITO.sky,
  "prefix-caching": OKABE_ITO.green,
  disaggregation: OKABE_ITO.blue,
  parallelism: OKABE_ITO.vermillion,
  quantisation: OKABE_ITO.purple,
  speculative: OKABE_ITO.yellow,
  combined: "currentColor",
};

/** Better / worse than the baseline (the matrix, the what-if bars). */
export const VERDICT_COLOUR = {
  better: OKABE_ITO.blue,
  worse: OKABE_ITO.vermillion,
} as const;

/**
 * A request's phases in the what-if timeline: queueing is muted plus a
 * hatch (waiting is "stalled": a warning look, never colour alone), prefill
 * orange, the hand-off sky, decode blue.
 */
export const PHASE_COLOUR = {
  queue: "#a3a3a3",
  prefill: OKABE_ITO.orange,
  handoff: OKABE_ITO.sky,
  decode: OKABE_ITO.blue,
} as const;

/** States of an element in an animation. */
export const STATE_COLOUR = {
  active: OKABE_ITO.blue,
  error: OKABE_ITO.vermillion,
} as const;

/** Muted ("done", "idle") greys: Tailwind neutral-400 and neutral-600. */
export const MUTED = { light: "#a3a3a3", dark: "#525252" } as const;
