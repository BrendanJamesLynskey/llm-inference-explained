/**
 * Site-wide constants: this site's URL, its companion sites, and the
 * external series the chapters link into.
 */

/** This site (production). */
export const SITE_URL = "https://llm-inference-explained.vercel.app";

/** The companion Transformer Decoder Explainer. */
export const DECODER_URL = "https://transformer-decoder-explained.vercel.app";

/** The companion LLM Architectures Explained. */
export const ARCHITECTURES_URL =
  "https://llm-architectures-explained.vercel.app";

/** The companion GPU Kernels Explained. */
export const KERNELS_URL = "https://gpu-kernels-explained.vercel.app";

export const GITHUB_URL =
  "https://github.com/BrendanJamesLynskey/llm-inference-explained";

const PAGES = "https://brendanjameslynskey.github.io";

/** A chapter of the Transformer Decoder Explainer. */
export function decoderChapter(slug: string): string {
  return `${DECODER_URL}/learn/${slug}`;
}

/** An entry in the LLM Inference Simulators glossary (`#g-<key>`). */
export function infsimGlossary(key: string): string {
  return `${PAGES}/LLM_Hub_Inference_Simulators/#g-${key}`;
}
