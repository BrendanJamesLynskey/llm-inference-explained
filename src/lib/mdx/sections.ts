/**
 * Chapter catalogue + filesystem loader for /learn content.
 *
 * MDX sources live under `/content/inference/`. Their slugs and order are
 * defined here (single source of truth) — the `[slug]` route validates
 * incoming params against this list before reading from disk. Same shape
 * as transformer-explainer's `src/lib/mdx/sections.ts`.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

/** Pedagogical order (brief 07 §3: chapters 1–10). */
export const SECTIONS = [
  {
    slug: "01-generation-loop",
    title: "From forward pass to generation loop",
    summary:
      "Prefill, decode, sampling and stopping: why generation is sequential.",
  },
  {
    slug: "02-kv-cache",
    title: "The KV cache",
    summary: "What is cached, why it is exact, and how big it gets.",
  },
  {
    slug: "03-roofline",
    title: "Arithmetic intensity and the roofline",
    summary: "Why decode is memory-bound and prefill compute-bound.",
  },
  {
    slug: "04-batching",
    title: "Batching",
    summary: "Static, dynamic, continuous and chunked prefill.",
  },
  {
    slug: "05-memory-management",
    title: "Memory management",
    summary: "PagedAttention, block tables, prefix sharing and preemption.",
  },
  {
    slug: "06-attention-kernels",
    title: "Attention kernels",
    summary: "FlashAttention's IO-aware tiling; split-K for long contexts.",
  },
  {
    slug: "07-speculative-decoding",
    title: "Speculative decoding",
    summary: "Draft, verify, and the expected tokens per step.",
  },
  {
    slug: "08-quantisation",
    title: "Quantisation for inference",
    summary: "Fewer bytes per weight and per cached value, faster decode.",
  },
  {
    slug: "09-parallelism",
    title: "Parallelism",
    summary: "Tensor, pipeline and expert parallelism and what they send.",
  },
  {
    slug: "10-serving-metrics",
    title: "Serving metrics",
    summary: "TTFT, TPOT, ITL, goodput, tail latency and Little's law.",
  },
] as const;

export type SectionSlug = (typeof SECTIONS)[number]["slug"];

const SLUG_SET = new Set<string>(SECTIONS.map((s) => s.slug));

export function isValidSlug(slug: string): slug is SectionSlug {
  return SLUG_SET.has(slug);
}

export function getSectionMeta(slug: SectionSlug): (typeof SECTIONS)[number] {
  return SECTIONS.find((s) => s.slug === slug) ?? SECTIONS[0];
}

/** Read the raw MDX source for a chapter, or `null` if it doesn't exist. */
export async function readSectionMdx(
  slug: SectionSlug,
): Promise<string | null> {
  const path = join(process.cwd(), "content", "inference", `${slug}.mdx`);
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}
