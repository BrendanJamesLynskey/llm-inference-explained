import Link from "next/link";

import { HeroFlopsSvg } from "@/components/viz/HeroFlopsSvg";
import { heroPreview } from "@/lib/inference/heroPreview";
import {
  ARCHITECTURES_URL,
  DECODER_URL,
  KERNELS_URL,
  NUMERICS_URL,
  SILICON_URL,
  TRADEOFFS_URL,
} from "@/lib/site";

const ENTRY_POINTS = [
  {
    href: "/learn",
    title: "Chapters",
    blurb:
      "Fourteen chapters, from the generation loop to disaggregated serving. Toggle the Concept, Maths and Code layers to choose your depth.",
    cta: "See the chapters →",
  },
  {
    href: "/learn/02-kv-cache",
    title: "The KV cache, live",
    blurb:
      "A real tiny decoder generating in your browser, its cache filling column by column, checked against recomputation at every step.",
    cta: "Watch it fill →",
  },
  {
    href: "/learn/03-roofline",
    title: "The roofline",
    blurb:
      "Why decode waits on memory and prefill on arithmetic, with H100 and A100 figures from a tested simulator.",
    cta: "Open the roofline →",
  },
] as const;

/**
 * Landing page: what the site is, one real computation, and the ways in.
 *
 * Server Component with no client JavaScript of its own (the pattern of
 * transformer-explainer's landing page): the figure is computed on the
 * server by the same KV-cache code the chapters run, and drawn as static SVG.
 */
export default function HomePage(): JSX.Element {
  const preview = heroPreview();
  const n = preview.cached.length;
  const pct = (preview.cached[n - 1]! / preview.uncached[n - 1]!) * 100;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <section className="grid items-center gap-10 lg:grid-cols-[1fr_auto]">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
            LLM Inference Explained
          </p>
          <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            How a model actually generates text, one token at a time.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            A forward pass makes one prediction. Serving a model means running
            that pass thousands of times a second for many users at once. These
            chapters cover what real inference systems do about it: the KV
            cache, the roofline, batching, paged memory, FlashAttention,
            speculative decoding, quantisation, parallelism, the metrics that
            judge them, and disaggregated prefill and decode, with a live
            simulator. Every interactive runs tested code in your browser.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/learn/01-generation-loop"
              className="focus-ring rounded bg-accent px-4 py-2 text-sm font-medium text-accent-fg hover:opacity-90"
            >
              Start with chapter 1 →
            </Link>
            <a
              href={DECODER_URL}
              className="focus-ring rounded border border-neutral-300 px-4 py-2 text-sm font-medium hover:border-accent dark:border-neutral-700"
            >
              First, how one forward pass works
            </a>
          </div>
        </div>

        <figure className="mx-auto w-full max-w-[300px] rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
          <HeroFlopsSvg data={preview} />
          <figcaption className="mt-3 text-xs text-neutral-600 dark:text-neutral-400">
            Generating {n} tokens with a tiny decoder, computed on the server
            for this page. With the KV cache it does {pct.toFixed(0)}% of the
            arithmetic of recomputing everything each step, and its logits
            differ from the recomputed ones by{" "}
            {preview.maxDiff === 0
              ? "exactly 0"
              : preview.maxDiff.toExponential(1)}
            .{" "}
            <Link
              href="/learn/02-kv-cache"
              className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
            >
              See why
            </Link>
          </figcaption>
        </figure>
      </section>

      <nav aria-label="Ways in" className="mt-16 grid gap-4 sm:grid-cols-3">
        {ENTRY_POINTS.map((e) => (
          <Link
            key={e.href}
            href={e.href}
            className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
          >
            <h2 className="font-semibold">{e.title}</h2>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              {e.blurb}
            </p>
            <p className="mt-4 text-sm font-medium text-accent dark:text-indigo-300">
              {e.cta}
            </p>
          </Link>
        ))}
      </nav>

      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        The companion sites: the{" "}
        <a
          href={DECODER_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Transformer Decoder Explainer
        </a>{" "}
        walks through the forward pass itself,{" "}
        <a
          href={ARCHITECTURES_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          LLM Architectures Explained
        </a>{" "}
        shows how real models&rsquo; designs differ,{" "}
        <a
          href={KERNELS_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          GPU Kernels Explained
        </a>{" "}
        shows how a GPU executes the maths,{" "}
        <a
          href={NUMERICS_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Numerics Explained
        </a>{" "}
        shows the number formats and quantisation behind it,{" "}
        <a
          href={SILICON_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Systolic Arrays Explained
        </a>{" "}
        shows the matrix hardware of TPUs, and{" "}
        <a
          href={TRADEOFFS_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Inference Trade-offs Explained
        </a>{" "}
        measures which serving lever helps which metric. The source of this one
        is on{" "}
        <a
          href="https://github.com/BrendanJamesLynskey/llm-inference-explained"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          GitHub
        </a>
        .
      </p>
    </main>
  );
}
