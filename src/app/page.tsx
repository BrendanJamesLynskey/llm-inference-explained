import Link from "next/link";

import { CheckedStrip, StartButton } from "@/components/ui/Landing";
import ServingTeaserHero, {
  type TeaserData,
} from "@/components/heroes/ServingTeaserHero";
import { runPlain } from "@/lib/disagg/engine";
import { heroLanes, type HeroFile } from "@/lib/disagg/hero";
import heroFile from "../../public/disagg/hero.json";
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

/** The hero's runs: the vendored engine, at build time, on hero.json. */
function teaserData(): TeaserData {
  const f = heroFile as unknown as HeroFile;
  const run = (key: "colocated" | "disagg") => {
    const m = f.modes.find((x) => x.key === key)!;
    const r = runPlain(m.cfg, f.rows);
    return { lanes: heroLanes(r.result.reqs), metrics: r.metrics };
  };
  return {
    colocated: run("colocated"),
    disagg: run("disagg"),
    requests: f.rows.length,
    commit: f.commit,
  };
}

/**
 * Landing page: what the site is, the live simulator in one picture, and the
 * ways in. The hero's first frame is computed on the server by the vendored
 * engine (brief 41); in the browser the same engine re-runs all the requests
 * in a Web Worker when the hero comes into view.
 */
export default function HomePage(): JSX.Element {
  const data = teaserData();
  const ms = (x: number) => `${(x * 1e3).toFixed(x < 0.1 ? 1 : 0)} ms`;
  const s2 = (x: number) => `${x.toFixed(2)} s`;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        LLM Inference Explained
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
        How a model actually generates text, one token at a time.
      </h1>
      <div className="mt-6">
        <p className="max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
          An interactive guide to how real systems serve language models, from
          the KV cache to disaggregated serving.
        </p>
        <div className="mt-6">
          <StartButton href="/learn/01-generation-loop" />
        </div>
        <p className="mt-3 text-sm text-neutral-600 dark:text-neutral-400">
          New to transformers?{" "}
          <a
            href={DECODER_URL}
            className="focus-ring rounded font-medium underline decoration-neutral-400 underline-offset-4 hover:decoration-current"
          >
            First, how one forward pass works
          </a>
        </p>
      </div>
      <figure data-hero-figure className="mt-8 min-w-0">
        <ServingTeaserHero data={data} />
        <figcaption className="mt-2 max-w-3xl text-sm text-neutral-600 dark:text-neutral-400">
          Split, the decode GPUs never wait behind a prompt: on these{" "}
          {data.requests} requests the slowest 1% of gaps between tokens (ITL
          p99) falls from {ms(data.colocated.metrics.itl_p99)} to{" "}
          {ms(data.disagg.metrics.itl_p99)}, while the first token comes later
          (TTFT p99 {s2(data.colocated.metrics.ttft_p99)} against{" "}
          {s2(data.disagg.metrics.ttft_p99)}). The colocated side here has no
          chunked prefill, the usual first fix (
          <Link
            href="/learn/04-batching"
            className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
          >
            chapter 4
          </Link>
          ); with every lever open to both, the{" "}
          <a
            href={`${TRADEOFFS_URL}/learn/13-combining-levers`}
            className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
          >
            Trade-offs site
          </a>{" "}
          finds the split no longer wins on any workload&apos;s goodput.{" "}
          <Link
            href="/learn/13-live-simulator"
            className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
          >
            Run the simulator yourself
          </Link>
        </figcaption>
      </figure>
      <CheckedStrip href="/about#numbers">
        The serving simulator in chapters 11&ndash;14 is
        Disaggregated_Inference_Sim&apos;s own engine, copied byte for byte from
        a pinned commit, and CI requires every request&apos;s timestamps to
        match its Python original; the tiny model&apos;s cached logits are
        bit-identical to full recomputation.
      </CheckedStrip>
      <section aria-labelledby="learn-heading" className="mt-12">
        <h2 id="learn-heading" className="text-xl font-semibold tracking-tight">
          What you&apos;ll learn
        </h2>
        <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
          A forward pass makes one prediction. Serving a model means running
          that pass thousands of times a second for many users at once. These
          chapters cover what real inference systems do about it: the KV cache,
          the roofline, batching, paged memory, FlashAttention, speculative
          decoding, quantisation, parallelism, the metrics that judge them, and
          disaggregated prefill and decode, with a live simulator. Every
          interactive runs tested code in your browser.
        </p>
      </section>

      <nav aria-label="Ways in" className="mt-12 grid gap-4 sm:grid-cols-3">
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

      <p
        data-track-start
        className="mt-12 text-sm text-neutral-600 dark:text-neutral-400"
      >
        New to the Inference track?{" "}
        <a
          href="/inference"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Start here
        </a>
        : the four sites in reading order, what you need first, and a glossary
        of the terms they define.
      </p>
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
