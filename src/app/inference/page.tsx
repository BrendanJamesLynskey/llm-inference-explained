/**
 * /inference — the Inference track's start page (brief 44): the four sites
 * in reading order, what to know first, how a chapter works (layers,
 * self-checks, what to remember), and a glossary whose terms link to the
 * chapter section that defines each one, on whichever site defines it.
 * Hosted here because Serving is the track's first site. Server Component,
 * no client JavaScript.
 */
import { ReadingOrder } from "@/components/learn/ReadingOrder";
import {
  FOUNDATIONS,
  HARDWARE,
  INFERENCE_GLOSSARY as GLOSSARY,
  INFERENCE_SITES,
} from "@/lib/learn/inference";
import { withShare } from "@/lib/share";
import { infsimGlossary } from "@/lib/site";

export const metadata = withShare("/inference", {
  title: "Inference: start here",
  description:
    "The Inference track's start page: Serving, Architectures, Numerics and Trade-offs in reading order, what you need to know first, and a glossary of the terms they define.",
});

const LINK =
  "focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300";

export default function InferencePage(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        Inference track
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Start here: serving a model, and what it costs
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        Four sites, one question: what does it take to serve a large language
        model well? Serving follows one request from prompt to answer and
        through a serving cluster; Architectures shows how modern models change
        what is served; Numerics covers the number formats that make it cheaper;
        Trade-offs measures every lever on the same workloads. Read them in
        order: each assumes the one before.
      </p>

      <ReadingOrder
        track="Inference"
        intro="Each site's chapters build on each other; start at chapter 1 of Serving. Every number is computed by the site's tested model or recorded by its simulator."
        sites={INFERENCE_SITES}
        current="serving"
      />

      <section aria-labelledby="prereq-heading" className="mt-12">
        <h2
          id="prereq-heading"
          className="text-xl font-semibold tracking-tight"
        >
          What you need first
        </h2>
        <ul className="mt-4 list-disc space-y-2 pl-6 text-neutral-700 dark:text-neutral-300">
          <li>
            <strong>How a transformer decoder works.</strong> Tokens, attention
            with its keys and values, the feed-forward block and sampling. The{" "}
            <a href={FOUNDATIONS} className={LINK}>
              Foundations track
            </a>{" "}
            (Decoder and Training) covers it.
          </li>
          <li>
            <strong>Latency and throughput.</strong> The difference between how
            long one request waits and how many requests are served a second,
            and why percentiles say more than averages.
          </li>
          <li>
            <strong>A little about GPUs.</strong> That memory bandwidth and
            arithmetic are separate limits. The{" "}
            <a href={HARDWARE} className={LINK}>
              Hardware track
            </a>{" "}
            goes deep; Serving&rsquo;s roofline chapter is enough to start.
          </li>
          <li>
            <strong>Reading code is optional.</strong> The Code layer shows the
            TypeScript that runs, and the{" "}
            <a href={infsimGlossary("ttft")} className={LINK}>
              simulator decks&rsquo; glossary
            </a>{" "}
            defines the serving terms again; hide the code with the layer
            switch.
          </li>
        </ul>
      </section>

      <section aria-labelledby="how-heading" className="mt-12">
        <h2 id="how-heading" className="text-xl font-semibold tracking-tight">
          How a chapter works
        </h2>
        <p className="mt-4 text-neutral-700 dark:text-neutral-300">
          A chapter opens with an animation of its mechanism, computed by the
          site&rsquo;s tested model, then explains it in three layers you can
          switch on and off: Concept (words and pictures), Maths (the formulas)
          and Code (the code that runs). It ends with{" "}
          <strong>Check yourself</strong>, three questions to commit to before
          you reveal the answers (the numbers come from the same model), and{" "}
          <strong>What to remember</strong>, the chapter in a few lines.
        </p>
      </section>

      <section aria-labelledby="glossary-heading" className="mt-12">
        <h2
          id="glossary-heading"
          className="text-xl font-semibold tracking-tight"
        >
          Glossary
        </h2>
        <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
          Each term links to the section that defines it, on the site that
          defines it. Some terms mean different things on different sites
          (goodput, ρ, the pipeline bubble); the{" "}
          <a
            href="https://agent-harnesses-explained.vercel.app/definitions"
            className={LINK}
            data-definitions-link
          >
            shared definitions
          </a>{" "}
          give each one definition and one formula, and say where a site
          differs.
        </p>
        <dl
          data-track-glossary
          className="mt-4 divide-y divide-neutral-200 text-sm dark:divide-neutral-800"
        >
          {GLOSSARY.map((g) => (
            <div key={g.term} className="py-2 sm:flex sm:gap-4">
              <dt className="font-medium sm:w-56 sm:shrink-0">
                <a href={g.href} className={LINK}>
                  {g.term}
                </a>
              </dt>
              <dd className="text-neutral-600 dark:text-neutral-400">
                {g.what}
              </dd>
            </div>
          ))}
        </dl>
      </section>
    </main>
  );
}
