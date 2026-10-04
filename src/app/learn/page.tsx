/**
 * /learn — index of chapters.
 *
 * Server Component, statically rendered. Same layout as
 * transformer-explainer's /learn, minus the per-user progress badges (this
 * site has no accounts).
 */
import Link from "next/link";

import { SECTIONS } from "@/lib/mdx/sections";
import { DECODER_URL } from "@/lib/site";

export const metadata = {
  title: "Learn",
  description:
    "Fourteen chapters on how real LLM inference works, each with a live interactive.",
};

export default function LearnIndex(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /learn
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Inference, chapter by chapter
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        Read in order: each chapter builds on the one before it. Toggle layers
        (Concept / Maths / Code) inside any chapter to choose how deep to go.
        New to transformers? Start with the{" "}
        <a
          href={DECODER_URL}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          Transformer Decoder Explainer
        </a>
        , which walks through one forward pass.
      </p>

      <ol className="mt-10 divide-y divide-neutral-200 dark:divide-neutral-800">
        {SECTIONS.map((s, i) => (
          <li key={s.slug} className="py-5">
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
                {String(i + 1).padStart(2, "0")}
              </span>
              <Link
                href={`/learn/${s.slug}`}
                className="focus-ring rounded text-lg font-medium text-neutral-900 hover:text-accent dark:text-neutral-100"
              >
                {s.title}
              </Link>
            </div>
            <p className="mt-1 pl-9 text-sm text-neutral-600 dark:text-neutral-400">
              {s.summary}
            </p>
          </li>
        ))}
      </ol>
    </main>
  );
}
