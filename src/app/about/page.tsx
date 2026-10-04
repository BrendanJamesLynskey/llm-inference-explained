/**
 * /about — what this site is, where its numbers come from, and credits.
 * Server Component, static.
 */
import Link from "next/link";

import { DECODER_URL, GITHUB_URL } from "@/lib/site";

export const metadata = {
  title: "About",
  description:
    "What LLM Inference Explained is, where its numbers come from, and how it is built.",
};

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

export default function AboutPage(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /about
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        About this site
      </h1>
      <div className="mdx-content mt-6">
        <p>
          <strong>LLM Inference Explained</strong> is the companion to the{" "}
          <a className={A} href={DECODER_URL}>
            Transformer Decoder Explainer
          </a>
          . That site shows what happens inside one forward pass of a decoder.
          This one shows what it takes to serve one: generating token by token,
          and doing it fast for many users at once.
        </p>

        <h2>Where the numbers come from</h2>
        <ul>
          <li>
            <strong>The live tiny model</strong> in chapters 1 and 2 is the
            Transformer Decoder Explainer&rsquo;s pure-TypeScript transformer
            (verified against PyTorch to 1e-5), vendored here and extended with
            a KV cache. Its tests prove cached generation gives bit-identical
            logits to full recomputation. Its weights are random, so its text is
            gibberish; the arithmetic is real.
          </li>
          <li>
            <strong>Hardware and step times</strong> (H100, A100, Llama-3-8B and
            70B, links) come from the cost model of{" "}
            <a
              className={A}
              href="https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim"
            >
              Disaggregated_Inference_Sim
            </a>
            , ported to TypeScript and checked in CI against fixtures that the
            Python package writes, and against its recorded results.
          </li>
          <li>
            <strong>Everything else</strong> (batching, paging, kernels,
            speculative decoding, parallelism, metrics) is small, tested code in{" "}
            <code>src/lib/inference/</code>. Where a model is simplified or a
            coefficient is illustrative, the chapter says so.
          </li>
          <li>
            Papers are cited by arXiv ID and were checked against arXiv. Claims
            about serving engines (vLLM, SGLang, TensorRT-LLM and others)
            describe published designs and change often; follow the links for
            the current state.
          </li>
        </ul>

        <h2>How it is built</h2>
        <p>
          Next.js 14 (App Router) with strict TypeScript, Tailwind, MDX chapters
          rendered on the server with KaTeX, and D3 for chart scales. The design
          system is copied from the Transformer Decoder Explainer so the two
          sites look like one. There is no database and no sign-in: every page
          is static, and the interactives compute in the browser. Vitest covers
          the maths; Playwright checks every chapter at desktop and phone widths
          in light and dark mode.
        </p>
        <p>
          Source, tests and the README:{" "}
          <a className={A} href={GITHUB_URL}>
            github.com/BrendanJamesLynskey/llm-inference-explained
          </a>
          .
        </p>

        <h2>Going further</h2>
        <p>
          The{" "}
          <a
            className={A}
            href="https://brendanjameslynskey.github.io/LLM_Hub_Inference_Simulators/"
          >
            LLM Inference Simulators
          </a>{" "}
          slide series covers how to model all of this in a simulator, with a
          glossary every chapter here links into. Start at{" "}
          <Link className={A} href="/learn">
            the chapters
          </Link>
          .
        </p>
      </div>
    </main>
  );
}
