/**
 * scripts/smoke-check.ts
 *
 * Post-deploy smoke check, adapted from transformer-explainer's (which also
 * checks a health endpoint and a comment list; this site has neither: no
 * database, no API). Fetches every page and exits non-zero if any fails:
 *
 *     pnpm smoke https://llm-inference-explained.vercel.app
 *
 * With no argument it checks http://localhost:3000. For a protected
 * preview deployment, pass the bypass token as VERCEL_BYPASS (sent as the
 * `x-vercel-protection-bypass` header; never printed).
 *
 * Fails when a page is not a 200 (redirects count as failures), when a
 * chapter's server-rendered HTML lacks its title or its interactive, or when
 * KaTeX reported an error while rendering its maths. It also fetches every
 * recorded workload the live simulator runs on and checks it is whole and
 * from the vendored engine's commit (real data, not just a page shell).
 */
import VENDORED from "@/lib/disagg/vendor/VENDORED.json";
import { WORKLOAD_RATES } from "@/lib/disagg/workloads";
import { SECTIONS } from "@/lib/mdx/sections";

/** The author credit: the last line of the About page (brief 35). */
const CREDIT = [
  'data-author-credit="true">Built by',
  'href="https://brendanjameslynskey.github.io/">Brendan Lynskey</a>',
];

type Result = { path: string; ok: boolean; detail: string };

const headers: Record<string, string> = process.env.VERCEL_BYPASS
  ? { "x-vercel-protection-bypass": process.env.VERCEL_BYPASS }
  : {};

/** The four-track site switch, with this site's place on the button and in the header. */
const SWITCH = [
  'data-site-switch="menu"',
  'data-site-group="foundations"',
  'data-site-group="hardware"',
  'href="https://ai-racks-explained.vercel.app"',
  'href="https://ai-accelerators-explained.vercel.app"',
  'href="https://agent-evals-explained.vercel.app"',
  'href="https://agent-security-explained.vercel.app"',
  'data-site-key="decoder"',
  'href="https://transformer-training-explained.vercel.app"',
  'data-site-place="inference"',
  'data-site-brand="true" class="focus-ring whitespace-nowrap rounded text-sm font-medium tracking-tight" href="/">LLM Inference Explained</a>',
  "data-site-brand",
];

async function checkPage(
  base: string,
  path: string,
  /** Each entry must appear; `a|b` means either. */
  mustContain: string[],
): Promise<Result> {
  try {
    const res = await fetch(base + path, { redirect: "manual", headers });
    if (res.status !== 200) {
      return { path, ok: false, detail: String(res.status) };
    }
    const html = await res.text();
    const missing = mustContain.filter(
      (s) => !s.split("|").some((alt) => html.includes(alt)),
    );
    if (missing.length) {
      return {
        path,
        ok: false,
        detail: `200 but missing ${missing.join(", ")}`,
      };
    }
    if (html.includes("katex-error")) {
      return { path, ok: false, detail: "200 but KaTeX reported an error" };
    }
    return { path, ok: true, detail: "200" };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

async function checkWorkload(base: string, rate: number): Promise<Result> {
  const path = `/disagg/workloads/seed1-rate${rate}.json`;
  try {
    const res = await fetch(base + path, { redirect: "manual", headers });
    if (res.status !== 200)
      return { path, ok: false, detail: String(res.status) };
    const w = (await res.json()) as { commit: string; rows: unknown[] };
    if (w.commit !== VENDORED.commit)
      return { path, ok: false, detail: `200 but commit ${w.commit}` };
    if (w.rows.length !== 800)
      return { path, ok: false, detail: `200 but ${w.rows.length} rows` };
    return {
      path,
      ok: true,
      detail: `200, 800 rows @ ${w.commit.slice(0, 7)}`,
    };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

/** The chapter title as React escapes it in HTML. */
function htmlEscape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/'/g, "&#x27;");
}

async function main(): Promise<void> {
  const base = (process.argv[2] ?? "http://localhost:3000").replace(/\/$/, "");
  console.log(`Smoke-checking ${base}`);
  const results: Result[] = [
    await checkPage(base, "/", [
      "How a model actually generates text",
      ...SWITCH,
    ]),
    await checkPage(base, "/learn", ["Inference, chapter by chapter"]),
    await checkPage(base, "/about", [...CREDIT, "Where the numbers come from"]),
    await checkPage(base, "/inference", [
      'data-testid="reading-order"',
      "data-track-glossary",
    ]),
  ];
  for (const s of SECTIONS) {
    results.push(
      await checkPage(base, `/learn/${s.slug}`, [
        htmlEscape(s.title),
        // client-only interactives (chapters 11, 13, 14) render a marked placeholder
        "data-testid=|data-pending-widget=",
        'data-callout="deeper"',
        "data-self-check=",
      ]),
    );
  }
  // The live simulator's data: every recorded workload must be served, whole,
  // and from the commit the vendored engine came from.
  for (const rate of WORKLOAD_RATES)
    results.push(await checkWorkload(base, rate));
  for (const r of results) {
    console.log(`${r.ok ? "ok  " : "FAIL"} ${r.path} ${r.detail}`);
  }
  const failed = results.filter((r) => !r.ok).length;
  console.log(
    failed
      ? `${failed} of ${results.length} checks failed`
      : `${results.length}/${results.length} ok`,
  );
  process.exit(failed ? 1 : 0);
}

void main();
