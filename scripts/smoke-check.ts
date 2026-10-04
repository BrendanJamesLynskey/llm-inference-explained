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
 * KaTeX reported an error while rendering its maths.
 */
import { SECTIONS } from "@/lib/mdx/sections";

type Result = { path: string; ok: boolean; detail: string };

const headers: Record<string, string> = process.env.VERCEL_BYPASS
  ? { "x-vercel-protection-bypass": process.env.VERCEL_BYPASS }
  : {};

async function checkPage(
  base: string,
  path: string,
  mustContain: string[],
): Promise<Result> {
  try {
    const res = await fetch(base + path, { redirect: "manual", headers });
    if (res.status !== 200) {
      return { path, ok: false, detail: String(res.status) };
    }
    const html = await res.text();
    const missing = mustContain.filter((s) => !html.includes(s));
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

/** The chapter title as React escapes it in HTML. */
function htmlEscape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/'/g, "&#x27;");
}

async function main(): Promise<void> {
  const base = (process.argv[2] ?? "http://localhost:3000").replace(/\/$/, "");
  console.log(`Smoke-checking ${base}`);
  const results: Result[] = [
    await checkPage(base, "/", ["How a model actually generates text"]),
    await checkPage(base, "/learn", ["Inference, chapter by chapter"]),
    await checkPage(base, "/about", ["Where the numbers come from"]),
  ];
  for (const s of SECTIONS) {
    results.push(
      await checkPage(base, `/learn/${s.slug}`, [
        htmlEscape(s.title),
        "data-testid=",
        'data-callout="deeper"',
      ]),
    );
  }
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
