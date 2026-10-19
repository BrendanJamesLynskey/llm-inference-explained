/**
 * Brief 44, the learning layer: every chapter ends with three self-checks
 * and a "What to remember" box, and every numeric answer is the model's
 * value, printed as the chapter prints its numbers.
 *
 * 1. Structure: each chapter has one <CheckYourself> with exactly three
 *    <Check>s (ids unique, numbered 1-3) and, after it, one <Remember> with
 *    three to five takeaways and a <Keep>.
 * 2. Links: every in-page link names an anchor in that chapter; every
 *    cross-chapter link a chapter (and anchor) that exists.
 * 3. Numeric answers: each `of` resolves in the value tree and each `v` in
 *    the learn values; the number the reader is judged against is the
 *    printed one, and it equals the model's value to the printed precision.
 * 4. Choice answers that rest on a model result: the right choice is what
 *    the model says.
 * 5. The shared component renders the question and the answer with no
 *    JavaScript (a native <details>).
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { SelfCheck, isClose } from "@/components/learn/SelfCheck";
import { numericAnswer } from "@/lib/learn/answer";
import {
  LEARN_VALUE_NAMES,
  learnValue,
  lookup,
  type Fmt,
} from "@/lib/learn/values";
import { H100_SXM, LLAMA3_8B } from "@/lib/inference/costModel";
import { decodeBreakdown, withFormats } from "@/lib/inference/quantisation";
import { SECTIONS } from "@/lib/mdx/sections";

const DIR = path.resolve(__dirname, "../../content/inference");
const chapters = readdirSync(DIR)
  .filter((f) => f.endsWith(".mdx"))
  .sort()
  .map((f) => ({ name: f, source: readFileSync(path.join(DIR, f), "utf-8") }));

type CheckTag = { props: Record<string, string>; body: string };

/** Each <Check …>…</Check> in a chapter: its props (raw) and its body. */
function checks(source: string): CheckTag[] {
  return [
    ...source.matchAll(/<Check\s([\s\S]*?)\n?>\n([\s\S]*?)<\/Check>/g),
  ].map((m) => {
    const props: Record<string, string> = {};
    for (const p of (m[1] ?? "").matchAll(
      /(\w+)(?:=("[^"]*"|\{(?:\[[\s\S]*?\]|[^}]*)\}))?/g,
    )) {
      const val = p[2];
      props[p[1]!] = val === undefined ? "true" : val.slice(1, -1);
    }
    return { props, body: m[2] ?? "" };
  });
}

const all = chapters.flatMap((c) =>
  checks(c.source).map((t) => ({ chapter: c.name, ...t })),
);
const byId = new Map(all.map((c) => [c.props.id!, c]));

/** The SI scale of each unit the formatters print. */
const SCALE: Record<string, number> = {
  "": 1,
  "%": 0.01,
  " s": 1,
  " ms": 1e-3,
  " µs": 1e-6,
  " ns": 1e-9,
  " B": 1,
  " kB": 1e3,
  " MB": 1e6,
  " GB": 1e9,
  " TB": 1e12,
  " KB": 1024,
  " B/s": 1,
  " GB/s": 1e9,
  " TB/s": 1e12,
  " TFLOP/s": 1e12,
  " GFLOP/s": 1e9,
  " flop/byte": 1,
  " dB": 1,
  " KiB": 1024,
  " tok/s": 1,
  " req/s": 1,
};

describe("every chapter: three self-checks and a Remember box", () => {
  it("finds every chapter", () => {
    expect(chapters.map((c) => c.name)).toEqual(
      SECTIONS.map((s) => `${s.slug}.mdx`),
    );
  });

  it.each(chapters.map((c) => [c.name, c.source]))("%s", (_name, source) => {
    expect(source.match(/<CheckYourself>/g)).toHaveLength(1);
    expect(checks(source).map((c) => c.props.n)).toEqual(["1", "2", "3"]);
    const start = source.indexOf("<CheckYourself>");
    const end = source.indexOf("</CheckYourself>");
    expect(source.slice(start, end).match(/<Check\s/g)).toHaveLength(3);
    const r = source.indexOf("<Remember>");
    expect(r).toBeGreaterThan(end);
    const remember = source.slice(r, source.indexOf("</Remember>"));
    const items = remember.match(/^- /gm) ?? [];
    expect(items.length).toBeGreaterThanOrEqual(3);
    expect(items.length).toBeLessThanOrEqual(5);
    expect(remember.match(/<Keep>/g)).toHaveLength(1);
    // a line that is only a <V /> becomes its own paragraph in MDX: never
    // leave one mid-sentence (it broke a reveal's text in two)
    expect(
      source.slice(start).match(/^\s*(<V [^>]*\/>|\{" "\})+\s*$/m),
    ).toBeNull();
    expect(source.slice(start).match(/^\s*<V\s*$/m)).toBeNull();
    // nor a bracket left dangling by the wrap ("( 2 ms )" in the page)
    expect(source.slice(start).match(/\($|^\s*\)/m)).toBeNull();
  });

  it("ids are unique across the site and name their chapter", () => {
    const ids = all.map((c) => c.props.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toHaveLength(3 * chapters.length);
    for (const c of all)
      expect(c.props.id, c.chapter).toMatch(
        new RegExp(`^i${c.chapter.slice(0, 2)}-q[123]$`),
      );
  });

  it("every self-check has a question, a reason and a link to its section", () => {
    for (const c of all) {
      expect(c.props.q, c.props.id).toBeTruthy();
      expect(c.body.trim().length, c.props.id).toBeGreaterThan(40);
      expect(c.props.href, c.props.id).toMatch(/^(#|\/)/);
      expect(c.props.more, c.props.id).toBeTruthy();
    }
  });

  it("every in-page link points at an anchor in that chapter", () => {
    for (const c of all) {
      const href = c.props.href!;
      if (!href.startsWith("#")) continue;
      const source = chapters.find((ch) => ch.name === c.chapter)!.source;
      expect(source, `${c.props.id} → ${href}`).toContain(
        `id="${href.slice(1)}"`,
      );
    }
  });

  it("every cross-chapter link names a chapter that exists, and its anchor", () => {
    for (const c of all) {
      const m = /^\/learn\/([\w-]+)(?:#([\w-]+))?$/.exec(c.props.href!);
      if (!m) continue;
      const target = chapters.find((ch) => ch.name === `${m[1]}.mdx`);
      expect(target, `${c.props.id} → ${c.props.href}`).toBeDefined();
      if (m[2]) expect(target!.source).toContain(`id="${m[2]}"`);
    }
  });

  it("a choice question has 2-4 choices and a valid answer", () => {
    for (const c of all) {
      if ((c.props.input ?? "choice") !== "choice") continue;
      const n = (c.props.choices!.match(/"[^"]*"/g) ?? []).length;
      expect(n, c.props.id).toBeGreaterThanOrEqual(2);
      expect(n, c.props.id).toBeLessThanOrEqual(4);
      const a = Number(c.props.answer);
      expect(a, c.props.id).toBeGreaterThanOrEqual(0);
      expect(a, c.props.id).toBeLessThan(n);
    }
  });
});

describe("numeric answers come from the model", () => {
  const numeric = all.filter((c) => (c.props.input ?? "choice") !== "choice");

  it("at least one per chapter on average", () => {
    expect(numeric.length).toBeGreaterThanOrEqual(chapters.length);
  });

  it.each(numeric.map((c) => [c.props.id, c]))(
    "%s equals the model's value as printed",
    (_id, c) => {
      const { of, v } = c.props;
      expect(of ?? v, "needs of or v").toBeDefined();
      const a = numericAnswer(of, v, (c.props.fmt ?? "num") as Fmt);
      if (of !== undefined) expect(a.raw).toBe(lookup(of));
      else expect(a.raw).toBe(learnValue(v!));
      // the printed number, in its printed unit, is the model's value
      const scale = SCALE[a.unit];
      expect(scale, `unit "${a.unit}"`).toBeDefined();
      const back = a.value * scale!;
      const decimals = (a.display.split(".")[1] ?? "").length;
      const half = 0.5 * 10 ** -decimals * scale!;
      expect(Math.abs(back - a.raw)).toBeLessThanOrEqual(
        Math.max(half, Math.abs(a.raw) * 0.005) * (1 + 1e-9) + 1e-12,
      );
      // typing the printed number counts as right
      const tol = c.props.tol !== undefined ? Number(c.props.tol) : 0.1;
      const abs = c.props.abs !== undefined ? Number(c.props.abs) : 0;
      expect(isClose(a.value, a.value, tol, abs)).toBe(true);
      // a slider can reach the answer
      if (c.props.input === "slider") {
        const min = Number(c.props.min);
        const max = Number(c.props.max);
        const step = Number(c.props.step ?? 1);
        expect(a.value).toBeGreaterThanOrEqual(min);
        expect(a.value).toBeLessThanOrEqual(max);
        const k = Math.round((a.value - min) / step);
        expect(isClose(min + k * step, a.value, tol, abs)).toBe(true);
      }
    },
  );

  it("every learn value computes", () => {
    for (const name of LEARN_VALUE_NAMES)
      expect(Number.isFinite(learnValue(name)), name).toBe(true);
  });

  it("the learn values match the chapters' worked numbers", () => {
    expect(learnValue("ch02.kv8b")).toBe(131072);
    expect(learnValue("ch02.kv70b")).toBe(320 * 1024);
    expect(learnValue("ch06.scoresMiB8k")).toBe(128);
    expect(learnValue("ch09.bubble4x4")).toBeCloseTo(3 / 7, 12);
    expect(Math.round(learnValue("ch03.ridge"))).toBe(203);
    expect(Math.round(learnValue("ch03.decode64.tps"))).toBe(5114);
  });
});

describe("choice answers that rest on the model", () => {
  const right = (id: string): string => {
    const c = byId.get(id)!;
    const choices = c.props.choices!.match(/"([^"]*)"/g)!;
    return choices[Number(c.props.answer)]!.slice(1, -1);
  };

  it("i08-q3: at batch 32 and 8,192 context the cache outweighs the weights", () => {
    const d = decodeBreakdown(
      withFormats(LLAMA3_8B, 2, 2),
      H100_SXM,
      1,
      8192,
      32,
    );
    expect(d.kvBytes).toBeGreaterThan(2 * d.weightBytes);
  });

  it("i11-q2 / i12-q3 / i13-q1: the recorded runs say so", () => {
    expect(learnValue("ch11.disagg.tpot99")).toBeLessThan(
      learnValue("ch11.coloc.tpot99"),
    );
    expect(learnValue("ch12.fp8.tpot99")).toBeLessThan(
      learnValue("ch12.none.tpot99"),
    );
    expect(learnValue("ch13.disagg8.slo")).toBeLessThan(0.5);
    expect(right("i12-q3")).toMatch(/^A fix for a saturated link/);
  });

  it("i12-q2: 25 GbE takes longer than the prefill that made the cache", () => {
    expect(learnValue("ch12.eth25")).toBeGreaterThan(0.059);
  });
});

describe("the shared component", () => {
  it("renders the question and the answer without JavaScript", () => {
    const html = renderToStaticMarkup(
      createElement(SelfCheck, {
        id: "t",
        n: 1,
        question: "How many?",
        input: "number",
        value: 32,
        display: "32",
        why: "Because.",
        href: "#x",
      }),
    );
    expect(html).toContain("<details");
    expect(html).toContain("How many?");
    expect(html).toContain("32");
    expect(html).toContain("Because.");
  });
});
