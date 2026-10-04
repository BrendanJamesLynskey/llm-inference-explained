/**
 * Every number quoted in chapters 11–14 comes from one of three sources,
 * and this test checks each against its source and checks the chapter still
 * quotes it:
 *   - a results.md cell, from the fixture written by the Python package at
 *     the vendored commit (and reproduced by the engine in resultsMd.test.ts);
 *   - a results.md line quoted as text (bounds, bullets), stored verbatim in
 *     the same fixture;
 *   - a value computed here by the tested code (the engine on a recorded
 *     workload, or a closed form).
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { engine, FORMATS, run } from "@/lib/disagg/engine";
import { handoff } from "@/lib/disagg/handoff";
import { configs, DEFAULT_CONTROLS } from "@/lib/disagg/simulator";
import { WORKLOAD_RATES } from "@/lib/disagg/workloads";
import {
  H100_SXM,
  LLAMA3_70B,
  LLAMA3_8B,
  costModel,
} from "@/lib/inference/costModel";

import { readJson, results, workload } from "./helpers";

const mdx = (slug: string): string =>
  readFileSync(
    join(process.cwd(), "content", "inference", `${slug}.mdx`),
    "utf-8",
  ).replace(/\s+/g, " ");
const quoted = (slug: string, ...xs: string[]): void => {
  const t = mdx(slug);
  for (const x of xs) expect(t, `${slug} should quote "${x}"`).toContain(x);
};
const quotes = readJson<{ quotes: Record<string, string> }>(
  "tests/unit/fixtures/disagg_results.json",
).quotes;
const rows = new Map(results.rows.map((r) => [r.id, r]));
/** The results.md cell text of fixture row `id`, run `run`, metric `key`. */
const cell = (id: string, key: string, runKey = "run"): string => {
  const c = rows.get(id)!.cells.find((x) => x.key === key && x.run === runKey);
  expect(c, `${id} ${runKey}.${key}`).toBeDefined();
  return c!.text;
};
/** A string that appears verbatim in a quoted results.md line. */
const fromLine = (line: string, x: string): string => {
  expect(line, `results.md line should contain "${x}"`).toContain(x);
  return x;
};

describe("chapter 11: why disaggregate", () => {
  it("prefill and decode step times (results.md §1, via the ported cost model)", () => {
    const cm = costModel(LLAMA3_70B, H100_SXM, { nDevices: 4 });
    expect((cm.prefill([2048]).time * 1e3).toFixed(0)).toBe("134");
    expect((cm.decode(2300 * 16, 16).time * 1e3).toFixed(0)).toBe("15");
    quoted("11-why-disaggregate", "about 134 ms", "about 15 ms");
  });

  it("inter-token latency at 4 req/s (results.md §4)", () => {
    const itl = rows.get("itl-4")!;
    const py = itl.python as Record<string, Record<string, number>>;
    const ratio = (py.colocated!.itl_p99! / py.disagg!.itl_p99!).toFixed(1);
    quoted(
      "11-why-disaggregate",
      `colocated p50 ${cell("itl-4", "itl_p50", "colocated")}`,
      `disaggregated ${cell("itl-4", "itl_p50", "disagg")}`,
      `p99 ${cell("itl-4", "itl_p99", "colocated")}`,
      `worst gap of ${cell("itl-4", "itl_max", "colocated")}`,
      `against p99 ${cell("itl-4", "itl_p99", "disagg")}`,
      `worst gap of ${cell("itl-4", "itl_max", "disagg")}`,
      `p99 ITL ${fromLine(itl.line, `${ratio}x`)}`,
    );
  });

  it("TPOT, SLOs, TTFT and energy (results.md §3–4)", () => {
    quoted(
      "11-why-disaggregate",
      `from ${cell("sweep-4", "tpot_p99", "colocated")} to ${cell("sweep-4", "tpot_p99", "disagg")}`,
      `from ${cell("sweep-4", "slo", "colocated")} to ${cell("sweep-4", "slo", "disagg")}`,
      `${cell("sweep-6", "slo", "colocated")} against ${cell("sweep-6", "slo", "disagg")}`,
      `TTFT p99 is ${cell("energy-colocated", "ttft_p99")} colocated and ${cell("energy-1p1d", "ttft_p99")} disaggregated`,
      `draws ${cell("energy-1p1d", "avg_w")} on average against ${cell("energy-colocated", "avg_w")}`,
      `${cell("energy-1p1d", "j_tok")} J per output token against ${cell("energy-colocated", "j_tok")}`,
    );
  });
});

describe("chapter 12: moving the KV cache", () => {
  const base = {
    model: LLAMA3_8B,
    devices: 1,
    prompt: 2048,
    compression: "none",
    layerwise: false,
  } as const;
  const ms1 = (s: number) => `${(s * 1e3).toFixed(1)} ms`;

  it("hand-off sizes and the link table (closed form; §14's header)", () => {
    const h8 = handoff({ ...base, link: "ib-ndr" });
    const h70 = handoff({
      ...base,
      model: LLAMA3_70B,
      devices: 4,
      link: "ib-ndr",
    });
    const mb = (b: number) => `${(b / 1e6).toFixed(1)} MB`;
    fromLine(quotes["handoff-bytes"]!, `GQA KV cache ${mb(h8.bytes)}`);
    quoted(
      "12-moving-the-kv-cache",
      "128 KiB per token",
      "320 KiB per token",
      mb(h8.bytes),
      mb(h70.bytes),
      "131{,}072",
    );
    const table: [string, string, string][] = [
      ["nvlink4", "NVLink 4", "450 GB/s"],
      ["cpo-optical", "Co-packaged optics (illustrative)", "200 GB/s"],
      ["ib-ndr", "InfiniBand NDR 400G", "50 GB/s"],
      ["eth-100g", "100 GbE", "12.5 GB/s"],
      ["eth-25g", "25 GbE", "3.125 GB/s"],
    ];
    for (const [link, name, bw] of table) {
      expect(`${engine.LINKS[link]!.bw / 1e9} GB/s`).toBe(bw);
      quoted(
        "12-moving-the-kv-cache",
        `| ${name} | ${bw} | ${ms1(handoff({ ...base, link }).transferTime)} |`,
      );
    }
  });

  it("prefill, link saturation and layer-wise streaming (closed form)", () => {
    const h = handoff({ ...base, link: "eth-25g" });
    const s = handoff({ ...base, link: "eth-25g", layerwise: true });
    const s8 = handoff({
      ...base,
      link: "eth-25g",
      layerwise: true,
      compression: "fp8",
    });
    fromLine(
      quotes["eth25-saturation"]!,
      `near ${h.linkSaturationRate.toFixed(1)} req/s`,
    );
    quoted(
      "12-moving-the-kv-cache",
      `${ms1(h.prefillTime)} for this prompt`,
      `at most **${h.linkSaturationRate.toFixed(1)} hand-offs per second**`,
      `$T_x' = ${ms1(h.transferTime - 20e-6).replace(" ms", "")}$ ms, $T_p = ${ms1(h.prefillTime).replace(" ms", "")}$ ms`,
      `leaves ${ms1(s.exposedTime)} exposed instead of ${ms1(h.exposedTime)}`,
      `with FP8 as well, ${(s8.exposedTime * 1e3).toFixed(2)} ms`,
      `64/17 = ${(64 / 17).toFixed(2)}×`,
    );
  });

  it("links at 8 req/s and compression on 25 GbE (results.md §14–15)", () => {
    for (const lk of [
      "nvlink4",
      "ib-ndr",
      "cpo-optical",
      "eth-100g",
      "eth-25g",
    ])
      expect(cell(`link-${lk}`, "ttft_p99")).toBe("353.6 ms");
    quoted(
      "12-moving-the-kv-cache",
      `TTFT p99 is ${cell("link-ib-ndr", "ttft_p99")} on every one`,
      `from ${cell("link-ib-ndr", "tpot_p99")} on InfiniBand to ${cell("link-eth-25g", "tpot_p99")} on 25 GbE`,
      `busy ${cell("link-eth-25g", "link_util")} of the time`,
      `link is ${cell("eth25-14-none", "link_util")} busy`,
      `reaches ${cell("eth25-14-none", "handoff_p99")}`,
      `TPOT p99 is ${cell("eth25-14-none", "tpot_p99")} and only ${cell("eth25-14-none", "slo")}`,
      `link to ${cell("eth25-14-fp8", "link_util")} busy`,
      `hand-off p99 to ${cell("eth25-14-fp8", "handoff_p99")}, TPOT p99 to ${cell("eth25-14-fp8", "tpot_p99")}`,
      `SLO attainment to ${cell("eth25-14-fp8", "slo")}`,
      `${cell("ib-fp8", "tpot_p99")} either way`,
    );
    expect(cell("ib-fp8", "tpot_p99")).toBe(cell("link-ib-ndr", "tpot_p99"));
    const cpo = rows.get("link-cpo-optical")!.line;
    const ib = rows.get("link-ib-ndr")!.line;
    quoted(
      "12-moving-the-kv-cache",
      `${fromLine(cpo, "| 0.006 |").slice(2, 7)} J of link energy per request against ${fromLine(ib, "| 0.032 |").slice(2, 7)} J`,
      `same TPOT p99 of ${cell("link-cpo-optical", "tpot_p99")}`,
    );
  });

  it("heterogeneous pools (results.md §10)", () => {
    quoted(
      "12-moving-the-kv-cache",
      `leaves TTFT p99 at ${cell("hetero-ha", "ttft_p99")}`,
      `TPOT p99 from ${cell("hetero-hh", "tpot_p99")} to ${cell("hetero-ha", "tpot_p99")}`,
      `${cell("hetero-ha", "slo")} met`,
      `from ${cell("hetero-hh", "j_tok")} to ${cell("hetero-ha", "j_tok")} J per output token`,
      `TTFT p99 grows to ${cell("hetero-ah", "ttft_p99")}`,
      `back to ${cell("hetero-2ah", "ttft_p99")}, with ${cell("hetero-2ah", "slo")}`,
    );
  });
});

describe("chapter 13: the live simulator", () => {
  it("presets, the analytic bound and the optical pool (results.md §4, §8, §11, §15)", () => {
    quoted(
      "13-live-simulator",
      `about ${fromLine(quotes.capacity!, "7.49")} req/s`,
      `TTFT p99 jumps to ${cell("sweep-8", "ttft_p99", "disagg")} and only ${cell("sweep-8", "slo", "disagg")}`,
      `KV link (${cell("eth25-14-none", "link_util")} busy)`,
      `TTFT p99 of ${cell("circ-optical", "ttft_p99")} against ${cell("circ-gpu", "ttft_p99")}`,
      `at ${cell("circ-optical", "j_tok")} J per token against ${cell("circ-gpu", "j_tok")}`,
      "up to 8,192 prompt tokens",
      "up to 256 sequences",
    );
  });

  it("the parity claim: twelve of thirteen identical, the thirteenth within 2.2e-16", () => {
    type Case = {
      name: string;
      cfg: Parameters<typeof engine.simulate>[0];
      rows: Parameters<typeof engine.simulate>[1];
      stamps: (number | null)[][];
    };
    const cases = readJson<{ cases: Case[] }>(
      "tests/unit/fixtures/disagg_parity.json",
    ).cases;
    let identical = 0;
    let worst = 0;
    for (const c of cases) {
      const r = engine.simulate(c.cfg, c.rows);
      let same = true;
      r.reqs.forEach((q, i) =>
        [
          q.prefillStart,
          q.firstToken,
          q.kvStart,
          q.kvReady,
          q.decodeStart,
          q.finish,
        ].forEach((x, k) => {
          const y = c.stamps[i]![k]!;
          if (x !== y) {
            same = false;
            worst = Math.max(worst, Math.abs((x as number) - y) / Math.abs(y));
          }
        }),
      );
      if (same) identical++;
    }
    expect(cases).toHaveLength(13);
    expect(identical).toBe(12);
    expect(worst).toBeLessThanOrEqual(2.2e-16);
    quoted(
      "13-live-simulator",
      "for thirteen configurations",
      "twelve of the thirteen are, bit for bit",
      "within 2.2e-16",
    );
  });
});

describe("chapter 14: trade-offs", () => {
  const m = (
    model: "llama3-8b" | "llama3-70b",
    link: "ib-ndr" | "eth-25g",
    rate: (typeof WORKLOAD_RATES)[number],
  ) => {
    const c = configs({ ...DEFAULT_CONTROLS, model, link });
    const rows = workload(rate).rows;
    return { c: run(c.colocated, rows), d: run(c.disagg, rows) };
  };

  it("small model: both meet every SLO from 2 to 14 req/s (engine)", () => {
    for (const r of WORKLOAD_RATES) {
      const x = m("llama3-8b", "ib-ndr", r);
      expect(x.c.slo).toBe(1);
      expect(x.d.slo).toBe(1);
    }
    quoted(
      "14-tradeoffs",
      "(59.0 ms for 2,048 tokens)",
      "100.0% of requests at every load from 2 to 14 req/s; so does 1P1D",
      `TPOT p99 improves from ${cell("t11-colocated", "tpot_p99")} to ${cell("t11-1p1d", "tpot_p99")}`,
      `TTFT p99 worsens from ${cell("t11-colocated", "ttft_p99")} to ${cell("t11-1p1d", "ttft_p99")}`,
    );
  });

  it("high load and the 2P1D fix (engine; results.md §4 and §8)", () => {
    const x = m("llama3-70b", "ib-ndr", 10);
    quoted(
      "14-tradeoffs",
      `about ${fromLine(quotes.capacity!, "7.49")} req/s of these prompts`,
      `1P1D meets ${FORMATS.pct(x.d.slo)} of SLOs, worse than colocated (${FORMATS.pct(x.c.slo)})`,
      `2P1D meets ${fromLine(quotes["2p1d"]!, "100.0%")} at every load from 2 to 10 req/s`,
      `prefill ${fromLine(quotes.capacity!, "7.49")} req/s, decode ${fromLine(quotes.capacity!, "28.8")} and KV link ${fromLine(quotes.capacity!, "74.5")}`,
    );
    expect(quotes["2p1d"]).toContain(
      "SLO met 100.0%, 100.0%, 100.0%, 100.0%, 100.0%",
    );
  });

  it("slow links (engine and closed form)", () => {
    const h = handoff({
      model: LLAMA3_70B,
      devices: 4,
      prompt: 2048,
      link: "eth-25g",
      compression: "none",
      layerwise: false,
    });
    for (const r of [2, 3, 4, 5, 6, 8, 10, 12] as const) {
      const x = m("llama3-70b", "eth-25g", r);
      expect(x.d.slo, `${r} req/s`).toBeLessThan(x.c.slo);
    }
    const x14 = m("llama3-70b", "eth-25g", 14);
    expect([x14.c.slo, x14.d.slo]).toEqual([0, 0]);
    const x = m("llama3-70b", "eth-25g", 5);
    quoted(
      "14-tradeoffs",
      `${(h.bytes / 1e6).toFixed(1)} MB hand-offs fill the link at about ${h.linkSaturationRate.toFixed(1)} req/s`,
      "every load from 2 to 12 req/s (at 14 both fail)",
      `meets ${FORMATS.pct(x.d.slo)} of SLOs against colocated's ${FORMATS.pct(x.c.slo)}, with the link ${FORMATS.pct(x.d.link_util)} busy`,
    );
  });

  it("low load: static share and energy per token (results.md §6; engine)", () => {
    const x = m("llama3-70b", "ib-ndr", 2);
    quoted(
      "14-tradeoffs",
      `at 0.5 req/s static power is ${fromLine(quotes["proportionality-0.5"]!, "54%")}`,
      `${x.d.j_tok.toFixed(3)} J per output token disaggregated, ${x.c.j_tok.toFixed(3)} colocated`,
    );
  });
});
