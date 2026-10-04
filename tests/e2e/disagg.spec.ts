/**
 * Chapters 11–14: the disaggregation interactives run the vendored engine in
 * the browser on recorded workloads. The simulator's presets must display
 * exactly the cells of their results.md lines, read here from the fixture
 * written by the Python package (not from the widget's own expectations).
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { PRESETS } from "../../src/lib/disagg/simulator";

type Fixture = {
  rows: {
    id: string;
    line: string;
    cells: { run: string; key: string; text: string }[];
  }[];
};
const fixture = JSON.parse(
  readFileSync(
    join(process.cwd(), "tests/unit/fixtures/disagg_results.json"),
    "utf-8",
  ),
) as Fixture;

async function open(page: Page, path: string): Promise<void> {
  await page.goto(path);
  await page.waitForLoadState("networkidle");
}

test("interference: disaggregation cuts p99 ITL 12.4x at 4 req/s (results.md §4)", async ({
  page,
}) => {
  await open(page, "/learn/11-why-disaggregate");
  const w = page.getByTestId("interference");
  const line = fixture.rows.find((r) => r.id === "itl-4")!.line;
  expect(line).toContain("12.4x");
  await expect(w.getByRole("status")).toContainText("12.4×", {
    timeout: 15_000,
  });
  await expect(w.getByText("ITL p99, colocated").locator("..")).toContainText(
    "188.8 ms",
  );
  await w.getByRole("radio", { name: "6" }).click();
  await expect(w.getByRole("status")).toContainText("At 6 req/s");
  await expect(w.getByRole("img")).toHaveAttribute("aria-label", /Token gaps/);
});

test("KV hand-off: 268.4 MB over 25 GbE, and streaming hides most of it", async ({
  page,
}) => {
  await open(page, "/learn/12-moving-the-kv-cache");
  const w = page.getByTestId("kv-handoff");
  await expect(w.getByText("KV to hand off").locator("..")).toContainText(
    "268.4 MB",
  );
  const exposed = w.getByText("Exposed after prefill").locator("..");
  await expect(exposed).toContainText("85.9 ms");
  await w.getByRole("radio", { name: "layer by layer" }).click();
  await expect(exposed).toContainText("28.7 ms");
  await w.getByRole("radio", { name: "fp8" }).click();
  await expect(exposed).toContainText("1.4 ms");
  const prompt = w.getByLabel(/Prompt tokens/);
  await prompt.focus();
  await page.keyboard.press("End");
  await expect(w.getByText("KV to hand off").locator("..")).toContainText(
    "4.29 GB",
  );
});

test.describe("live simulator", () => {
  for (const p of PRESETS) {
    test(`preset ${p.id} shows its results.md cells`, async ({ page }) => {
      const row = fixture.rows.find((r) => r.id === p.id)!;
      await open(page, "/learn/13-live-simulator");
      const w = page.getByTestId("disagg-sim");
      await w.getByLabel(/Preset/).selectOption(p.id);
      await expect(w.getByRole("status")).toContainText(
        `✓ Matches results.md §${p.section}`,
        { timeout: 20_000 },
      );
      await expect(w.getByRole("status")).toContainText(row.line);
      for (const c of row.cells) {
        const col = c.run === "colocated" ? "colocated" : "disagg";
        const shown = w.locator(`[data-cell="${col}.${c.key}"]`);
        if ((await shown.count()) === 0) continue; // not a table row
        await expect(shown).toContainText(c.text);
      }
      expect(await w.getByText("≠").count()).toBe(0);
    });
  }

  test("changing a control leaves the preset and reruns", async ({ page }) => {
    await open(page, "/learn/13-live-simulator");
    const w = page.getByTestId("disagg-sim");
    await expect(w.locator('[data-cell="disagg.ttft_p99"]')).toContainText(
      "854.2 ms",
      { timeout: 20_000 },
    );
    const decode = w.getByLabel("Decode instances");
    await decode.focus();
    await page.keyboard.press("ArrowRight");
    await expect(w.getByLabel(/Preset/)).toHaveValue("");
    await expect(w.getByRole("status")).toHaveCount(0, { timeout: 20_000 });
    await expect(w.locator('[data-cell="disagg.ttft_p99"]')).toBeVisible();
    await expect(w.locator("thead")).toContainText("2D");
  });
});

test("trade-offs: the sweep runs nine loads in both modes", async ({
  page,
}) => {
  await open(page, "/learn/14-tradeoffs");
  const w = page.getByTestId("tradeoffs");
  const status = w.getByRole("status");
  await expect(status).toContainText("of 9 loads", { timeout: 30_000 });
  await expect(status).toContainText(
    "colocated 287.6 ms, disaggregated 353.6 ms",
  );
  await w.getByRole("radio", { name: "Llama-3-70B ×4" }).click();
  await expect(status).toContainText("disaggregated 6,085 ms", {
    timeout: 30_000,
  });
  await expect(w.getByRole("img")).toHaveCount(2);
});
