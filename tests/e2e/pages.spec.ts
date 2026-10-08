/**
 * Every page renders at desktop and phone widths, in light and dark mode,
 * with no page errors, no console errors and no horizontal overflow.
 */
import { expect, test, type Page } from "@playwright/test";

import { SECTIONS } from "../../src/lib/mdx/sections";

const PAGES = [
  "/",
  "/learn",
  "/about",
  ...SECTIONS.map((s) => `/learn/${s.slug}`),
];

async function collectErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return errors;
}

for (const scheme of ["light", "dark"] as const) {
  for (const width of [1280, 390]) {
    test.describe(`${scheme} @ ${width}px`, () => {
      test.use({
        colorScheme: scheme,
        viewport: { width, height: 900 },
      });
      for (const path of PAGES) {
        test(`${path} renders cleanly`, async ({ page }) => {
          const errors = await collectErrors(page);
          const res = await page.goto(path);
          expect(res?.status()).toBe(200);
          await expect(page.locator("h1").first()).toBeVisible();
          await page.waitForLoadState("networkidle");
          // Widgets are code-split: wait for every interactive to mount.
          const widgets = page.locator("figure[data-testid]");
          if (path.startsWith("/learn/")) {
            await expect(widgets.first()).toBeVisible();
          }
          const overflow = await page.evaluate(() => {
            const el = document.scrollingElement!;
            return el.scrollWidth - el.clientWidth;
          });
          expect(overflow, "horizontal overflow (px)").toBeLessThanOrEqual(0);
          const bg = await page.evaluate(
            () => getComputedStyle(document.body).backgroundColor,
          );
          // neutral-950 in dark mode, white in light (globals.css body rule).
          expect(bg).toBe(
            scheme === "dark" ? "rgb(10, 10, 10)" : "rgb(255, 255, 255)",
          );
          expect(await page.locator(".katex-error").count()).toBe(0);
          expect(errors).toEqual([]);
        });
      }
    });
  }
}

test("every chapter has an interactive, maths and go-deeper links", async ({
  page,
}) => {
  for (const s of SECTIONS) {
    await page.goto(`/learn/${s.slug}`);
    await expect(page.locator("figure[data-testid]").first()).toBeVisible();
    expect(
      await page.locator('[data-callout="deeper"] a').count(),
    ).toBeGreaterThan(1);
    expect(
      await page.locator('[data-callout="prereq"] a').count(),
    ).toBeGreaterThan(0);
    expect(await page.locator('.te-layer[data-layer="maths"]').count()).toBe(1);
  }
});

test.describe("all layers on, at 390px", () => {
  test.use({ viewport: { width: 390, height: 900 } });
  test("no chapter overflows with the Maths and Code layers shown", async ({
    page,
  }) => {
    for (const s of SECTIONS) {
      await page.goto(`/learn/${s.slug}`);
      for (const k of ["maths", "code"]) {
        const sw = page.getByRole("switch", { name: new RegExp(k, "i") });
        if ((await sw.getAttribute("aria-checked")) === "false")
          await sw.click();
      }
      await expect(
        page.locator('.te-layer[data-layer="code"]').first(),
      ).toBeVisible();
      const overflow = await page.evaluate(() => {
        const el = document.scrollingElement!;
        return el.scrollWidth - el.clientWidth;
      });
      expect(
        overflow,
        `${s.slug}: horizontal overflow (px)`,
      ).toBeLessThanOrEqual(0);
    }
  });
});

test("the layer toggle shows the maths layer", async ({ page }) => {
  await page.goto("/learn/02-kv-cache");
  const maths = page.getByRole("switch", { name: /maths/i });
  await expect(maths).toHaveAttribute("aria-checked", "false");
  await expect(page.locator('.te-layer[data-layer="maths"]')).toBeHidden();
  await maths.click();
  await expect(page.locator("html")).toHaveAttribute("data-layer-maths", "on");
  await expect(page.locator('.te-layer[data-layer="maths"]')).toBeVisible();
  expect(
    await page.locator('.te-layer[data-layer="maths"] .katex').count(),
  ).toBeGreaterThan(3);
});

test("the two-group site switch: a toggle and a row on desktop, a dropdown below lg", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  const full = page.locator("[data-site-switch='full']");
  await expect(full).toBeVisible();
  const llm = full.locator("nav[data-site-group='llm']");
  const agents = full.locator("nav[data-site-group='agents']");
  // starts on this site's group
  await expect(llm).toBeVisible();
  await expect(agents).toBeHidden();
  await expect(llm.getByRole("link", { name: "Inference" })).toHaveAttribute(
    "aria-current",
    "true",
  );
  for (const [name, href] of [
    ["Decoder", "https://transformer-decoder-explained.vercel.app"],
    ["Inference", "https://llm-inference-explained.vercel.app"],
    ["Architectures", "https://llm-architectures-explained.vercel.app"],
    ["Kernels", "https://gpu-kernels-explained.vercel.app"],
    ["Numerics", "https://numerics-explained.vercel.app"],
    ["Silicon", "https://systolic-arrays-explained.vercel.app"],
    ["Trade-offs", "https://inference-tradeoffs-explained.vercel.app"],
  ] as const)
    await expect(llm.getByRole("link", { name })).toHaveAttribute("href", href);
  // the toggle shows the agent sites (CSS only)
  await full.getByText("Agents", { exact: true }).click();
  await expect(agents).toBeVisible();
  await expect(llm).toBeHidden();
  await expect(agents.getByRole("link", { name: "Harnesses" })).toHaveAttribute(
    "href",
    "https://agent-harnesses-explained.vercel.app",
  );
  await expect(agents.getByRole("link", { name: "Protocols" })).toHaveAttribute(
    "href",
    "https://agent-protocols-explained.vercel.app",
  );
  for (const soon of ["Context", "Orchestration", "Evals", "Security"]) {
    await expect(agents.getByText(soon)).toBeVisible();
    await expect(agents.getByRole("link", { name: soon })).toHaveCount(0);
  }
  // keyboard: the toggle is a pair of radio buttons
  await page.getByRole("radio", { name: "Show the LLM systems sites" }).focus();
  await page.keyboard.press("Space");
  await expect(llm).toBeVisible();

  // below lg (not sm): the dropdown
  await page.setViewportSize({ width: 768, height: 800 });
  await expect(full).toBeHidden();
  const compact = page.locator("[data-site-switch='compact']");
  await expect(compact).toBeVisible();
  await page.setViewportSize({ width: 390, height: 800 });
  await expect(compact).toBeVisible();
  await compact.locator("summary").click();
  await expect(compact.getByText("LLM systems")).toBeVisible();
  await expect(compact.getByText("Agents", { exact: true })).toBeVisible();
  await expect(
    compact.getByRole("link", { name: "Inference" }),
  ).toHaveAttribute("aria-current", "true");
  await expect(
    compact.getByRole("link", { name: "Harnesses" }),
  ).toHaveAttribute("href", "https://agent-harnesses-explained.vercel.app");
  await expect(
    compact.getByRole("link", { name: "Protocols" }),
  ).toHaveAttribute("href", "https://agent-protocols-explained.vercel.app");
  await expect(compact.getByText("Security")).toBeVisible();
  await expect(compact.getByRole("link", { name: "Security" })).toHaveCount(0);
  const box = await compact
    .getByRole("link", { name: "Decoder" })
    .boundingBox();
  expect(box!.height).toBeGreaterThanOrEqual(44);
  const overflow = await page.evaluate(
    () =>
      document.scrollingElement!.scrollWidth -
      document.scrollingElement!.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test("the learn index lists every chapter", async ({ page }) => {
  await page.goto("/learn");
  for (const s of SECTIONS) {
    await expect(
      page.getByRole("link", { name: s.title, exact: true }),
    ).toBeVisible();
  }
});
