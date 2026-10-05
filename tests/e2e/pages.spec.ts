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

test("the cross-site switch links all three sites, this one current", async ({
  page,
}) => {
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Companion sites" });
  await expect(nav.getByRole("link", { name: "Decoder" })).toHaveAttribute(
    "href",
    "https://transformer-decoder-explained.vercel.app",
  );
  await expect(
    nav.getByRole("link", { name: "Architectures" }),
  ).toHaveAttribute("href", "https://llm-architectures-explained.vercel.app");
  await expect(nav.getByRole("link", { name: "Inference" })).toHaveAttribute(
    "aria-current",
    "true",
  );
});

test("the learn index lists every chapter", async ({ page }) => {
  await page.goto("/learn");
  for (const s of SECTIONS) {
    await expect(
      page.getByRole("link", { name: s.title, exact: true }),
    ).toBeVisible();
  }
});
