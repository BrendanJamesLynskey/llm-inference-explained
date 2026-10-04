/**
 * axe-core scans (as in transformer-explainer): fail on serious or critical
 * violations, in light and dark mode.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const PAGES = [
  "/",
  "/learn",
  "/about",
  "/learn/02-kv-cache",
  "/learn/03-roofline",
  "/learn/04-batching",
  "/learn/07-speculative-decoding",
  "/learn/12-moving-the-kv-cache",
  "/learn/13-live-simulator",
] as const;

for (const scheme of ["light", "dark"] as const) {
  test.describe(scheme, () => {
    test.use({ colorScheme: scheme });
    for (const path of PAGES) {
      test(`a11y: ${path} has no serious or critical violations`, async ({
        page,
      }) => {
        await page.goto(path);
        await page.waitForLoadState("networkidle");
        if (path.includes("13-live"))
          await page
            .locator('[data-cell="disagg.ttft_p99"]')
            .waitFor({ timeout: 20_000 });
        const results = await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze();
        const blocking = results.violations.filter((v) =>
          ["serious", "critical"].includes(v.impact ?? ""),
        );
        if (blocking.length > 0) {
          console.log(
            "axe blocking violations on",
            path,
            JSON.stringify(
              blocking.map((b) => ({
                id: b.id,
                nodes: b.nodes.slice(0, 3).map((n) => n.target),
              })),
              null,
              2,
            ),
          );
        }
        expect(blocking).toEqual([]);
      });
    }
  });
}
