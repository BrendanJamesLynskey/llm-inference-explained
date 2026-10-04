/**
 * Captures the README screenshots. Manual run, output committed.
 *
 *   pnpm build && pnpm start   # in another shell
 *   pnpm screenshots           # headless Chromium writes docs/screenshots/*.png
 *
 * Light theme, fixed viewport, and the interactives' default (deterministic)
 * settings, as transformer-explainer's script does.
 */
import { mkdirSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "screenshots");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Shot = { name: string; path: string; widget?: string; click?: RegExp[] };

const SHOTS: Shot[] = [
  { name: "01-landing", path: "/" },
  { name: "02-learn-index", path: "/learn" },
  {
    name: "03-kv-cache",
    path: "/learn/02-kv-cache",
    widget: "kv-cache",
    click: [
      /prefill/i,
      /decode one token/i,
      /decode one token/i,
      /decode one token/i,
      /decode one token/i,
      /decode one token/i,
    ],
  },
  { name: "04-roofline", path: "/learn/03-roofline", widget: "roofline" },
  { name: "05-batching", path: "/learn/04-batching", widget: "batching" },
  {
    name: "06-speculative",
    path: "/learn/07-speculative-decoding",
    widget: "speculative",
  },
];

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "light",
  });
  const page = await context.newPage();
  for (const s of SHOTS) {
    await page.goto(BASE + s.path);
    await page.waitForLoadState("networkidle");
    const target = s.widget ? page.getByTestId(s.widget) : null;
    for (const name of s.click ?? []) {
      await target!.getByRole("button", { name }).first().click();
    }
    const file = path.join(OUT, `${s.name}.png`);
    if (target) await target.screenshot({ path: file });
    else await page.screenshot({ path: file });
    console.log("wrote", path.relative(process.cwd(), file));
  }
  await browser.close();
}

void main();
