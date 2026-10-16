/**
 * Frame tests on the page (brief 41): set key frames of each chapter hero
 * with its scrub bar and require the caption on screen to be the caption
 * built from scripts/hero_reference.py's state for that frame
 * (tests/unit/fixtures/heroes.json); and the landing hero re-runs the
 * engine in the browser and gets the same answer as the server.
 */
import { expect, test, type Locator } from "@playwright/test";

import fx from "../unit/fixtures/heroes.json";

import {
  flashCaption,
  type FlashTileState,
} from "@/lib/inference/attentionKernels";
import {
  roofCaption,
  timelineCaption,
  type RoofClimb,
  type TimelineStep,
} from "@/lib/inference/heroes";
import { kvHeroCaption, type KvHeroState } from "@/lib/inference/kvHero";

test.use({ contextOptions: { reducedMotion: "reduce" } });

async function open(
  page: import("@playwright/test").Page,
  path: string,
  id: string,
) {
  await page.goto(path);
  await expect(page.locator("[data-pending-widget]")).toHaveCount(0, {
    timeout: 30_000,
  });
  const fig = page.getByTestId(id);
  await expect(fig).toBeVisible({ timeout: 30_000 });
  return fig;
}

async function show(fig: Locator, s: number): Promise<void> {
  await fig.getByTestId("scrub").fill(String(s));
  await expect(fig).toHaveAttribute("data-step", String(s));
}

const some = (n: number) => [...new Set([0, 1, Math.floor(n / 2), n - 1])];

test("chapter 1: the timeline's frames are the cost model's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/01-generation-loop", "hero-timeline");
  const steps = fx.timeline.steps as TimelineStep[];
  for (const k of some(steps.length)) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(
      timelineCaption(steps, k),
    );
    await expect(fig.getByTestId(`pass-${k}`)).toHaveAttribute(
      "data-on",
      "true",
    );
  }
});

test("chapter 2: the cache's frames are the reference's", async ({ page }) => {
  const fig = await open(page, "/learn/02-kv-cache", "hero-kv");
  const st = fx.kv.states as unknown as KvHeroState[];
  for (const k of some(st.length)) {
    await show(fig, k);
    const s = st[k]!;
    await expect(fig.getByTestId("caption")).toHaveText(kvHeroCaption(s));
    // one column per cached position in every grid
    await expect(fig.locator("[data-testid^='kv-col-0-K-']")).toHaveCount(
      s.K[0]!.length,
    );
  }
});

test("chapter 3: the roofline's frames are the cost model's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/03-roofline", "hero-roofline");
  const c = fx.roofline as unknown as RoofClimb;
  for (const k of some(c.points.length)) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(roofCaption(c, k));
  }
});

test("chapter 6: the tiles' frames are the reference's", async ({ page }) => {
  const fig = await open(page, "/learn/06-attention-kernels", "hero-flash");
  const st = (fx.flash.states as unknown as FlashTileState[]).map((s) => ({
    ...s,
    m: s.m.map((v) => (v === null ? -Infinity : v)),
  }));
  for (const k of some(st.length)) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(
      flashCaption(st[k]!, 16),
    );
    await expect(fig.locator("[data-written='true']")).toHaveCount(
      st[k]!.written.length,
    );
  }
});

test("landing: the browser re-runs the engine and agrees with the server", async ({
  page,
}) => {
  const fig = await open(page, "/", "hero-teaser");
  await fig.scrollIntoViewIfNeeded();
  await expect(fig.getByTestId("teaser-check")).toHaveAttribute(
    "data-check",
    "same",
    {
      timeout: 60_000,
    },
  );
  await show(fig, 59);
  await expect(
    fig.locator("[data-testid^='teaser-colocated-'][data-phase='done']"),
  ).toHaveCount(8);
});
