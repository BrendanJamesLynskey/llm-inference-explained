/**
 * Each chapter's interactive responds to input (mouse and keyboard).
 */
import { expect, test, type Page } from "@playwright/test";

/** Open a chapter and wait until its code-split widgets have hydrated. */
async function open(page: Page, path: string): Promise<void> {
  await page.goto(path);
  await page.waitForLoadState("networkidle");
}

test("generation loop: prefill, then decode one token at a time", async ({
  page,
}) => {
  await open(page, "/learn/01-generation-loop");
  const w = page.getByTestId("generation-loop");
  await w.getByRole("button", { name: /prefill the prompt/i }).click();
  await expect(w.locator("tbody tr")).toHaveCount(1);
  await expect(w.locator("tbody tr").first()).toContainText("prefill");
  const decode = w.getByRole("button", { name: /decode one token/i });
  await decode.click();
  await decode.click();
  await expect(w.locator("tbody tr")).toHaveCount(3);
  await expect(w.locator("tbody tr").nth(2)).toContainText("decode");
  await w.getByRole("button", { name: "Reset" }).click();
  await expect(w.locator("tbody tr")).toHaveCount(0);
});

test("KV cache: cached logits are bit-identical to recomputation", async ({
  page,
}) => {
  await open(page, "/learn/02-kv-cache");
  const w = page.getByTestId("kv-cache");
  await w.getByRole("button", { name: /prefill/i }).click();
  for (let i = 0; i < 5; i++) {
    await w.getByRole("button", { name: /decode one token/i }).click();
  }
  await expect(w.getByText("bit-identical")).toBeVisible();
  await expect(w.getByRole("status")).toContainText("% of the FLOPs");
  // 6 prompt + 6 generated positions: the 6 prompt columns are filled, plus 5 decodes.
  await expect(w.getByText("Cached positions").locator("..")).toContainText(
    "11",
  );
});

test("KV calculator: MHA is 4× GQA for Llama-3-8B", async ({ page }) => {
  await open(page, "/learn/02-kv-cache");
  const w = page.getByTestId("kv-calculator");
  await expect(
    w.getByText("Bytes / token", { exact: true }).first().locator(".."),
  ).toContainText("128.0 KiB");
  await w.getByRole("radio", { name: "MHA" }).click();
  await expect(
    w.getByText("Bytes / token", { exact: true }).first().locator(".."),
  ).toContainText("512.0 KiB");
});

test("roofline: decode is memory-bound at batch 1, and the batch slider works by keyboard", async ({
  page,
}) => {
  await open(page, "/learn/03-roofline");
  const w = page.getByTestId("roofline");
  await expect(w.getByRole("img")).toHaveAttribute(
    "aria-label",
    /Decode intensity 1\.1, memory-bound/,
  );
  await expect(
    w.getByText("Decode step", { exact: true }).locator(".."),
  ).toContainText("6.20 ms");
  const batch = w.getByLabel(/Decode: batch/);
  await batch.focus();
  await page.keyboard.press("End");
  await expect(w.getByText("Decode tokens/s").locator("..")).toContainText(
    "batch 512",
  );
});

test("batching: four policies, and the timeline switches", async ({ page }) => {
  await open(page, "/learn/04-batching");
  const w = page.getByTestId("batching");
  await expect(w.locator("tbody tr")).toHaveCount(4);
  await w.getByRole("radio", { name: "chunked" }).click();
  await expect(w.getByRole("img")).toHaveAttribute("aria-label", /chunked/);
  await expect(w.locator("svg rect.fill-amber-500").first()).toBeVisible();
});

test("paging: stepping forward allocates blocks and shares a prefix", async ({
  page,
}) => {
  await open(page, "/learn/05-memory-management");
  const w = page.getByTestId("paging");
  const step = w.getByLabel(/^Step/);
  await step.focus();
  for (let i = 0; i < 6; i++) await page.keyboard.press("ArrowRight");
  await expect(w.getByText("S", { exact: true }).first()).toBeVisible();
  await expect(w.getByRole("status")).toContainText("admitted");
});

test("attention kernels: tiled and split-K are exact", async ({ page }) => {
  await open(page, "/learn/06-attention-kernels");
  const w = page.getByTestId("flash-attention");
  const statuses = w.getByRole("status");
  await expect(statuses).toHaveCount(2);
  for (const t of await statuses.allTextContents()) {
    const m = /=\s*([0-9.]+e[-+]\d+)/.exec(t);
    expect(m, t).not.toBeNull();
    expect(Number(m![1])).toBeLessThan(1e-12);
  }
});

test("speculative decoding: emitted tokens track the target", async ({
  page,
}) => {
  await open(page, "/learn/07-speculative-decoding");
  const w = page.getByTestId("speculative");
  await expect(w.getByText("Tokens / target pass").locator("..")).toContainText(
    "3.36",
  );
  await w.getByRole("button", { name: /re-run/i }).click();
  await expect(w.getByRole("button", { name: /seed 2/ })).toBeVisible();
});

test("quantisation: 4-bit weights speed up decode", async ({ page }) => {
  await open(page, "/learn/08-quantisation");
  const w = page.getByTestId("quantisation");
  const tps = w.getByText("Tokens / s").locator("..");
  await expect(tps).toContainText("1.00× BF16");
  await w
    .getByRole("radiogroup", { name: "Weights" })
    .getByRole("radio", { name: "int4" })
    .click();
  await expect(tps).not.toContainText("1.00× BF16");
  await expect(w.getByRole("status")).toContainText("Fits.");
});

test("parallelism: slower links cost more", async ({ page }) => {
  await open(page, "/learn/09-parallelism");
  const w = page.getByTestId("parallelism");
  const share = async () =>
    Number(
      /is (\d+)%/.exec(
        (await w.getByTestId("comm-share").textContent()) ?? "",
      )![1],
    );
  const nv = await share();
  await w.getByRole("radio", { name: "eth-100g" }).click();
  await expect.poll(share).toBeGreaterThan(nv);
});

test("serving metrics: Little's law holds on the simulated run", async ({
  page,
}) => {
  await open(page, "/learn/10-serving-metrics");
  const w = page.getByTestId("metrics");
  const L = await w.getByText("L (measured)").locator("..").textContent();
  const LW = await w.getByText("λ·W").locator("..").textContent();
  const num = (s: string | null) => Number(/(\d+\.\d+)/.exec(s ?? "")![1]);
  expect(num(L)).toBe(num(LW));
  await expect(w.getByRole("img")).toHaveCount(2);
});
