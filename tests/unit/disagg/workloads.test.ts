/** The browser-side workload loader: one fetch per load, errors not cached. */
import { afterEach, describe, expect, it, vi } from "vitest";

import { loadWorkload, workloadUrl } from "@/lib/disagg/workloads";

import { workload } from "./helpers";

afterEach(() => vi.unstubAllGlobals());

describe("loadWorkload", () => {
  it("fetches the recorded file once and returns its rows", async () => {
    const rows = workload(6).rows;
    const fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ rows }),
    }));
    vi.stubGlobal("fetch", fetch);
    expect(workloadUrl(6)).toBe("/disagg/workloads/seed1-rate6.json");
    expect(await loadWorkload(6)).toBe(rows);
    expect(await loadWorkload(6)).toBe(rows);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("reports an HTTP error and tries again next time", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ rows: [[0.1, 2, 3]] }),
      });
    vi.stubGlobal("fetch", fetch);
    await expect(loadWorkload(12)).rejects.toThrow(
      "workload 12 req/s: HTTP 404",
    );
    expect(await loadWorkload(12)).toEqual([[0.1, 2, 3]]);
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});
