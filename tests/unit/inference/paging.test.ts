import { describe, expect, it } from "vitest";

import {
  append,
  blocksFor,
  contiguousWaste,
  createPool,
  fork,
  release,
  runScenario,
  stats,
} from "@/lib/inference/paging";

describe("paged allocator", () => {
  it("allocates ⌈tokens / blockSize⌉ blocks and wastes at most one block's slack", () => {
    const p = createPool(16, 4);
    expect(append(p, 1, 10)).toBe(true);
    expect(p.tables.get(1)).toHaveLength(blocksFor(p, 10));
    expect(stats(p)).toEqual({
      usedBlocks: 3,
      tokens: 10,
      slack: 2,
      sharedTokens: 0,
    });
    expect(append(p, 1, 2)).toBe(true); // fills the third block exactly
    expect(stats(p).slack).toBe(0);
  });

  it("refuses an append it can't satisfy, changing nothing", () => {
    const p = createPool(2, 4);
    expect(append(p, 1, 8)).toBe(true);
    expect(append(p, 2, 1)).toBe(false);
    expect(p.tables.has(2)).toBe(false);
    expect(stats(p).usedBlocks).toBe(2);
  });

  it("shares blocks on fork and copies a shared partial block on write", () => {
    const p = createPool(8, 4);
    append(p, 1, 6); // blocks [a, b], b half full
    fork(p, 1, 2);
    expect(stats(p)).toMatchObject({
      usedBlocks: 2,
      tokens: 12,
      sharedTokens: 6,
    });
    const before = p.tables.get(2)!.slice();
    append(p, 2, 1); // writes into the shared partial block → copy
    const after = p.tables.get(2)!;
    expect(after[0]).toBe(before[0]); // full block still shared
    expect(after[1]).not.toBe(before[1]); // partial block copied
    expect(p.refs[before[0]!]).toBe(2);
    expect(p.refs[before[1]!]).toBe(1);
    release(p, 1);
    release(p, 2);
    expect(p.free).toHaveLength(8);
    expect(p.refs.every((r) => r === 0)).toBe(true);
  });

  it("fork needs an existing parent", () => {
    expect(() => fork(createPool(2, 4), 9, 10)).toThrow(/no sequence/);
  });

  it("contiguous reservation wastes max_len − length per request", () => {
    expect(contiguousWaste([10, 30, 100], 128)).toBe(118 + 98 + 28);
  });
});

describe("scenario", () => {
  const reqs = [
    { id: 0, arrive: 0, prompt: 12, output: 20 },
    { id: 1, arrive: 1, prompt: 8, output: 30 },
    { id: 2, arrive: 2, prompt: 12, output: 10, sharesWith: 0 },
    { id: 3, arrive: 3, prompt: 20, output: 25 },
    { id: 4, arrive: 4, prompt: 16, output: 15 },
  ];

  it("every request finishes, and no block is ever double-booked", () => {
    const frames = runScenario(reqs, 24, 4, 64, 120);
    const last = frames[frames.length - 1]!;
    expect(last.finished.sort()).toEqual([0, 1, 2, 3, 4]);
    for (const f of frames) {
      const used = f.owners.filter((o) => o !== -1).length;
      expect(used).toBe(f.stats.usedBlocks);
      expect(f.stats.slack).toBeLessThan(4 * Math.max(1, f.running.length));
      // Paged slack is far below what contiguous max-length slabs would waste.
      if (f.running.length)
        expect(f.stats.slack).toBeLessThan(f.contiguousWaste);
    }
  });

  it("shares the prefix when request 2 joins request 0", () => {
    const frames = runScenario(reqs, 24, 4, 64, 6);
    expect(frames.some((f) => f.owners.includes(-2))).toBe(true);
    expect(frames.some((f) => f.stats.sharedTokens > 0)).toBe(true);
  });

  it("preempts the latest-arrived running sequence when blocks run out", () => {
    const frames = runScenario(reqs, 12, 4, 64, 200);
    const i = frames.findIndex((f) => f.event.includes("preempted"));
    expect(i).toBeGreaterThan(0);
    const victim = Number(/#(\d+) preempted/.exec(frames[i]!.event)![1]);
    const before = frames[i - 1]!.running;
    const arrive = (id: number) => reqs.find((r) => r.id === id)!.arrive;
    expect(arrive(victim)).toBe(Math.max(...before.map(arrive)));
    expect(frames[frames.length - 1]!.finished.sort()).toEqual([0, 1, 2, 3, 4]);
  });
});
