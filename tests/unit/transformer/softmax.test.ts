import { describe, expect, it } from "vitest";

import { softmax, softmaxRows } from "@/lib/transformer/softmax";

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe("softmax", () => {
  it("uniform on uniform input", () => {
    const y = softmax([1, 1, 1, 1]);
    expect(y.every((v) => Math.abs(v - 0.25) < 1e-12)).toBe(true);
  });

  it("sums to 1", () => {
    const y = softmax([0.1, -2, 5, 3]);
    expect(Math.abs(sum(y) - 1)).toBeLessThan(1e-12);
  });

  it("returns [] for an empty input", () => {
    expect(softmax([])).toEqual([]);
  });

  it("handles huge inputs without overflowing", () => {
    const y = softmax([1000, 1001, 1002]);
    // The shift-by-max trick keeps these finite.
    expect(y.every((v) => Number.isFinite(v))).toBe(true);
    expect(Math.abs(sum(y) - 1)).toBeLessThan(1e-12);
  });

  it("masks positions to zero when given an additive -Infinity mask", () => {
    const y = softmax([1, 1, 1], [0, -Infinity, 0]);
    expect(y[1]).toBe(0);
    expect(Math.abs(sum(y) - 1)).toBeLessThan(1e-12);
  });

  it("returns all-zeros when the entire row is masked", () => {
    const y = softmax([1, 2, 3], [-Infinity, -Infinity, -Infinity]);
    expect(y).toEqual([0, 0, 0]);
  });
});

describe("softmaxRows", () => {
  it("softmaxes each row independently", () => {
    const rows = softmaxRows([
      [1, 1, 1],
      [10, 0, 0],
    ]);
    expect(Math.abs(sum(rows[0]!) - 1)).toBeLessThan(1e-12);
    expect(rows[1]![0]).toBeGreaterThan(0.99);
  });

  it("applies a per-row mask", () => {
    const rows = softmaxRows(
      [
        [1, 1],
        [1, 1],
      ],
      [
        [0, -Infinity],
        [-Infinity, 0],
      ],
    );
    expect(rows[0]).toEqual([1, 0]);
    expect(rows[1]).toEqual([0, 1]);
  });
});
