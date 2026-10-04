import { describe, expect, it } from "vitest";

import {
  addMat,
  addRowBias,
  addVec,
  shape,
  zeros,
  zerosVec,
} from "@/lib/transformer/tensor";

describe("zerosVec", () => {
  it("returns a vector of zeros", () => {
    expect(zerosVec(3)).toEqual([0, 0, 0]);
  });

  it("uses the provided fill value", () => {
    expect(zerosVec(2, 7)).toEqual([7, 7]);
  });

  it("returns an empty array for n=0", () => {
    expect(zerosVec(0)).toEqual([]);
  });
});

describe("zeros", () => {
  it("returns a rectangular matrix of zeros", () => {
    expect(zeros(2, 3)).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it("respects the fill value", () => {
    expect(zeros(2, 2, 1)).toEqual([
      [1, 1],
      [1, 1],
    ]);
  });

  it("returns [] for 0×N", () => {
    expect(zeros(0, 5)).toEqual([]);
  });
});

describe("shape", () => {
  it("reports rows and cols", () => {
    expect(shape([[1, 2, 3]])).toEqual([1, 3]);
    expect(shape(zeros(4, 5))).toEqual([4, 5]);
  });

  it("returns [0, 0] for an empty matrix", () => {
    expect(shape([])).toEqual([0, 0]);
  });

  it("reports cols as 0 if the first row is empty", () => {
    expect(shape([[]])).toEqual([1, 0]);
  });
});

describe("addVec", () => {
  it("adds element-wise", () => {
    expect(addVec([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
  });

  it("throws on length mismatch", () => {
    expect(() => addVec([1], [1, 2])).toThrow(/length mismatch/);
  });
});

describe("addMat", () => {
  it("adds matrices of the same shape", () => {
    expect(
      addMat(
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ),
    ).toEqual([
      [6, 8],
      [10, 12],
    ]);
  });

  it("throws on shape mismatch", () => {
    expect(() => addMat([[1, 2]], [[1]])).toThrow(/shape mismatch/);
  });
});

describe("addRowBias", () => {
  it("adds the bias vector to every row", () => {
    expect(
      addRowBias(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [10, 20, 30],
      ),
    ).toEqual([
      [11, 22, 33],
      [14, 25, 36],
    ]);
  });

  it("throws when the bias length doesn't match cols", () => {
    expect(() => addRowBias([[1, 2]], [1])).toThrow(/bias length/);
  });
});
