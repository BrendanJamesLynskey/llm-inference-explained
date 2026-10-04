import { describe, expect, it } from "vitest";

import {
  ALPHABET,
  VOCAB_SIZE,
  decode,
  encode,
} from "@/lib/transformer/tokenizer";

describe("tokenizer", () => {
  it("alphabet has exactly 64 characters and matches reference order", () => {
    expect(ALPHABET.length).toBe(64);
    expect(VOCAB_SIZE).toBe(64);
    expect(ALPHABET[0]).toBe("a");
    expect(ALPHABET[25]).toBe("z");
    expect(ALPHABET[26]).toBe("0");
    expect(ALPHABET[35]).toBe("9");
    expect(ALPHABET[36]).toBe(" ");
  });

  it("encodes a known string to the right ids and pads with 0", () => {
    const ids = encode("ab z", 8);
    expect(ids).toEqual([0, 1, 36, 25, 0, 0, 0, 0]);
  });

  it("truncates strings longer than seqLen", () => {
    expect(encode("abcdefghij", 4)).toEqual([0, 1, 2, 3]);
  });

  it("maps unknown characters to id 0", () => {
    // U+00E9 (é) is not in the alphabet.
    expect(encode("éa", 3)).toEqual([0, 0, 0]);
  });

  it("decode is the inverse of encode for an in-vocab string", () => {
    const ids = encode("hello world", 16);
    const text = decode(ids);
    expect(text.startsWith("hello world")).toBe(true);
  });

  it("decode renders unknown ids as the alphabet's id-0 character", () => {
    expect(decode([0, 999])).toBe(`${ALPHABET[0]}${ALPHABET[0]}`);
  });
});
