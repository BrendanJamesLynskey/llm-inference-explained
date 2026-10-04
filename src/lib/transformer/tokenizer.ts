/*
 * src/lib/transformer/tokenizer.ts
 *
 * Operation: char-level tokenizer over a fixed 64-char alphabet.
 * Shapes:    encode(text, seq_len) → number[seq_len];
 *            decode(ids)            → string.
 * Intuition: deliberately the simplest possible tokenizer (SPEC §4 calls it
 *            out as a teaching simplification). Unknown chars map to id 0
 *            and short input is right-padded to `seq_len` with id 0.
 * MDX:       /learn/decoder/02-embeddings (ids → vectors).
 */

/**
 * The 64-char alphabet. Must match `scripts/reference.py:ALPHABET` exactly,
 * otherwise the PyTorch fixtures will disagree with our forward pass.
 */
export const ALPHABET =
  "abcdefghijklmnopqrstuvwxyz" + // 26
  "0123456789" + //                 10
  " .,!?;:'\"()-\n" + //            13
  "+-*/=<>[]{}@#$%"; //             15  (total: 64)

/* c8 ignore start — guard against accidental alphabet edits; unreachable today. */
if (ALPHABET.length !== 64) {
  throw new Error(
    `tokenizer: alphabet must be 64 chars, got ${ALPHABET.length}`,
  );
}
/* c8 ignore stop */

/** Vocabulary size of the toy tokenizer. */
export const VOCAB_SIZE = ALPHABET.length;

const CHAR_TO_ID = new Map<string, number>();
for (let i = 0; i < ALPHABET.length; i++) {
  CHAR_TO_ID.set(ALPHABET[i]!, i);
}

/**
 * Encode `text` to token ids, truncated or right-padded with id 0 to `seqLen`.
 * Unknown characters map to id 0 (same fallback as the padding token).
 */
export function encode(text: string, seqLen: number): number[] {
  const out = new Array<number>(seqLen).fill(0);
  const limit = Math.min(text.length, seqLen);
  for (let i = 0; i < limit; i++) {
    const ch = text[i]!;
    out[i] = CHAR_TO_ID.get(ch) ?? 0;
  }
  return out;
}

/**
 * Decode an array of token ids back to a string. Out-of-range ids are
 * rendered as the alphabet's id-0 character (a placeholder, not an error).
 */
export function decode(ids: number[]): string {
  let s = "";
  for (const id of ids) {
    s += ALPHABET[id] ?? ALPHABET[0]!;
  }
  return s;
}
