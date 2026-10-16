/*
 * src/lib/inference/kvHero.ts
 *
 * Operation: the states of chapter 2's hero, the KV cache filling: prefill
 *            a prompt (every prompt position's keys and values written at
 *            once), then, round by round, the new token's query attends over
 *            every cached key (per block and head), its own key and value
 *            join the cache, and the next token is picked (greedy).
 * Shapes:    kvHeroStates(prompt, rounds, weights) → KvHeroState[];
 *            K and V per block are [positions, d_model].
 * Intuition: the cache only ever grows by one column per block; nothing
 *            already in it changes. The attention weights show which cached
 *            columns the new token actually reads from.
 *            scripts/hero_reference.py rebuilds every state with PyTorch on
 *            the same weights; tests/unit/inference/kvHero.test.ts requires
 *            them to agree.
 * MDX:       /learn/02-kv-cache.
 */
import { decodeStep, prefill } from "@/lib/transformer/kvcache";
import type { ModelWeights } from "@/lib/transformer/model";
import { softmax } from "@/lib/transformer/softmax";
import type { Matrix, Vector } from "@/lib/transformer/types";

import { TOY_CONFIG, toyEncode, toyLabel } from "./toyModel";

export type KvHeroState = {
  phase: "prefill" | "attend" | "append" | "pick";
  /** Round: 0 is the prefill, r ≥ 1 the r-th decode step. */
  round: number;
  /** The tokens so far (prompt, then generated). */
  ids: number[];
  /** Positions cached in every block. */
  cached: number;
  /** Position of the token being processed (the newest). */
  pos: number;
  /** Per block: the cache's keys and values, one row per cached position. */
  K: Matrix[];
  V: Matrix[];
  /** attend: per block and head, the new query's weights over 0..pos. */
  attn: Vector[][] | null;
  /** pick: the chosen next token and its probability. */
  next: number | null;
  prob: number | null;
};

const copy = (m: Matrix): Matrix => m.map((r) => r.slice());

function argmaxFirst(v: Vector): number {
  let b = 0;
  for (let i = 1; i < v.length; i++) if (v[i]! > v[b]!) b = i;
  return b;
}

/**
 * Prefill `prompt`, then `rounds` greedy decode steps. Each decode round
 * has three states: attend (the weights, before the new key and value are
 * drawn as cached), append (the cache with the new column) and pick.
 */
export function kvHeroStates(
  prompt: string,
  rounds: number,
  weights: ModelWeights,
): KvHeroState[] {
  const cfg = TOY_CONFIG;
  const ids = toyEncode(prompt);
  const { logits, cache } = prefill(ids, cfg, weights);
  const out: KvHeroState[] = [];
  const snap = (): { K: Matrix[]; V: Matrix[] } => ({
    K: cache.layers.map((l) => copy(l.K)),
    V: cache.layers.map((l) => copy(l.V)),
  });
  const p0 = softmax(logits[ids.length - 1]!);
  let next = argmaxFirst(p0);
  out.push({
    phase: "prefill",
    round: 0,
    ids: ids.slice(),
    cached: cache.length,
    pos: ids.length - 1,
    ...snap(),
    attn: null,
    next,
    prob: p0[next]!,
  });
  for (let r = 1; r <= rounds; r++) {
    const before = snap();
    ids.push(next);
    const pos = ids.length - 1;
    const attn: Vector[][] = [];
    const l = decodeStep(next, cache, cfg, weights, undefined, attn);
    const base = { round: r, ids: ids.slice(), pos, next: null, prob: null };
    out.push({ ...base, phase: "attend", cached: pos, ...before, attn });
    const after = snap();
    out.push({ ...base, phase: "append", cached: pos + 1, ...after, attn });
    const p = softmax(l);
    next = argmaxFirst(p);
    out.push({
      ...base,
      phase: "pick",
      cached: pos + 1,
      ...after,
      attn,
      next,
      prob: p[next]!,
    });
  }
  return out;
}

const pct = (v: number): string => `${(100 * v).toFixed(0)}%`;
const q = (id: number): string => `“${toyLabel(id)}”`;

/** The caption for one state. */
export function kvHeroCaption(s: KvHeroState): string {
  switch (s.phase) {
    case "prefill":
      return `Prefill: the ${s.ids.length} prompt tokens go through the model in one pass, and each block caches their ${s.ids.length} keys and values. Most likely next token: ${q(s.next!)} (${pct(s.prob!)}).`;
    case "attend": {
      // the cached position the first block's first head reads most
      const w = s.attn![0]![0]!;
      const j = argmaxFirst(w);
      return `Step ${s.round}: the new token ${q(s.ids[s.pos]!)} at position ${s.pos} asks every cached key; block 1, head 0 reads most from position ${j} ${q(s.ids[j]!)} (${pct(w[j]!)}).`;
    }
    case "append":
      return `Step ${s.round}: its own key and value join the cache, one column per block (${s.cached} now); the ${s.cached - 1} earlier columns are unchanged.`;
    case "pick":
      return `Step ${s.round}: the output head picks ${q(s.next!)} (${pct(s.prob!)}); next step it is the new token.`;
  }
}
