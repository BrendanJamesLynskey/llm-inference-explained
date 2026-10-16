/*
 * src/lib/inference/attentionKernels.ts
 *
 * Operation: three ways to compute the same attention output —
 *            the textbook one, FlashAttention-style tiling with an online
 *            softmax, and split-K ("Flash-Decoding") for one query over a
 *            long context — plus the HBM traffic model that motivates them.
 * Shapes:    Q: [N, d], K, V: [Nk, d] → O: [N, d]. No causal mask here:
 *            the point is the tiling, not the mask.
 * Intuition: softmax needs the row maximum and the row sum, which seem to
 *            need the whole row. Keep a running max m and running sum l and
 *            rescale the partial output whenever m grows; the result is
 *            exact. That lets a kernel stream K and V through fast on-chip
 *            memory in tiles without ever writing the N×N score matrix.
 * MDX:       /learn/06-attention-kernels.
 *
 * After Dao et al., "FlashAttention" (arXiv:2205.14135), Algorithms 0 and 1
 * and Theorem 2, and the Flash-Decoding note (Dao, Haziza, Massa and
 * Sizov, crfm.stanford.edu, 2023).
 */
import { mulberry32, normalSampler } from "@/lib/transformer/random";
import type { Matrix, Vector } from "@/lib/transformer/types";

function dot(a: Vector, b: Vector): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/** Textbook attention: S = QKᵀ/√d, P = softmax(S) row-wise, O = PV. */
export function attentionNaive(Q: Matrix, K: Matrix, V: Matrix): Matrix {
  const d = Q[0]!.length;
  const scale = 1 / Math.sqrt(d);
  return Q.map((q) => {
    const s = K.map((k) => dot(q, k) * scale);
    const m = Math.max(...s);
    const e = s.map((x) => Math.exp(x - m));
    const l = e.reduce((a, b) => a + b, 0);
    const o = new Array<number>(V[0]!.length).fill(0);
    e.forEach((w, j) => V[j]!.forEach((v, c) => (o[c]! += (w / l) * v)));
    return o;
  });
}

/** Running softmax state for one query row: max, sum and unnormalised output. */
export type Partial = { m: number; l: number; o: Vector };

/** Fold keys/values [from, to) into a running state (one query). */
export function foldTile(
  q: Vector,
  K: Matrix,
  V: Matrix,
  from: number,
  to: number,
  st: Partial,
): Partial {
  const scale = 1 / Math.sqrt(q.length);
  let mTile = -Infinity;
  const s: number[] = [];
  for (let j = from; j < to; j++) {
    const x = dot(q, K[j]!) * scale;
    s.push(x);
    if (x > mTile) mTile = x;
  }
  const mNew = Math.max(st.m, mTile);
  // Rescale what we had: every earlier term was exp(x − m_old).
  const alpha = st.m === -Infinity ? 0 : Math.exp(st.m - mNew);
  const o = st.o.map((v) => v * alpha);
  let l = st.l * alpha;
  s.forEach((x, i) => {
    const p = Math.exp(x - mNew);
    l += p;
    V[from + i]!.forEach((v, c) => (o[c]! += p * v));
  });
  return { m: mNew, l, o };
}

/** Merge two partial states (log-sum-exp combine). */
export function mergePartials(a: Partial, b: Partial): Partial {
  const m = Math.max(a.m, b.m);
  const fa = a.m === -Infinity ? 0 : Math.exp(a.m - m);
  const fb = b.m === -Infinity ? 0 : Math.exp(b.m - m);
  return {
    m,
    l: a.l * fa + b.l * fb,
    o: a.o.map((v, c) => v * fa + b.o[c]! * fb),
  };
}

const empty = (d: number): Partial => ({
  m: -Infinity,
  l: 0,
  o: new Array<number>(d).fill(0),
});

/**
 * FlashAttention-style tiling: for each query, stream K/V in tiles of
 * `blockK` keys through the online softmax. Never materialises S or P.
 */
export function attentionTiled(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
  blockK: number,
): Matrix {
  const dv = V[0]!.length;
  return Q.map((q) => {
    let st = empty(dv);
    for (let j = 0; j < K.length; j += blockK) {
      st = foldTile(q, K, V, j, Math.min(K.length, j + blockK), st);
    }
    return st.o.map((v) => v / st.l);
  });
}

/**
 * Split-K for decode: one query, a long context. Each of `splits` workers
 * folds its own slice of the keys in parallel; the partial states are then
 * merged. This is what keeps a GPU busy when batch × heads is too small.
 */
export function attentionSplitK(
  q: Vector,
  K: Matrix,
  V: Matrix,
  splits: number,
): { out: Vector; partials: Partial[] } {
  const dv = V[0]!.length;
  const size = Math.ceil(K.length / splits);
  const partials: Partial[] = [];
  for (let s = 0; s < splits; s++) {
    const from = s * size;
    const to = Math.min(K.length, from + size);
    partials.push(
      from < to ? foldTile(q, K, V, from, to, empty(dv)) : empty(dv),
    );
  }
  const all = partials.reduce(mergePartials, empty(dv));
  return { out: all.o.map((v) => v / all.l), partials };
}

// ─── HBM traffic (elements moved), after FlashAttention §3.2 ──────────

/**
 * Standard attention (the paper's Algorithm 0): read Q, K (2Nd), write S
 * (N²), read S and write P (2N²), read P and V (N² + Nd), write O (Nd).
 * Total 4Nd + 4N² elements: Θ(Nd + N²).
 */
export function hbmStandard(N: number, d: number): number {
  return 4 * N * d + 4 * N * N;
}

/**
 * FlashAttention (Algorithm 1) with on-chip memory of `M` elements: key
 * blocks of Bc = ⌈M / 4d⌉, query blocks of Br = min(Bc, d). K and V are read
 * once (2Nd); for each of the Tc = ⌈N / Bc⌉ key blocks, every query block
 * reads Q and O and writes O (3Nd), and reads and writes the row
 * statistics m and l (4N). Θ(N²d²/M).
 */
export function hbmFlash(N: number, d: number, M: number): number {
  const Bc = Math.ceil(M / (4 * d));
  const Tc = Math.ceil(N / Bc);
  return 2 * N * d + Tc * (3 * N * d + 4 * N);
}

// ─── chapter 6's hero: FlashAttention tile by tile (brief 41) ──────────

/** A seeded N(0, scale²) matrix (mulberry32 + Box-Muller, the site's generator). */
export function seededMatrix(
  rows: number,
  cols: number,
  seed: number,
  scale = 1,
): Matrix {
  const n = normalSampler(mulberry32(seed));
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => n() * scale),
  );
}

/** The hero's problem: 16 queries and keys of width 8, tiles of 4 by 4. */
export const FLASH_HERO = { n: 16, d: 8, br: 4, bc: 4, seeds: [11, 12, 13] };

/** The hero's Q, K and V. */
export function flashHeroInputs(): { Q: Matrix; K: Matrix; V: Matrix } {
  const { n, d, seeds } = FLASH_HERO;
  return {
    Q: seededMatrix(n, d, seeds[0]!),
    K: seededMatrix(n, d, seeds[1]!),
    V: seededMatrix(n, d, seeds[2]!),
  };
}

/** One tile of FlashAttention: query block `qi` folds key block `kj`. */
export type FlashTileState = {
  /** Tile index in schedule order (query block major). */
  t: number;
  qi: number;
  kj: number;
  /** The query rows and key columns this tile covers. */
  rows: [number, number];
  cols: [number, number];
  /** Every query row's running state after this tile (rows not yet started: m = −∞). */
  m: number[];
  l: number[];
  /** The query block's rows of O are written back when its last key block is done. */
  written: number[];
  /** Elements moved to and from HBM so far: Q, K and V tiles read, O written. */
  hbm: number;
};

/**
 * The tile schedule of FlashAttention's forward pass on a small problem,
 * driven by `foldTile` (the same online softmax `attentionTiled` runs):
 * for each block of `br` query rows, stream the keys and values in blocks of
 * `bc`, updating each row's running max m and sum l; when the row block has
 * seen every key block, write its rows of O. The N×N score matrix is never
 * stored: each tile's scores live only inside the tile.
 */
export function flashTileStates(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
  br: number,
  bc: number,
): FlashTileState[] {
  const N = Q.length;
  const d = Q[0]!.length;
  const dv = V[0]!.length;
  const st: Partial[] = Q.map(() => empty(dv));
  const out: FlashTileState[] = [];
  const written: number[] = [];
  let hbm = 0;
  let t = 0;
  for (let i0 = 0; i0 < N; i0 += br) {
    const i1 = Math.min(N, i0 + br);
    hbm += (i1 - i0) * d; // read the query block once
    for (let j0 = 0; j0 < K.length; j0 += bc) {
      const j1 = Math.min(K.length, j0 + bc);
      hbm += (j1 - j0) * (d + dv); // read the key and value block
      for (let i = i0; i < i1; i++)
        st[i] = foldTile(Q[i]!, K, V, j0, j1, st[i]!);
      const last = j1 === K.length;
      if (last) {
        hbm += (i1 - i0) * dv; // write O for these rows
        for (let i = i0; i < i1; i++) written.push(i);
      }
      out.push({
        t: t++,
        qi: i0 / br,
        kj: j0 / bc,
        rows: [i0, i1],
        cols: [j0, j1],
        m: st.map((s) => s.m),
        l: st.map((s) => s.l),
        written: written.slice(),
        hbm,
      });
    }
  }
  return out;
}

/** The caption for one tile state. */
export function flashCaption(s: FlashTileState, nk: number): string {
  const [r0, r1] = s.rows;
  const [c0, c1] = s.cols;
  const done = s.cols[1] === nk;
  const m = s.m[r0]!;
  return `Tile ${s.t + 1}: queries ${r0}–${r1 - 1} against keys ${c0}–${c1 - 1}. Their scores live only in on-chip memory; row ${r0}'s running max is now ${m.toFixed(2)} and its sum ${s.l[r0]!.toFixed(2)}.${done ? ` Rows ${r0}–${r1 - 1} have seen every key: O is normalised and written.` : ""} HBM traffic so far: ${s.hbm.toLocaleString("en-GB")} numbers.`;
}
