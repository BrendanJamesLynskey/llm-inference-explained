/*
 * src/lib/inference/paging.ts
 *
 * Operation: a paged KV-cache allocator (the PagedAttention idea): fixed-
 *            size blocks, a block table per sequence, reference counts for
 *            shared prefixes, copy-on-write, and preemption when full.
 * Shapes:    a `Pool` of `numBlocks` blocks of `blockSize` tokens.
 * Intuition: instead of reserving a contiguous max-length slab per request,
 *            hand out small blocks as tokens arrive. Waste is at most one
 *            partly filled block per sequence, and sequences with the same
 *            prompt can point at the same blocks.
 * MDX:       /learn/05-memory-management.
 *
 * After Kwon et al., "Efficient Memory Management for Large Language Model
 * Serving with PagedAttention" (arXiv:2309.06180). A teaching model: real
 * engines (vLLM and others) differ in policy and detail.
 */

export type Pool = {
  blockSize: number;
  numBlocks: number;
  /** Free block ids (a stack: the most recently freed is reused first). */
  free: number[];
  /** Reference count per block (0 = free). */
  refs: number[];
  /** Block table per sequence id. */
  tables: Map<number, number[]>;
  /** Tokens per sequence. */
  lengths: Map<number, number>;
};

export function createPool(numBlocks: number, blockSize: number): Pool {
  const free: number[] = [];
  for (let i = numBlocks - 1; i >= 0; i--) free.push(i);
  return {
    blockSize,
    numBlocks,
    free,
    refs: new Array<number>(numBlocks).fill(0),
    tables: new Map(),
    lengths: new Map(),
  };
}

function takeBlock(p: Pool): number | null {
  const b = p.free.pop();
  if (b === undefined) return null;
  p.refs[b] = 1;
  return b;
}

/** Blocks a sequence of `tokens` tokens needs: ⌈tokens / blockSize⌉. */
export function blocksFor(p: Pool, tokens: number): number {
  return Math.ceil(tokens / p.blockSize);
}

/**
 * Append `n` tokens to sequence `id` (creating it if new). Allocates new
 * blocks as needed; if the last block is shared, copies it first
 * (copy-on-write). Returns false, changing nothing, if blocks run out.
 */
export function append(p: Pool, id: number, n: number): boolean {
  const table = p.tables.get(id) ?? [];
  const len = p.lengths.get(id) ?? 0;
  const needed = blocksFor(p, len + n) - table.length;
  const last = table[table.length - 1];
  const cow =
    last !== undefined && len % p.blockSize !== 0 && p.refs[last]! > 1;
  if (needed + (cow ? 1 : 0) > p.free.length) return false;
  if (cow) {
    // The shared partial block is copied before this sequence writes to it.
    p.refs[last]!--;
    table[table.length - 1] = takeBlock(p)!;
  }
  for (let i = 0; i < needed; i++) table.push(takeBlock(p)!);
  p.tables.set(id, table);
  p.lengths.set(id, len + n);
  return true;
}

/**
 * Start sequence `child` sharing every block of `parent` (a shared prompt
 * prefix, or parallel samples of one prompt). No blocks are copied.
 */
export function fork(p: Pool, parent: number, child: number): void {
  const table = p.tables.get(parent);
  if (!table) throw new Error(`fork: no sequence ${parent}`);
  for (const b of table) p.refs[b]!++;
  p.tables.set(child, table.slice());
  p.lengths.set(child, p.lengths.get(parent)!);
}

/** Release a finished (or preempted) sequence's blocks. */
export function release(p: Pool, id: number): void {
  for (const b of p.tables.get(id) ?? []) {
    p.refs[b]!--;
    if (p.refs[b] === 0) p.free.push(b);
  }
  p.tables.delete(id);
  p.lengths.delete(id);
}

export type PoolStats = {
  usedBlocks: number;
  tokens: number;
  /** Slots in allocated blocks that hold no token (last-block slack). */
  slack: number;
  /** Tokens stored once but used by several sequences. */
  sharedTokens: number;
};

export function stats(p: Pool): PoolStats {
  const usedBlocks = p.numBlocks - p.free.length;
  let tokens = 0;
  for (const n of p.lengths.values()) tokens += n;
  // Count each physical block once: how many of its slots are filled.
  const filled = new Map<number, number>();
  for (const [id, table] of p.tables) {
    const len = p.lengths.get(id)!;
    table.forEach((b, i) => {
      const inBlock = Math.min(p.blockSize, len - i * p.blockSize);
      filled.set(b, Math.max(filled.get(b) ?? 0, inBlock));
    });
  }
  let stored = 0;
  for (const f of filled.values()) stored += f;
  return {
    usedBlocks,
    tokens,
    slack: usedBlocks * p.blockSize - stored,
    sharedTokens: tokens - stored,
  };
}

/**
 * The same requests under contiguous allocation: each reserves `maxLen`
 * slots up front. Returns the slots reserved but not holding a token.
 */
export function contiguousWaste(lengths: number[], maxLen: number): number {
  return lengths.reduce((w, n) => w + (maxLen - n), 0);
}

// ─── a scripted scenario for the widget ────────────────────────────────

export type ScenarioRequest = {
  id: number;
  /** Step at which it arrives. */
  arrive: number;
  prompt: number;
  output: number;
  /** Shares the prompt of this earlier request (prefix caching). */
  sharesWith?: number;
};

export type ScenarioFrame = {
  step: number;
  /** Owner per block (sequence id), -1 free, -2 shared by several. */
  owners: number[];
  running: number[];
  waiting: number[];
  preempted: number[];
  finished: number[];
  stats: PoolStats;
  /** What contiguous max-length reservation would waste at this step. */
  contiguousWaste: number;
  event: string;
};

/**
 * Run a scenario step by step: arrivals join a FCFS queue; each step every
 * running sequence appends one token. When a sequence can't get a block,
 * the latest-arrived running sequence is preempted (its blocks freed, to be
 * recomputed later): the vLLM paper's rule (§4.5, "the latest requests are
 * preempted first").
 */
export function runScenario(
  reqs: ScenarioRequest[],
  numBlocks: number,
  blockSize: number,
  maxLen: number,
  steps: number,
): ScenarioFrame[] {
  const p = createPool(numBlocks, blockSize);
  const generated = new Map<number, number>();
  const running: number[] = [];
  const waiting: number[] = [];
  const preempted = new Set<number>();
  const finished: number[] = [];
  const byId = new Map(reqs.map((r) => [r.id, r]));
  const frames: ScenarioFrame[] = [];

  const admit = (id: number): boolean => {
    const r = byId.get(id)!;
    const parent = r.sharesWith;
    const done = generated.get(id) ?? 0;
    if (parent !== undefined && p.tables.has(parent) && done === 0) {
      // Share the parent's full prompt blocks; only the partial tail copies.
      const full = Math.floor(r.prompt / blockSize) * blockSize;
      const ptab = p.tables.get(parent)!;
      const shared = ptab.slice(0, full / blockSize);
      if (blocksFor(p, r.prompt) - shared.length > p.free.length) return false;
      for (const b of shared) p.refs[b]!++;
      p.tables.set(id, shared.slice());
      p.lengths.set(id, full);
      return append(p, id, r.prompt - full) || (release(p, id), false);
    }
    return append(p, id, r.prompt + done);
  };

  for (let step = 0; step < steps; step++) {
    let event = "";
    for (const r of reqs) if (r.arrive === step) waiting.push(r.id);
    // Admit in FCFS order while blocks allow.
    while (waiting.length && admit(waiting[0]!)) {
      const id = waiting.shift()!;
      running.push(id);
      preempted.delete(id);
      event += `#${id} admitted. `;
    }
    // Decode: every running sequence appends one token.
    for (const id of running.slice()) {
      if (!running.includes(id)) continue;
      while (!append(p, id, 1)) {
        // The latest-arrived running sequence is preempted first.
        const victim = running.reduce((a, b) => {
          const ra = byId.get(a)!;
          const rb = byId.get(b)!;
          return rb.arrive > ra.arrive || (rb.arrive === ra.arrive && b > a)
            ? b
            : a;
        });
        running.splice(running.indexOf(victim), 1);
        release(p, victim);
        preempted.add(victim);
        waiting.push(victim);
        waiting.sort((a, b) => byId.get(a)!.arrive - byId.get(b)!.arrive); // FCFS
        event += `Out of blocks: #${victim} preempted. `;
        if (victim === id) break;
      }
      if (!running.includes(id)) continue;
      const g = (generated.get(id) ?? 0) + 1;
      generated.set(id, g);
      if (g >= byId.get(id)!.output) {
        release(p, id);
        running.splice(running.indexOf(id), 1);
        finished.push(id);
        event += `#${id} finished. `;
      }
    }
    const owners = new Array<number>(numBlocks).fill(-1);
    for (const [id, table] of p.tables) {
      for (const b of table) owners[b] = p.refs[b]! > 1 ? -2 : id;
    }
    const lens = running.map((id) => p.lengths.get(id)!);
    frames.push({
      step,
      owners,
      running: running.slice(),
      waiting: waiting.slice(),
      preempted: [...preempted],
      finished: finished.slice(),
      stats: stats(p),
      contiguousWaste: contiguousWaste(lens, maxLen),
      event: event.trim(),
    });
  }
  return frames;
}
