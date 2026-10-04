/**
 * scripts/verify-maths.ts
 *
 * Cross-checks our TypeScript transformer implementation against the
 * PyTorch reference fixtures in `tests/unit/fixtures/`.
 *
 * This runs in CI (see .github/workflows/ci.yml). If you change an op in
 * `src/lib/transformer/`, regenerate the fixtures with
 *     python scripts/reference.py
 * and commit them in the same PR. The commit message must explain why.
 *
 * Tolerance is read from the manifest (currently 1e-5).
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { matmul } from "@/lib/transformer/matmul";
import { softmax } from "@/lib/transformer/softmax";
import { layernorm } from "@/lib/transformer/layernorm";
import { gelu } from "@/lib/transformer/gelu";
import { causalMask } from "@/lib/transformer/attention";
import { multiHeadAttention } from "@/lib/transformer/attention";
import { forward } from "@/lib/transformer/model";

const FIX_DIR = join(process.cwd(), "tests", "unit", "fixtures");

type Manifest = {
  version: number;
  tolerance: number;
  config: Record<string, number>;
  alphabet: string;
  files: string[];
};

/**
 * Python's `json.dump` emits non-finite floats as the literals `Infinity`,
 * `-Infinity`, and `NaN` — none of which are valid JSON, so `JSON.parse`
 * rejects them. We see these only in `causal_mask.json` (the upper-triangular
 * mask is all `-Infinity`). Rewrite them to sentinel strings before parsing,
 * then revive them back to the matching JS numbers.
 */
const SENTINELS: Record<string, number> = {
  __NEG_INF__: -Infinity,
  __POS_INF__: Infinity,
  __NAN__: NaN,
};

function loadJson<T>(name: string): T {
  const path = join(FIX_DIR, name);
  if (!existsSync(path)) {
    console.error(
      `\n  Missing fixture: ${path}\n  Run: python scripts/reference.py\n`,
    );
    process.exit(2);
  }
  const raw = readFileSync(path, "utf-8")
    .replace(/(?<![A-Za-z0-9_"])-Infinity\b/g, '"__NEG_INF__"')
    .replace(/(?<![A-Za-z0-9_"-])Infinity\b/g, '"__POS_INF__"')
    .replace(/(?<![A-Za-z0-9_"])NaN\b/g, '"__NAN__"');
  return JSON.parse(raw, (_k, v) =>
    typeof v === "string" && Object.prototype.hasOwnProperty.call(SENTINELS, v)
      ? SENTINELS[v]
      : v,
  ) as T;
}

function maxAbsErr2D(a: number[][], b: number[][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    for (let j = 0; j < ai.length; j++) {
      const e = Math.abs((ai[j] ?? 0) - (bi[j] ?? 0));
      if (e > m) m = e;
    }
  }
  return m;
}

function maxAbsErr3D(a: number[][][], b: number[][][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const e = maxAbsErr2D(a[i]!, b[i]!);
    if (e > m) m = e;
  }
  return m;
}

function check(name: string, err: number, tol: number): boolean {
  const ok = err <= tol;
  const status = ok ? "✓" : "✗";
  console.log(
    `  ${status} ${name.padEnd(28)} max-abs-err=${err.toExponential(3)} (tol ${tol.toExponential(0)})`,
  );
  return ok;
}

async function main(): Promise<void> {
  console.log("Verifying TS maths against PyTorch fixtures...\n");

  const manifest = loadJson<Manifest>("manifest.json");
  const tol = manifest.tolerance;
  let allOk = true;

  // --- matmul --------------------------------------------------------------
  {
    const f = loadJson<{ A: number[][]; B: number[][]; output: number[][] }>(
      "matmul.json",
    );
    const got = matmul(f.A, f.B);
    allOk = check("matmul", maxAbsErr2D(got, f.output), tol) && allOk;
  }

  // --- softmax (row-wise, dim=-1) -----------------------------------------
  {
    const f = loadJson<{ input: number[][]; output_dim_neg1: number[][] }>(
      "softmax.json",
    );
    const got = f.input.map((row) => softmax(row));
    allOk = check("softmax", maxAbsErr2D(got, f.output_dim_neg1), tol) && allOk;
  }

  // --- layernorm -----------------------------------------------------------
  {
    const f = loadJson<{
      input: number[][];
      gamma: number[];
      beta: number[];
      output: number[][];
    }>("layernorm.json");
    const got = f.input.map((row) => layernorm(row, f.gamma, f.beta));
    allOk = check("layernorm", maxAbsErr2D(got, f.output), tol) && allOk;
  }

  // --- gelu ----------------------------------------------------------------
  {
    const f = loadJson<{ input: number[][]; output: number[][] }>("gelu.json");
    const got = f.input.map((row) => row.map((v) => gelu(v)));
    allOk = check("gelu", maxAbsErr2D(got, f.output), tol) && allOk;
  }

  // --- causal mask ---------------------------------------------------------
  {
    const f = loadJson<{ seq_len: number; output: number[][] }>(
      "causal_mask.json",
    );
    const got = causalMask(f.seq_len);
    // Mask contains -Infinity; compare with finite-aware function.
    let m = 0;
    for (let i = 0; i < got.length; i++) {
      for (let j = 0; j < got[i]!.length; j++) {
        const a = got[i]![j]!;
        const b = f.output[i]![j]!;
        if (!Number.isFinite(a) && !Number.isFinite(b)) continue;
        const e = Math.abs(a - b);
        if (e > m) m = e;
      }
    }
    allOk = check("causal_mask", m, tol) && allOk;
  }

  // --- multi-head attention -----------------------------------------------
  {
    type AttnFix = {
      input: number[][];
      W_q: number[][];
      W_k: number[][];
      W_v: number[][];
      W_o: number[][];
      n_heads: number;
      Q: number[][];
      K: number[][];
      V: number[][];
      scores: number[][][];
      weights: number[][][];
      output: number[][];
    };
    const f = loadJson<AttnFix>("attention.json");
    const trace = {
      Q: [] as number[][],
      K: [] as number[][],
      V: [] as number[][],
      scores: [] as number[][][],
      weights: [] as number[][][],
    };
    const out = multiHeadAttention(
      f.input,
      { W_q: f.W_q, W_k: f.W_k, W_v: f.W_v, W_o: f.W_o },
      f.n_heads,
      trace,
    );
    allOk = check("attention.Q", maxAbsErr2D(trace.Q, f.Q), tol) && allOk;
    allOk = check("attention.K", maxAbsErr2D(trace.K, f.K), tol) && allOk;
    allOk = check("attention.V", maxAbsErr2D(trace.V, f.V), tol) && allOk;
    allOk =
      check("attention.scores", maxAbsErr3D(trace.scores, f.scores), tol) &&
      allOk;
    allOk =
      check("attention.weights", maxAbsErr3D(trace.weights, f.weights), tol) &&
      allOk;
    allOk = check("attention.output", maxAbsErr2D(out, f.output), tol) && allOk;
  }

  // --- full forward --------------------------------------------------------
  {
    type FwdFix = {
      config: Record<string, number>;
      input_text: string;
      token_ids: number[];
      tok_emb: number[][];
      pos_emb: number[][];
      x_after_emb: number[][];
      x_final: number[][];
      logits: number[][];
      weights: Record<string, number[][] | number[]>;
    };
    const f = loadJson<FwdFix>("forward.json");
    const got = forward(f.token_ids, f.config, f.weights);
    allOk =
      check("forward.tok_emb", maxAbsErr2D(got.tokEmb, f.tok_emb), tol) &&
      allOk;
    allOk =
      check("forward.pos_emb", maxAbsErr2D(got.posEmb, f.pos_emb), tol) &&
      allOk;
    allOk =
      check("forward.x_final", maxAbsErr2D(got.xFinal, f.x_final), tol) &&
      allOk;
    allOk =
      check("forward.logits", maxAbsErr2D(got.logits, f.logits), tol) && allOk;
  }

  console.log();
  if (!allOk) {
    console.error("Verification FAILED. See checks above.\n");
    process.exit(1);
  }
  console.log("All checks passed.\n");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
