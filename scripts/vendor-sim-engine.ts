/**
 * scripts/vendor-sim-engine.ts
 *
 * Vendor Disaggregated_Inference_Sim's browser engine (`web/sim_engine.js`)
 * at a pinned commit, and record which one:
 *
 *     pnpm vendor:sim <commit> [path/to/Disaggregated_Inference_Sim]
 *
 * The file is copied byte for byte from `git show <commit>:web/sim_engine.js`
 * (not from the working tree, so local edits can't leak in) into
 * `src/lib/disagg/vendor/sim_engine.js`, and `VENDORED.json` next to it
 * records the repository, the full commit hash and the file's SHA-256. The
 * commit must already be on the simulator's `origin`, so anyone can check
 * the pin. `tests/unit/disagg/vendor.test.ts` fails if the copy, the
 * recorded hash and the parity fixtures' commit disagree.
 *
 * After vendoring, regenerate the parity fixtures at the same commit:
 *
 *     <sim>/.venv/bin/python scripts/disagg_reference.py <sim>
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ref = process.argv[2];
if (!ref) {
  console.error(
    "usage: pnpm vendor:sim <commit> [path/to/Disaggregated_Inference_Sim]",
  );
  process.exit(2);
}
const sim = resolve(
  process.argv[3] ?? join(process.cwd(), "..", "Disaggregated_Inference_Sim"),
);
const git = (...args: string[]): string =>
  execFileSync("git", ["-C", sim, ...args], { encoding: "utf-8" }).trim();

const commit = git("rev-parse", "--verify", `${ref}^{commit}`);
const remote = git("branch", "-r", "--contains", commit);
if (!/origin\//.test(remote)) {
  console.error(`${commit} is not on origin: push it before vendoring.`);
  process.exit(1);
}
const source = execFileSync(
  "git",
  ["-C", sim, "show", `${commit}:web/sim_engine.js`],
  { encoding: "utf-8" },
);
const dest = join(process.cwd(), "src", "lib", "disagg", "vendor");
writeFileSync(join(dest, "sim_engine.js"), source);
const sha256 = createHash("sha256").update(source).digest("hex");
const record = {
  repository:
    "https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim",
  path: "web/sim_engine.js",
  commit,
  committed: git("show", "-s", "--format=%cI", commit),
  sha256,
};
writeFileSync(
  join(dest, "VENDORED.json"),
  JSON.stringify(record, null, 2) + "\n",
);
console.log(`vendored web/sim_engine.js @ ${commit.slice(0, 7)} (${sha256})`);
