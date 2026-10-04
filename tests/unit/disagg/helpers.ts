/** Fixture loaders shared by the disaggregation tests. */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import type { Format, Row, SimConfig } from "@/lib/disagg/engine";

export const readJson = <T>(path: string): T =>
  JSON.parse(readFileSync(join(process.cwd(), path), "utf-8")) as T;

export type Workload = { commit: string; rate: number; rows: Row[] };
export const workload = (rate: number): Workload =>
  readJson<Workload>(`public/disagg/workloads/seed1-rate${rate}.json`);

export type ResultsRow = {
  id: string;
  section: string;
  rate: number;
  line: string;
  runs: Record<string, SimConfig>;
  python: Record<string, Record<string, number | string>>;
  cells: { run: string; key: string; fmt: Format; text: string }[];
};
export const results = readJson<{ commit: string; rows: ResultsRow[] }>(
  "tests/unit/fixtures/disagg_results.json",
);
