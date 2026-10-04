"use client";

/**
 * FlashAttention in two halves: the HBM-traffic model (why tiling pays) and
 * a live check that tiled attention and split-K decoding give the textbook
 * result.
 */
import { useMemo, useState } from "react";

import { Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import {
  attentionNaive,
  attentionSplitK,
  attentionTiled,
  hbmFlash,
  hbmStandard,
} from "@/lib/inference/attentionKernels";
import { H100_SXM } from "@/lib/inference/costModel";
import { mulberry32, normalSampler } from "@/lib/transformer/random";
import type { Matrix } from "@/lib/transformer/types";

const BYTES = 2; // 16-bit elements
const bw = H100_SXM.mem_bw * H100_SXM.bw_eff;

function randn(
  rows: number,
  cols: number,
  seed: number,
  scale: number,
): Matrix {
  const n = normalSampler(mulberry32(seed));
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => n() * scale),
  );
}

function maxDiff(a: number[], b: number[]): number {
  let m = 0;
  a.forEach((v, i) => (m = Math.max(m, Math.abs(v - b[i]!))));
  return m;
}

export function FlashAttentionWidget(): JSX.Element {
  const [logN, setLogN] = useState(12);
  const [d, setD] = useState(64);
  const [sramKb, setSramKb] = useState(192);
  const [tile, setTile] = useState(8);
  const [splits, setSplits] = useState(4);
  const N = 2 ** logN;
  const M = (sramKb * 1024) / BYTES;
  const std = hbmStandard(N, d);
  const fl = hbmFlash(N, d, M);

  const demo = useMemo(() => {
    const Q = randn(48, 16, 1, 3);
    const K = randn(48, 16, 2, 3);
    const V = randn(48, 16, 3, 1);
    const ref = attentionNaive(Q, K, V);
    const tiled = attentionTiled(Q, K, V, tile);
    let worst = 0;
    tiled.forEach((row, i) => (worst = Math.max(worst, maxDiff(row, ref[i]!))));
    const Kl = randn(4096, 16, 4, 2);
    const Vl = randn(4096, 16, 5, 1);
    const q = randn(1, 16, 6, 2)[0]!;
    const sk = attentionSplitK(q, Kl, Vl, splits);
    const one = attentionNaive([q], Kl, Vl)[0]!;
    return { worst, split: maxDiff(sk.out, one), partials: sk.partials };
  }, [tile, splits]);

  return (
    <WidgetFrame
      testId="flash-attention"
      title="HBM traffic, and exact tiling"
      caption="Top: elements moved between HBM and the chip for one attention head, standard (FlashAttention's Algorithm 0) against tiled (Algorithm 1), at 16-bit precision and the simulator's derated H100 bandwidth (80% of 3.35 TB/s). The default SRAM is the paper's 192 KB per A100 multiprocessor. The tiled count grows with d² and shrinks with SRAM: try d = 256 with little SRAM. Bottom: real computations in your browser."
    >
      <div className="grid gap-3 sm:grid-cols-3">
        <Slider
          label="Sequence N"
          value={logN}
          min={9}
          max={16}
          onChange={setLogN}
          format={() => N.toLocaleString("en-GB")}
        />
        <Slider
          label="Head dim d"
          value={d}
          min={32}
          max={256}
          step={32}
          onChange={setD}
        />
        <Slider
          label="On-chip SRAM"
          value={sramKb}
          min={16}
          max={256}
          step={16}
          onChange={setSramKb}
          format={(v) => `${v} KB`}
        />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Standard"
          value={`${((std * BYTES) / 1e6).toFixed(1)} MB`}
          hint={`${(((std * BYTES) / bw) * 1e6).toFixed(1)} µs`}
        />
        <Stat
          label="Tiled"
          value={`${((fl * BYTES) / 1e6).toFixed(1)} MB`}
          hint={`${(((fl * BYTES) / bw) * 1e6).toFixed(1)} µs`}
        />
        <Stat label="Reduction" value={`${(std / fl).toFixed(1)}×`} />
        <Stat
          label="N×N matrix"
          value={`${((N * N * BYTES) / 1e6).toFixed(1)} MB`}
          hint="never written when tiled"
        />
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <div>
          <Slider
            label="Tile size (keys)"
            value={tile}
            min={1}
            max={48}
            onChange={setTile}
          />
          <p className="mt-2 text-sm" role="status">
            48 queries × 48 keys, d = 16: tiled vs textbook, max |Δ| ={" "}
            <strong className="font-mono">{demo.worst.toExponential(1)}</strong>
          </p>
        </div>
        <div>
          <Slider
            label="Split-K workers"
            value={splits}
            min={1}
            max={16}
            onChange={setSplits}
          />
          <p className="mt-2 text-sm" role="status">
            One query over 4,096 keys, merged from {splits} partials: max |Δ| ={" "}
            <strong className="font-mono">{demo.split.toExponential(1)}</strong>
          </p>
          <ul className="mt-2 grid grid-cols-2 gap-1 font-mono text-[0.65rem] text-neutral-600 dark:text-neutral-400">
            {demo.partials.slice(0, 8).map((p, i) => (
              <li key={i}>
                #{i}: m={p.m.toFixed(2)} l={p.l.toFixed(1)}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </WidgetFrame>
  );
}
