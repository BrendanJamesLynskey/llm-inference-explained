/**
 * Brief 41's chapter heroes with their equations, for the chapters' MDX.
 * Server Components: KaTeX renders here, the animations load in the browser.
 * Highlight keys: the `hl-*` classes in globals.css.
 */
import { Eq } from "@/components/mdx/Eq";

import {
  LazyFlashTiles,
  LazyGenerationTimeline,
  LazyKvFill,
  LazyRooflineClimb,
} from "./lazy";

/** Chapter 1: a pass takes the longer of its arithmetic and its bytes. */
export function GenerationTimeline(): JSX.Element {
  return (
    <LazyGenerationTimeline
      equation={
        <Eq
          label="A forward pass takes the longer of its arithmetic time, FLOPs over the peak FLOP rate, and its memory time, bytes over the bandwidth, plus a fixed overhead. The first token comes after the prefill pass; every later token after one decode step."
          tex={String.raw`t_{\text{pass}} = \max\!\Big(\htmlClass{hl-p}{\tfrac{\text{FLOPs}}{P}},\ \htmlClass{hl-bw}{\tfrac{\text{bytes}}{B}}\Big) + t_0, \qquad \text{TTFT} = \htmlClass{hl-p}{t_{\text{prefill}}}, \quad \text{TPOT} = \htmlClass{hl-bw}{t_{\text{decode}}}`}
        />
      }
    />
  );
}

/** Chapter 2: the cache grows by one column; the new query reads it all. */
export function KvFill(): JSX.Element {
  return (
    <LazyKvFill
      equation={
        <Eq
          label="The cached keys up to step t are the earlier keys with the new key appended. The new token's attention weights are the softmax of its query against every cached key, scaled by the square root of the head width; its output is those weights times the cached values. The next token is the most likely one."
          tex={String.raw`K_{\le t} = \big[\,K_{<t};\ \htmlClass{hl-c}{k_t}\,\big], \quad \htmlClass{hl-a}{a_t = \operatorname{softmax}\!\big(q_t K_{\le t}^{\top} / \sqrt{d_h}\big)}, \quad o_t = a_t V_{\le t}, \quad \htmlClass{hl-p}{x_{t+1} = \arg\max \ell_t}`}
        />
      }
    />
  );
}

/** Chapter 3: attainable FLOP/s is the lower roof at the step's intensity. */
export function RooflineClimb(): JSX.Element {
  return (
    <LazyRooflineClimb
      equation={
        <Eq
          label="Attainable FLOP per second is the smaller of the peak and the intensity times the bandwidth. A decode step of n sequences of context c has intensity about two times the parameter count times n, over the weight bytes plus n times c times the KV bytes per token."
          tex={String.raw`\text{attainable} = \min\big(\htmlClass{hl-p}{P},\ I \cdot \htmlClass{hl-bw}{B}\big), \qquad I \approx \frac{2\,N_p\,n}{N_p\,\beta_w + \htmlClass{hl-c}{n\,c\,\beta_{kv}}}`}
        />
      }
    />
  );
}

/** Chapter 6: the online softmax that lets a tile be dropped. */
export function FlashTiles(): JSX.Element {
  return (
    <LazyFlashTiles
      equation={
        <Eq
          label="For each new tile of scores s, the running maximum becomes the larger of the old maximum and the tile's; the running sum is rescaled by e to the old minus new maximum and the tile's exponentials are added; the output is the weighted sum of values over the running sum."
          tex={String.raw`m' = \max\big(m, \max_j s_j\big), \qquad \htmlClass{hl-a}{l' = l\,e^{\,m - m'} + \textstyle\sum_j e^{\,s_j - m'}}, \qquad \htmlClass{hl-c}{o = \tfrac{1}{l}\textstyle\sum_j e^{\,s_j - m}\,v_j}`}
        />
      }
    />
  );
}
