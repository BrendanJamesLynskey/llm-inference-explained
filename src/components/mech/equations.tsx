/**
 * The equations beside the mechanism animations, copied verbatim from
 * Inference Trade-offs Explained's src/components/mdx/equations.tsx
 * (@239d29b) with the animations themselves (brief 41). Server Components:
 * KaTeX renders on the server.
 */
import { Eq } from "@/components/mdx/Eq";

/** How a request's time splits along the timeline. */
export function TimelineEq(): JSX.Element {
  return (
    <Eq
      label="End-to-end time is queueing plus prefill (together the time to first token), plus the KV hand-off, plus decode."
      tex={String.raw`t_{\text{finish}} - t_{\text{arrive}} = \underbrace{t_{\text{queue}} + t_{\text{prefill}}}_{\text{TTFT}} + t_{\text{hand-off}} + t_{\text{decode}}`}
    />
  );
}

/** Chapter 1: what one step may hold, and what a decode row waits. */
export function BatchEq(): JSX.Element {
  return (
    <Eq
      label="A step holds b decode rows plus prompt chunks c_i, within the token budget tau; a decode row's inter-token latency is the time of every step since its last token."
      tex={String.raw`\htmlClass{hl-a}{b} + \htmlClass{hl-p}{\textstyle\sum_i c_i} \le \htmlClass{hl-u}{\tau}, \qquad \text{ITL}_{\text{row}} = \sum_{\text{steps since its last token}} t_{\text{step}}\big(\htmlClass{hl-a}{b}, \htmlClass{hl-p}{\textstyle\sum_i c_i}\big)`}
    />
  );
}

/** Chapter 2: reserved against paged KV. */
export function KvEq(): JSX.Element {
  return (
    <Eq
      label="Reserved: each request holds prompt plus output tokens from admission. Paged: it holds the blocks its tokens so far fill. Admission and growth must fit the capacity C."
      tex={String.raw`\text{reserved: } \htmlClass{hl-w}{p + o} \qquad \text{paged: } \htmlClass{hl-p}{B\,\Big\lceil \tfrac{p + t}{B} \Big\rceil} \qquad \sum_{\text{requests}} \le \htmlClass{hl-c}{C}`}
    />
  );
}

/** Chapter 6: the hand-off time. */
export function HandoffEq(): JSX.Element {
  return (
    <Eq
      label="A hand-off of S bytes compressed by ratio r over a link of bandwidth B and latency l takes l plus S over r B, plus any wait for the link."
      tex={String.raw`t_{\text{hand-off}} = t_{\text{wait}} + \htmlClass{hl-t}{\ell} + \frac{\htmlClass{hl-b}{S} / \htmlClass{hl-c}{r}}{\htmlClass{hl-bw}{B}}`}
    />
  );
}

/** Chapter 8: the ring all-reduce. */
export function RingEq(): JSX.Element {
  return (
    <Eq
      label="A ring all-reduce of S bytes over n GPUs takes 2 times n minus 1 steps, each moving S over n bytes at bandwidth B plus the link latency."
      tex={String.raw`t_{\text{AR}} = \htmlClass{hl-n}{2(n-1)} \left( \frac{\htmlClass{hl-b}{S}}{n\,\htmlClass{hl-bw}{B}} + \htmlClass{hl-t}{\ell} \right)`}
    />
  );
}

/** Chapter 8: the pipeline bubble. */
export function GpipeEq(): JSX.Element {
  return (
    <Eq
      label="With p stages and m micro-batches a step takes m plus p minus 1 slots, of which p minus 1 per stage are idle: the bubble."
      tex={String.raw`\text{slots} = \htmlClass{hl-a}{m} + \htmlClass{hl-p}{p} - 1, \qquad \text{bubble} = \frac{\htmlClass{hl-p}{p} - 1}{\htmlClass{hl-a}{m} + \htmlClass{hl-p}{p} - 1}`}
    />
  );
}

/** Chapter 10: Leviathan et al.'s equation (1). */
export function SpecEq(): JSX.Element {
  return (
    <Eq
      label="With acceptance rate alpha and gamma drafts, a verify pass yields one minus alpha to the gamma plus one, over one minus alpha, tokens on average."
      tex={String.raw`\mathbb{E}[\text{tokens per pass}] = \frac{1 - \htmlClass{hl-a}{\alpha}^{\htmlClass{hl-n}{\gamma} + 1}}{1 - \htmlClass{hl-a}{\alpha}}`}
    />
  );
}
