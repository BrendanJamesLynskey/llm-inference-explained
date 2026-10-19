/**
 * The Inference track's start page (/inference, brief 44): the four sites
 * in reading order and a glossary whose terms link to the chapter section
 * that defines each one, on whichever site defines it.
 * tests/unit/inference-track.test.ts checks that every link on this site
 * names a chapter and an anchor that exist; the other sites' anchors are
 * checked by their own learning-layer tests.
 */
import type { TrackSite } from "@/components/learn/ReadingOrder";

export const SERVING = "https://llm-inference-explained.vercel.app";
export const ARCHITECTURES = "https://llm-architectures-explained.vercel.app";
export const NUMERICS = "https://numerics-explained.vercel.app";
export const TRADEOFFS = "https://inference-tradeoffs-explained.vercel.app";
export const FOUNDATIONS =
  "https://transformer-decoder-explained.vercel.app/foundations";
export const HARDWARE = "https://gpu-kernels-explained.vercel.app/hardware";

export const INFERENCE_SITES: readonly TrackSite[] = [
  {
    key: "serving",
    name: "Serving",
    url: `${SERVING}/learn`,
    why: "how a model is served: the generation loop, the KV cache, the roofline, batching, paging, kernels, speculation, quantisation, parallelism and disaggregation; assumes the Foundations track",
  },
  {
    key: "architectures",
    name: "Architectures",
    url: `${ARCHITECTURES}/learn`,
    why: "how modern models differ from the textbook decoder (attention variants, RoPE, norms, MoE, long context, MTP, CED) and what each costs to serve; assumes Serving's KV cache and roofline",
  },
  {
    key: "numerics",
    name: "Numerics",
    url: `${NUMERICS}/learn`,
    why: "the number formats under it all, from bits and rounding to quantisation, outliers, GPTQ, AWQ and NF4, the KV cache and hardware; assumes Serving's quantisation chapter",
  },
  {
    key: "tradeoffs",
    name: "Trade-offs",
    url: `${TRADEOFFS}/learn`,
    why: "every serving lever measured on five workloads against goodput, latency, cost and energy, one chapter per lever; assumes the three before it",
  },
];

export type GlossaryTerm = { term: string; href: string; what: string };

const i = (p: string) => `${SERVING}/learn/${p}`;
const h = (p: string) => `${ARCHITECTURES}/learn/${p}`;
const n = (p: string) => `${NUMERICS}/learn/${p}`;
const t = (p: string) => `${TRADEOFFS}/learn/${p}`;

export const INFERENCE_GLOSSARY: readonly GlossaryTerm[] = [
  {
    term: "Prefill, decode",
    href: i("01-generation-loop#i-two-phases"),
    what: "one pass over the whole prompt, then one pass per output token",
  },
  {
    term: "KV cache",
    href: i("02-kv-cache#i-why-cache"),
    what: "every earlier token's keys and values, kept so decode steps need not recompute them",
  },
  {
    term: "GQA, MQA, MLA",
    href: i("02-kv-cache#i-shrink"),
    what: "sharing or compressing key/value heads to shrink the cache",
  },
  {
    term: "Roofline, arithmetic intensity, ridge point",
    href: i("03-roofline#i-roofline"),
    what: "a step takes the longer of its arithmetic and its memory time; the ridge is where they balance",
  },
  {
    term: "Memory-bound decode",
    href: i("03-roofline#i-decode"),
    what: "a batch-1 decode step reads every weight to make one token",
  },
  {
    term: "Continuous batching",
    href: i("04-batching#i-continuous"),
    what: "scheduling every step, so a finished sequence's slot is reused at once",
  },
  {
    term: "Chunked prefill",
    href: i("04-batching#i-chunked"),
    what: "a per-step token budget: decodes first, then a slice of a prompt",
  },
  {
    term: "PagedAttention, prefix sharing",
    href: i("05-memory-management#i-paging"),
    what: "KV memory in fixed-size blocks with a block table per sequence",
  },
  {
    term: "FlashAttention, online softmax",
    href: i("06-attention-kernels#i-flash"),
    what: "tiled attention that never writes the score matrix",
  },
  {
    term: "Speculative decoding",
    href: i("07-speculative-decoding#i-why-spec"),
    what: "a draft guesses tokens; one target pass checks them all",
  },
  {
    term: "Weight and KV quantisation",
    href: i("08-quantisation#i-three"),
    what: "fewer bytes per step: weights only, weights and activations, or the cache",
  },
  {
    term: "Tensor, pipeline and expert parallelism",
    href: i("09-parallelism#i-why-split"),
    what: "splitting a model over GPUs, and the messages each split adds",
  },
  {
    term: "TTFT, TPOT, ITL",
    href: i("10-serving-metrics#i-metrics"),
    what: "time to first token, time per output token, every inter-token gap",
  },
  {
    term: "Goodput",
    href: i("10-serving-metrics#i-metrics"),
    what: "requests per second that met both latency targets",
  },
  {
    term: "Little's law",
    href: i("10-serving-metrics#i-little"),
    what: "requests in the system = arrival rate × time in the system",
  },
  {
    term: "Disaggregated serving",
    href: i("11-why-disaggregate#i-conflict"),
    what: "prefill and decode on separate GPU pools",
  },
  {
    term: "KV hand-off",
    href: i("12-moving-the-kv-cache#i-cargo"),
    what: "each prompt's cache crossing the link from the prefill to the decode pool",
  },
  {
    term: "Sliding window, sparse and linear attention",
    href: h("01-attention#h-mixers"),
    what: "token mixers that read or keep less than the whole cache",
  },
  {
    term: "RoPE and its base θ",
    href: h("02-positional-encoding#h-theta"),
    what: "rotating query-key pairs so scores depend only on distance",
  },
  {
    term: "Pre-norm, post-LN, QK-norm",
    href: h("03-normalisation#h-stream"),
    what: "where the normalisation sits on the residual stream",
  },
  {
    term: "Mixture of experts",
    href: h("04-dense-and-moe#h-moe"),
    what: "many expert FFNs and a router: total and active parameters come apart",
  },
  {
    term: "Depth and width",
    href: h("05-depth-and-width#h-cost"),
    what: "the cache and attention's share grow with depth, embeddings with width",
  },
  {
    term: "Long-context cost",
    href: h("06-long-context#h-afford"),
    what: "with full attention every token adds cache and reads all of it",
  },
  {
    term: "Multi-token prediction",
    href: h("07-multi-token-prediction#h-mtp"),
    what: "extra heads that predict further ahead, and draft at inference",
  },
  {
    term: "Causal encoder-decoder (CED)",
    href: h("09-encoder-decoder-and-ced#h-ced"),
    what: "prompt tokens run only the encoder half: cheaper prefill",
  },
  {
    term: "Floating-point format, binade",
    href: n("01-bits-to-numbers#n-fields"),
    what: "sign, exponent and mantissa; relative precision about constant",
  },
  {
    term: "Round to nearest, stochastic rounding",
    href: n("02-rounding#n-rne"),
    what: "the default rounding and the one that is exact on average",
  },
  {
    term: "Accumulation, Kahan summation",
    href: n("03-accumulation#n-fixes"),
    what: "why long sums stall in low precision, and the fixes",
  },
  {
    term: "FP8, MX formats",
    href: n("04-formats-zoo#n-mx"),
    what: "narrow formats, and blocks of elements sharing a scale",
  },
  {
    term: "Absmax, zero point, group size",
    href: n("05-quantisation-basics#n-granularity"),
    what: "integer quantisation with a scale per group",
  },
  {
    term: "Outliers, LLM.int8(), SmoothQuant",
    href: n("06-outliers#n-input"),
    what: "a few huge activation channels, and two ways around them",
  },
  {
    term: "GPTQ",
    href: n("07-gptq#n-rtn"),
    what: "quantise column by column, compensating the error in the rest",
  },
  {
    term: "AWQ, NF4",
    href: n("08-awq-and-nf4#n-awq"),
    what: "protecting the weights large activations meet; codes at normal quantiles",
  },
  {
    term: "KV-cache quantisation",
    href: n("09-kv-cache#n-kv"),
    what: "fewer bits per cached key and value",
  },
  {
    term: "Narrow numbers in hardware",
    href: n("10-hardware#n-multiplying"),
    what: "what bits cost to store, move and multiply",
  },
  {
    term: "Batching policy",
    href: t("01-batching-and-chunked-prefill#mechanism"),
    what: "prefill-priority, decode-priority or chunked prefill, measured",
  },
  {
    term: "Prefix caching",
    href: t("03-prefix-caching#mechanism"),
    what: "compute a shared prompt prefix once",
  },
  {
    term: "The pool split",
    href: t("04-disaggregation#mechanism"),
    what: "what disaggregation does to goodput, measured on five workloads",
  },
  {
    term: "Combining levers",
    href: t("13-combining-levers#why"),
    what: "why levers compound, and where they do not",
  },
];
