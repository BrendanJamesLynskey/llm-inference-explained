/**
 * Cross-site navigation between the two companion sites: the Transformer
 * Decoder Explainer and LLM Inference Explained. The same component, with
 * the same classes, sits in both sites' headers; only `current` differs.
 *
 * Server Component (plain links, no state).
 */

const SITES = [
  {
    key: "decoder",
    label: "Decoder",
    href: "https://transformer-decoder-explained.vercel.app",
    title: "Transformer Decoder Explainer: how one forward pass works",
  },
  {
    key: "inference",
    label: "Inference",
    href: "https://llm-inference-explained.vercel.app",
    title: "LLM Inference Explained: how real systems generate text",
  },
] as const;

export function SiteSwitch({
  current,
}: {
  current: (typeof SITES)[number]["key"];
}): JSX.Element {
  return (
    <nav
      aria-label="Companion sites"
      className="inline-flex items-center rounded-md border border-neutral-300 text-xs dark:border-neutral-700"
    >
      {SITES.map((s, i) => (
        <span key={s.key} className="inline-flex items-center">
          {i > 0 && (
            <span aria-hidden className="text-neutral-400">
              ·
            </span>
          )}
          <a
            href={s.href}
            title={s.title}
            aria-current={s.key === current ? "true" : undefined}
            className={`focus-ring rounded-md px-2.5 py-1 ${
              s.key === current
                ? "bg-accent font-medium text-accent-fg"
                : "text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
            }`}
          >
            {s.label}
          </a>
        </span>
      ))}
    </nav>
  );
}
