/**
 * The panel every interactive sits in: the same border, background and
 * padding as transformer-explainer's widgets (e.g. `SamplingWidget`).
 */
import type { ReactNode } from "react";

export function WidgetFrame({
  title,
  caption,
  children,
  testId,
}: {
  title: string;
  /** One line under the title: what the widget computes, and how. */
  caption?: ReactNode;
  children: ReactNode;
  testId?: string;
}): JSX.Element {
  return (
    <figure
      data-testid={testId}
      className="my-8 min-w-0 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900"
    >
      <figcaption>
        <p className="font-mono text-[0.65rem] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
          Interactive
        </p>
        <p className="mt-1 font-semibold text-neutral-900 dark:text-neutral-100">
          {title}
        </p>
        {caption && (
          <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
            {caption}
          </p>
        )}
      </figcaption>
      <div className="mt-4">{children}</div>
    </figure>
  );
}
