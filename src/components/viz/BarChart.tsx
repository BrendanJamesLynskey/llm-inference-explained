"use client";

/**
 * Tiny SVG bar chart for a single 1-D vector. Positives go right, negatives
 * left, with the baseline in the middle.
 */
type Props = {
  values: number[];
  /** Pixel width of the chart. */
  width?: number;
  /** Pixel height *per bar* (the chart grows vertically with the data). */
  rowHeight?: number;
  /** Optional colour for positive bars. */
  positive?: string;
  /** Optional colour for negative bars. */
  negative?: string;
};

export function BarChart({
  values,
  width = 240,
  rowHeight = 8,
  positive = "rgb(99 102 241)",
  negative = "rgb(244 63 94)",
}: Props): JSX.Element {
  const n = values.length;
  let max = 0;
  for (const v of values) {
    const a = Math.abs(v);
    if (a > max) max = a;
  }
  if (max === 0) max = 1;
  const center = width / 2;
  const scale = (width - 8) / 2 / max;

  return (
    <svg
      width={width}
      height={n * rowHeight}
      role="img"
      aria-label={`Bar chart of ${n} values; max magnitude ${max.toFixed(2)}`}
      className="block max-w-full"
    >
      {values.map((v, i) => {
        const len = Math.abs(v) * scale;
        const x = v >= 0 ? center : center - len;
        return (
          <rect
            key={i}
            x={x}
            y={i * rowHeight + 1}
            width={len}
            height={rowHeight - 2}
            fill={v >= 0 ? positive : negative}
          >
            {/* One string, not `[{i}] {v}`: several text children inside an
                SVG <title> hydrate differently from the server HTML. */}
            <title>{`[${i}] ${v.toFixed(3)}`}</title>
          </rect>
        );
      })}
      <line
        x1={center}
        y1={0}
        x2={center}
        y2={n * rowHeight}
        stroke="rgba(120,120,120,0.5)"
        strokeWidth={1}
      />
    </svg>
  );
}
