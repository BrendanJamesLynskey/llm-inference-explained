"use client";

/**
 * A data-driven transition: whenever `key` changes (a new workload, a new
 * SLO filter, new axes), `t` runs from 0 to 1 over `ms` on a
 * requestAnimationFrame clock, and the widget draws `frame(from, to, t)`, a
 * pure function of the two model states. With `prefers-reduced-motion:
 * reduce` the transition is skipped (t = 1 at once). Returns the previous
 * target as `from` (null on the first render).
 */
import { useCallback, useEffect, useRef, useState } from "react";

function reducedMotion(): boolean {
  return (
    typeof window === "undefined" ||
    !window.matchMedia ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function useTween<T>(
  target: T,
  key: string,
  ms = 700,
): { from: T | null; to: T; t: number } {
  const [state, setState] = useState<{ from: T | null; key: string }>({
    from: null,
    key,
  });
  const [t, setT] = useState(1);
  // the target the last transition ran to (updated after each render)
  const last = useRef(target);

  useEffect(() => {
    if (key === state.key) {
      last.current = target;
      return;
    }
    setState({ from: last.current, key });
    last.current = target;
    if (reducedMotion()) {
      setT(1);
      return;
    }
    setT(0);
    let raf = 0;
    const start = performance.now();
    const loop = (now: number) => {
      const u = Math.min(1, (now - start) / ms);
      setT(u);
      if (u < 1) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, target]);

  // the render in which the key has just changed: start from the old target
  if (key !== state.key)
    return { from: last.current, to: target, t: reducedMotion() ? 1 : 0 };
  return { from: state.from, to: target, t };
}

/** The width of an element, followed with a ResizeObserver. */
export function useWidth(
  fallback = 640,
): [(el: HTMLElement | null) => void, number] {
  const [w, setW] = useState(fallback);
  const obs = useRef<ResizeObserver | null>(null);
  const ref = useCallback(
    (el: HTMLElement | null) => {
      obs.current?.disconnect();
      if (!el || typeof ResizeObserver === "undefined") return;
      setW(Math.round(el.getBoundingClientRect().width) || fallback);
      obs.current = new ResizeObserver((entries) => {
        const cw = entries[0]?.contentRect.width;
        if (cw) setW(Math.round(cw));
      });
      obs.current.observe(el);
    },
    [fallback],
  );
  return [ref, w];
}
