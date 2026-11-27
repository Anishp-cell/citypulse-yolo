// Non-component helpers shared by the officer console pages.
import { useEffect, useRef, useState } from 'react';

/** Hazard statuses in lifecycle order (rejected is left out: it is not a stage, it is an exit). */
export const PIPELINE = ['pending_review', 'open', 'filed', 'acknowledged', 'in_progress', 'fixed_unverified', 'fixed'];

export const reducedMotion = () => Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);

/** Animates a number towards `value` (from its previous value, 0 at first); no animation when motion is reduced. */
export function useCountUp(value, duration = 650) {
  const animate = typeof value === 'number' && !reducedMotion();
  const [shown, setShown] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    if (!animate) return undefined;
    let raf;
    const start = performance.now();
    const begin = from.current;
    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const v = begin + (value - begin) * (1 - (1 - p) ** 3);
      from.current = v;
      setShown(v);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, animate, duration]);
  return animate ? shown : value;
}

export const initials = (user) => (user?.full_name || user?.email || '?')
  .split(/[\s@.]+/).filter(Boolean).slice(0, 2).map((s) => s[0].toUpperCase()).join('');
