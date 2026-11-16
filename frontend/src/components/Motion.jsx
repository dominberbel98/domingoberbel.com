import React, { useEffect, useState } from "react";

/**
 * Small motion helpers shared by every section.
 *
 * Each one renders its final text on the first frame and only then animates, so
 * nothing is ever left waiting on a script to become readable. With
 * `prefers-reduced-motion` they render the final text and do nothing else.
 *
 * The animated characters are `aria-hidden`; the real text sits beside them in
 * an `sr-only` span, so a screen reader never hears a half-decoded title or a
 * number on its way up.
 */

export const motionAllowed = () =>
  typeof window !== "undefined" &&
  !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const GLYPHS = "█▓▒░<>/\\_01#";

/** A title that resolves from terminal noise, left to right. */
export function DecodeText({ text, duration = 620 }) {
  const [shown, setShown] = useState(text);

  useEffect(() => {
    if (typeof text !== "string" || !motionAllowed()) {
      setShown(text);
      return undefined;
    }
    let frame = 0;
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const fixed = Math.floor(progress * text.length);
      let out = text.slice(0, fixed);
      for (let i = fixed; i < text.length; i += 1) {
        out += text[i] === " " ? " " : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
      }
      setShown(progress < 1 ? out : text);
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [text, duration]);

  if (typeof text !== "string") return text;
  return (
    <>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">{shown}</span>
    </>
  );
}

// A plain number with an optional non-numeric tail: "0.82", "75%", "12 ms".
// Anything else ("1,234", "n/a") is shown as it is, not animated.
const NUMERIC = /^(-?\d+(?:\.\d+)?)(\D*)$/;

/** A figure that counts up to its value when it appears or changes. */
export function CountUp({ value, duration = 900 }) {
  const final = String(value);
  const [shown, setShown] = useState(final);

  useEffect(() => {
    const match = NUMERIC.exec(final);
    if (!match || !motionAllowed()) {
      setShown(final);
      return undefined;
    }
    const target = Number(match[1]);
    const decimals = (match[1].split(".")[1] || "").length;
    const tail = match[2];
    let frame = 0;
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setShown(progress < 1 ? (target * eased).toFixed(decimals) + tail : final);
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [final, duration]);

  return (
    <>
      <span className="sr-only">{final}</span>
      <span aria-hidden="true">{shown}</span>
    </>
  );
}
