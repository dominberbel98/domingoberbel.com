import React, { useState } from "react";

/**
 * The avatar rises out of the footer bar with its fists on its hips, swings
 * one arm up, gives a thumbs-up with a wink, and sinks back.
 *
 * The movement is hand-made animation: 16 drawn frames, played as a strip
 * (public/thumbs-up/frames.webp, built from data/thumbs-up-source.gif by
 * scripts/build_thumbs_up_frames.py). This component only loads the strip and
 * starts it; the rise, the frame stepping and the exit are CSS keyframes
 * (styles.css, "Thumbs-up"), so there is no animation library involved.
 *
 * Nothing moves until the strip has loaded, so a slow connection delays the
 * entrance instead of showing an empty box rising. It plays once per mount;
 * the parent re-keys it to play again and unmounts it on `onDone`.
 */
export default function ThumbsUp({ onDone }) {
  const [ready, setReady] = useState(false);

  return (
    <div className={`thumbs-up${ready ? " is-playing" : ""}`} aria-hidden="true">
      <div
        className="thumbs-figure"
        onAnimationEnd={(event) => {
          // thumbs-rise on wide screens, thumbs-rise-short elsewhere.
          if (event.animationName.startsWith("thumbs-rise")) onDone?.();
        }}
      >
        <div className="thumbs-window">
          <img
            className="thumbs-strip"
            src="/thumbs-up/frames.webp"
            alt=""
            draggable="false"
            onLoad={() => setReady(true)}
            // No figure is better than a broken one.
            onError={() => onDone?.()}
          />
        </div>
      </div>
    </div>
  );
}
