import React, { useEffect, useState } from "react";
import { useT } from "../i18n";

/**
 * The batch pipeline behind the dashboard, drawn as the path the data takes.
 *
 * The workflow runs every 30 minutes (.github/workflows/update-laliga.yml), so
 * the countdown is to the next half hour on the clock. GitHub starts scheduled
 * runs late under load, which is why the label says "≈".
 *
 * The pulses and the node glow are CSS (styles.css, "Pipeline strip") and only
 * run when motion is allowed; the countdown is information, so it always shows.
 */

const HALF_HOUR = 30 * 60 * 1000;
const msToNextRun = () => HALF_HOUR - (Date.now() % HALF_HOUR);

export default function PipelineStrip() {
  const tr = useT();
  const F = tr.laliga.flow;
  const [left, setLeft] = useState(msToNextRun);

  useEffect(() => {
    const id = setInterval(() => setLeft(msToNextRun()), 1000);
    return () => clearInterval(id);
  }, []);

  const minutes = String(Math.floor(left / 60000)).padStart(2, "0");
  const seconds = String(Math.floor((left % 60000) / 1000)).padStart(2, "0");
  const nodes = [
    ["API", F.api],
    ["PySpark", F.spark],
    ["JSON", F.json],
    ["Web", F.web],
  ];

  return (
    <div className="viz-panel mb-4">
      <h3 className="viz-title">
        <span className="material-symbols-outlined text-sm mr-2">account_tree</span>
        {F.label}
      </h3>
      <div className="pipe">
        <div className="pipe-track" aria-hidden="true">
          <span className="pipe-pulse" />
          <span className="pipe-pulse pipe-pulse--late" />
        </div>
        {nodes.map(([name, detail], index) => (
          <div key={name} className="pipe-node" style={{ "--n": index }}>
            <i aria-hidden="true" />
            <b className="text-[0.8rem] font-headline uppercase text-[#00FF41]">{name}</b>
            <span className="hidden sm:block text-[0.74rem] text-[#00FF41]/65 normal-case leading-snug px-1">
              {detail}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap justify-between gap-x-4 gap-y-1 text-[0.74rem] font-headline uppercase text-[#00FF41]/65">
        <span>{F.schedule}</span>
        <span>
          {F.next}{" "}
          <b className="text-[#FFD700] tabular-nums">
            {minutes}:{seconds}
          </b>
        </span>
      </div>
    </div>
  );
}
