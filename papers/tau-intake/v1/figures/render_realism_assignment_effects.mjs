import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const executablePath = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].find((candidate) => candidate && existsSync(candidate));

if (!executablePath) {
  throw new Error("Chrome or Chromium is required to render the figure");
}

const analysisPaths = new Map([
  [
    "agent_directed",
    "../analysis/intake_realism_effects_agent_directed_2026-09-11.json",
  ],
  ["scaffolded", "../analysis/intake_realism_effects_scaffolded_2026-09-11.json"],
]);
const analyses = new Map(
  [...analysisPaths].map(([arm, relativePath]) => {
    const analysis = JSON.parse(readFileSync(path.join(here, relativePath), "utf8"));
    if (analysis.arm !== arm) {
      throw new Error(`Expected ${arm} analysis, got ${analysis.arm}`);
    }
    return [arm, analysis];
  }),
);

const labels = new Map([
  ["Any caller realism", "Any realism"],
  ["Wrong-field answer", "Wrong-field"],
  ["Spelling variation", "Spelling variation"],
  ["Self-correction", "Self-correction"],
  ["Mispronunciation", "Mispronunciation"],
]);
const series = [
  {
    arm: "agent_directed",
    label: "Agent-directed",
    intervalClass: "agent-interval",
    markerClass: "agent-marker",
    offset: -12,
    marker: "circle",
  },
  {
    arm: "scaffolded",
    label: "Scaffolded",
    intervalClass: "scaffold-interval",
    markerClass: "scaffold-marker",
    offset: 12,
    marker: "square",
  },
];
const effects = [...labels].map(([sourceLabel, displayLabel]) => {
  const byArm = new Map(
    series.map(({ arm }) => {
      const effect = analyses
        .get(arm)
        .effects.find((candidate) => candidate.label === sourceLabel);
      if (!effect) {
        throw new Error(`Missing ${sourceLabel} in ${arm} analysis`);
      }
      return [arm, effect];
    }),
  );
  return { displayLabel, byArm };
});

const x = (value) => 409 + value * 5.56;
const signed = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
const rows = effects
  .map((effect, index) => {
    const y = 62 + index * 52;
    const labelClass = index === 0 ? ' class="summary"' : "";
    const seriesRows = series
      .map((spec) => {
        const estimate = effect.byArm.get(spec.arm);
        const seriesY = y + spec.offset;
        const [low, high] = estimate.interval_95_points;
        const pointX = x(estimate.effect_points);
        const lowX = x(low);
        const highX = x(high);
        const valueX = Math.min(highX + 8, 574);
        const marker =
          spec.marker === "circle"
            ? `<circle class="${spec.markerClass}" cx="${pointX}" cy="${seriesY}" r="6"/>`
            : `<rect class="${spec.markerClass}" x="${pointX - 6}" y="${seriesY - 6}" width="12" height="12"/>`;
        return `
    <g aria-label="${spec.label}, ${signed(estimate.effect_points)} points, 95 percent interval ${signed(low)} to ${signed(high)}">
      <line class="${spec.intervalClass}" x1="${lowX}" y1="${seriesY}" x2="${highX}" y2="${seriesY}"/>
      <line class="${spec.intervalClass} cap" x1="${lowX}" y1="${seriesY - 6}" x2="${lowX}" y2="${seriesY + 6}"/>
      <line class="${spec.intervalClass} cap" x1="${highX}" y1="${seriesY - 6}" x2="${highX}" y2="${seriesY + 6}"/>
      ${marker}
      <text class="value" x="${valueX}" y="${seriesY + 7}">${signed(estimate.effect_points)}</text>
    </g>`;
      })
      .join("\n");
    const divider =
      index === 0
        ? '<line x1="8" y1="89" x2="612" y2="89" stroke="#bbb" stroke-width="1.5"/>'
        : "";
    return `
  <g aria-label="${effect.displayLabel}">
    <text${labelClass} x="228" y="${y + 7}" text-anchor="end">${effect.displayLabel}</text>
${seriesRows}
  </g>${divider ? `\n  ${divider}` : ""}`;
  })
  .join("\n");

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  @page { size: 620px 384px; margin: 0; }
  html, body { width: 620px; height: 384px; margin: 0; background: #fff; }
  svg { display: block; width: 620px; height: 384px; font-family: "Times New Roman", Times, serif; }
  text { fill: #111; font-size: 23px; }
  .summary { font-weight: 700; }
  .grid { stroke: #dedede; stroke-width: 1; }
  .zero { stroke: #444; stroke-width: 2; }
  .agent-interval { stroke: #2b78b8; stroke-width: 3.5; }
  .scaffold-interval { stroke: #e87519; stroke-width: 3.5; stroke-dasharray: 8 5; }
  .cap { stroke-width: 2.5; stroke-dasharray: none; }
  .agent-marker { fill: #2b78b8; }
  .scaffold-marker { fill: #fff; stroke: #e87519; stroke-width: 3; }
  .axis { stroke: #555; stroke-width: 1.5; }
  .axis-title { font-size: 23px; }
  .value { fill: #111; }
</style>
</head>
<body>
<svg viewBox="0 0 620 384" role="img" aria-labelledby="title desc">
  <title id="title">Effect of assigning caller realisms by prompting arm</title>
  <desc id="desc">Horizontal forest plot comparing agent-directed and scaffolded percentage-point changes in exact task success. Every estimated 95 percent interval crosses zero.</desc>

  <g aria-label="Legend">
    <line class="agent-interval" x1="10" y1="17" x2="42" y2="17"/>
    <circle class="agent-marker" cx="26" cy="17" r="6"/>
    <text x="49" y="24">Agent-directed</text>
    <line class="scaffold-interval" x1="190" y1="17" x2="222" y2="17"/>
    <rect class="scaffold-marker" x="200" y="11" width="12" height="12"/>
    <text x="229" y="24">Scaffolded</text>
  </g>
  <text x="380" y="24" text-anchor="middle">hurts</text>
  <text x="490" y="24" text-anchor="middle">helps</text>

  <g aria-hidden="true">
    <line class="grid" x1="298" y1="34" x2="298" y2="310"/>
    <line class="grid" x1="353" y1="34" x2="353" y2="310"/>
    <line class="zero" x1="409" y1="34" x2="409" y2="310"/>
    <line class="grid" x1="465" y1="34" x2="465" y2="310"/>
    <line class="grid" x1="520" y1="34" x2="520" y2="310"/>
    <line class="axis" x1="270" y1="310" x2="520" y2="310"/>
    <line class="axis" x1="298" y1="310" x2="298" y2="317"/>
    <line class="axis" x1="353" y1="310" x2="353" y2="317"/>
    <line class="axis" x1="409" y1="310" x2="409" y2="317"/>
    <line class="axis" x1="465" y1="310" x2="465" y2="317"/>
    <line class="axis" x1="520" y1="310" x2="520" y2="317"/>
    <text x="298" y="338" text-anchor="middle">-20</text>
    <text x="353" y="338" text-anchor="middle">-10</text>
    <text x="409" y="338" text-anchor="middle">0</text>
    <text x="465" y="338" text-anchor="middle">+10</text>
    <text x="520" y="338" text-anchor="middle">+20</text>
    <text class="axis-title" x="410" y="368" text-anchor="middle">Exact-success change (percentage points)</text>
  </g>
${rows}
</svg>
</body>
</html>
`;

const htmlPath = path.join(here, "realism_assignment_effects.html");
const pdfPath = path.join(here, "realism_assignment_effects.pdf");
const pngPath = path.join(here, "realism_assignment_effects.png");
writeFileSync(htmlPath, html);

function runChrome(args) {
  const result = spawnSync(executablePath, args, { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(result.stderr || `Chrome exited with status ${result.status}`);
  }
}

const source = pathToFileURL(htmlPath).href;
runChrome([
  "--headless",
  "--disable-gpu",
  "--no-pdf-header-footer",
  `--print-to-pdf=${pdfPath}`,
  source,
]);
runChrome([
  "--headless",
  "--disable-gpu",
  "--hide-scrollbars",
  "--window-size=620,384",
  `--screenshot=${pngPath}`,
  source,
]);
