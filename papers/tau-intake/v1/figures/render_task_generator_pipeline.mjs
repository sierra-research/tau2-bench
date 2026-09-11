import { spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
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

const source = pathToFileURL(
  path.join(here, "task_generator_pipeline.html"),
).href;
const pdf = path.join(here, "task_generator_pipeline.pdf");
const vectorPdf = path.join(here, "task_generator_pipeline.vector.pdf");
const png = path.join(here, "task_generator_pipeline.png");

function runChrome(args) {
  const result = spawnSync(executablePath, args, { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(result.stderr || `Chrome exited with status ${result.status}`);
  }
}

runChrome([
  "--headless",
  "--disable-gpu",
  "--no-pdf-header-footer",
  `--print-to-pdf=${vectorPdf}`,
  source,
]);
runChrome([
  "--headless",
  "--disable-gpu",
  "--hide-scrollbars",
  "--force-device-scale-factor=2",
  "--window-size=1578,500",
  `--screenshot=${png}`,
  source,
]);

// Chrome represents CSS shadows and gradients as transparent image tiles.
// Flatten them so Quartz/Preview does not expose the tile boundaries.
const ghostscript = spawnSync(
  process.env.GS_PATH || "gs",
  [
    "-q",
    "-dSAFER",
    "-dBATCH",
    "-dNOPAUSE",
    "-sDEVICE=pdfwrite",
    "-dCompatibilityLevel=1.3",
    "-dPDFSETTINGS=/prepress",
    `-sOutputFile=${pdf}`,
    vectorPdf,
  ],
  { encoding: "utf8" },
);
if (ghostscript.status !== 0) {
  throw new Error(
    ghostscript.error?.message ||
      ghostscript.stderr ||
      `Ghostscript exited with status ${ghostscript.status}`,
  );
}
rmSync(vectorPdf, { force: true });
