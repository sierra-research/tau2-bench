import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
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

const source = pathToFileURL(path.join(here, "adaptivity_effort.html")).href;
const pdf = path.join(here, "adaptivity_effort.pdf");
const png = path.join(here, "adaptivity_effort.png");

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
  `--print-to-pdf=${pdf}`,
  source,
]);
runChrome([
  "--headless",
  "--disable-gpu",
  "--hide-scrollbars",
  "--window-size=620,330",
  `--screenshot=${png}`,
  source,
]);
