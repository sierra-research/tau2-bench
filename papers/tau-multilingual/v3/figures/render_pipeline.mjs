import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
const executablePath = [
  process.env.CHROME_PATH,
  chromium.executablePath(),
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].find((candidate) => candidate && existsSync(candidate));

if (!executablePath) {
  throw new Error("Chrome or Chromium is required to render pipeline.pdf");
}

const browser = await chromium.launch({
  headless: true,
  executablePath,
});
const page = await browser.newPage({
  viewport: { width: 720, height: 700 },
  deviceScaleFactor: 2,
});
await page.goto(pathToFileURL(path.join(here, "pipeline.html")).href);
await page.pdf({
  path: path.join(here, "pipeline.pdf"),
  width: "720px",
  height: "700px",
  margin: { top: 0, right: 0, bottom: 0, left: 0 },
  printBackground: true,
  preferCSSPageSize: true,
});
await page.screenshot({
  path: path.join(here, "pipeline.png"),
  type: "png",
  fullPage: true,
});
await browser.close();
