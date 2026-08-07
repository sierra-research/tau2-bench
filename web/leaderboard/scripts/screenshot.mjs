// Quick screenshot helper for local dev review.
//
// playwright is intentionally not a devDependency (only needed for this
// helper). Before first use, run: npm install --no-save playwright
//
// Usage: node scripts/screenshot.mjs <url> <outfile> [fullpage]
import { chromium } from 'playwright'

const [url, outfile, fullpage] = process.argv.slice(2)
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } })
await page.goto(url, { waitUntil: 'networkidle' })
await page.waitForTimeout(1000)
await page.screenshot({ path: outfile, fullPage: fullpage === 'fullpage' })
await browser.close()
console.log(`saved ${outfile}`)
