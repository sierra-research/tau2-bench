#!/usr/bin/env node
// Prerender the built SPA: one static HTML file per route, with that route's
// content and meta tags baked in, so crawlers and unfurl bots see real pages.
//
//   node scripts/prerender.mjs           # prerender dist/ in place
//   node scripts/prerender.mjs --serve   # just serve dist/ with GitHub Pages
//                                        # semantics (for e2e / manual testing)
//
// How it works: serve dist/ locally, load each route from src/routes.js in
// headless Chrome (--dump-dom serializes the DOM after React renders and
// App.jsx's applyPageMeta sets the head tags), validate the snapshot against
// per-route content guards, and write it to dist/<route>/index.html. The
// original index.html is copied to 404.html first, so unknown paths still
// boot the SPA (GitHub Pages serves 404.html for paths with no file).
//
// The guards are load-bearing: the leaderboard fetches live submission data
// while being snapshotted, and a network flake would otherwise deploy blank
// pages that look like a successful build.

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { PAGE_META, ROUTES } from '../src/routes.js'

const DIST = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../dist')
const VIRTUAL_TIME_BUDGET_MS = 20000

// Per-route sanity checks applied to each snapshot. `required` patterns must
// all match; `forbidden` patterns must not.
const GUARDS = {
  '/': {
    required: [/preview-table-wrapper/, /τ³-Banking/, /How τ-bench has evolved/],
    forbidden: [/Loading leaderboard/],
  },
  '/leaderboard': {
    required: [/τ³-Banking Leaderboard/, /(<tr[\s>].*?){5,}/s],
    forbidden: [/Loading leaderboard/],
  },
  '/progress': {
    required: [/id="progress"/],
    forbidden: [/Loading leaderboard/],
  },
  '/trajectory-visualizer': {
    required: [/τ-bench Visualizer/],
    forbidden: [],
    // Small by design: header + selectors, no data until a model is chosen.
    minBytes: 6000,
  },
  '/blog': {
    required: [/τ-voice/i],
    forbidden: [],
  },
}

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.woff2': 'font/woff2',
  '.txt': 'text/plain', '.xml': 'application/xml', '.webmanifest': 'application/json',
}

const resolveFile = (urlPath) => {
  const clean = path.normalize(decodeURIComponent(urlPath)).replace(/^(\.\.[/\\])+/, '')
  let filePath = path.join(DIST, clean)
  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, 'index.html')
  }
  return fs.existsSync(filePath) && fs.statSync(filePath).isFile() ? filePath : null
}

// Static server. `routeFallback` controls what happens for known-route paths
// with no file on disk yet:
//   true  → serve index.html (needed during the prerender pass itself)
//   false → GitHub Pages semantics: 404.html with a 404 status
const createServer = (routeFallback) =>
  http.createServer((req, res) => {
    const urlPath = new URL(req.url, 'http://localhost').pathname
    let filePath = resolveFile(urlPath)

    if (!filePath && routeFallback && ROUTES[urlPath.replace(/\/$/, '') || '/']) {
      filePath = path.join(DIST, 'index.html')
    }

    if (!filePath) {
      const notFound = path.join(DIST, '404.html')
      res.writeHead(404, { 'Content-Type': 'text/html' })
      res.end(fs.existsSync(notFound) ? fs.readFileSync(notFound) : 'Not found')
      return
    }

    res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' })
    res.end(fs.readFileSync(filePath))
  })

const listen = (server, port) =>
  new Promise((resolve) => server.listen(port, '127.0.0.1', () => resolve(server.address().port)))

const findChrome = () => {
  if (process.env.CHROME_BIN) return process.env.CHROME_BIN
  const candidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
  ]
  const found = candidates.find((c) => fs.existsSync(c))
  if (!found) throw new Error('Chrome not found; set CHROME_BIN')
  return found
}

// NOTE: must be async — the dist server runs in this same process, so a
// synchronous exec would block the event loop and deadlock Chrome's requests.
const dumpDom = async (chrome, url) => {
  const { stdout } = await execFileAsync(
    chrome,
    [
      '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-sandbox',
      `--virtual-time-budget=${VIRTUAL_TIME_BUDGET_MS}`, '--dump-dom', url,
    ],
    { maxBuffer: 64 * 1024 * 1024, encoding: 'utf8', timeout: 120000 }
  )
  return stdout
}

const checkGuards = (route, html) => {
  const failures = []
  const guard = GUARDS[route] || { required: [], forbidden: [] }

  const view = ROUTES[route]
  const expectedTitle = PAGE_META[view]?.title
  if (expectedTitle && !html.includes(`<title>${expectedTitle}</title>`)) {
    failures.push(`missing <title>${expectedTitle}</title>`)
  }
  for (const re of guard.required) {
    if (!re.test(html)) failures.push(`missing required pattern ${re}`)
  }
  for (const re of guard.forbidden) {
    if (re.test(html)) failures.push(`contains forbidden pattern ${re}`)
  }
  const minBytes = guard.minBytes ?? 10000
  if (html.length < minBytes) failures.push(`suspiciously small snapshot (${html.length} bytes)`)
  return failures
}

const prerender = async () => {
  if (!fs.existsSync(path.join(DIST, 'index.html'))) {
    throw new Error(`${DIST}/index.html not found — run \`npm run build\` first`)
  }

  // The pristine SPA shell doubles as the 404 fallback: it boots the app,
  // which renders the right view for any path (or home for unknown ones).
  fs.copyFileSync(path.join(DIST, 'index.html'), path.join(DIST, '404.html'))

  const server = createServer(true)
  const port = await listen(server, 0)
  const chrome = findChrome()

  const errors = []
  const snapshots = {}
  try {
    for (const route of Object.keys(ROUTES)) {
      const url = `http://127.0.0.1:${port}${route}`
      process.stdout.write(`prerendering ${route} ... `)
      let html = await dumpDom(chrome, url)
      if (!/^<!doctype html>/i.test(html)) html = `<!DOCTYPE html>\n${html}`

      const failures = checkGuards(route, html)
      if (failures.length > 0) {
        console.log('FAIL')
        errors.push(`${route}: ${failures.join('; ')}`)
        continue
      }
      snapshots[route] = html
      console.log(`ok (${(html.length / 1024).toFixed(0)} KB)`)
    }
  } finally {
    server.close()
  }

  if (errors.length > 0) {
    console.error('\nPrerender guard failures (refusing to write partial output):')
    for (const e of errors) console.error(`  - ${e}`)
    process.exit(1)
  }

  // All snapshots passed; write them out. '/' overwrites index.html itself.
  for (const [route, html] of Object.entries(snapshots)) {
    const outFile =
      route === '/' ? path.join(DIST, 'index.html') : path.join(DIST, route.slice(1), 'index.html')
    fs.mkdirSync(path.dirname(outFile), { recursive: true })
    fs.writeFileSync(outFile, html)
  }
  console.log(`\nPrerendered ${Object.keys(snapshots).length} routes into ${DIST}`)
}

const serveOnly = async () => {
  const port = Number(process.env.PORT || 4173)
  await listen(createServer(false), port)
  console.log(`Serving ${DIST} at http://127.0.0.1:${port} (GitHub Pages semantics, Ctrl-C to stop)`)
}

if (process.argv.includes('--serve')) {
  serveOnly()
} else {
  prerender().catch((err) => {
    console.error(err.message)
    process.exit(1)
  })
}
