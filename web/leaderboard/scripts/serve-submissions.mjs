// Serves submission data for local development the way S3 will once staged
// viewer data is uploaded: files from a staging tree written by
// `python -m tau2.scripts.leaderboard.viewer_data` (gzipped JSON, served with
// Content-Encoding: gzip), and everything else proxied from the public bucket.
//
//   node scripts/serve-submissions.mjs [--staging DIR] [--port 8787]
//   VITE_SUBMISSIONS_BASE_URL=http://localhost:8787/submissions npm run dev

import { createReadStream, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { homedir } from 'node:os'
import { join, normalize } from 'node:path'
import { Readable } from 'node:stream'
import { parseArgs } from 'node:util'

const { values } = parseArgs({
  options: {
    staging: { type: 'string', default: join(homedir(), '.cache', 'tau2', 'viewer-staging') },
    port: { type: 'string', default: '8787' },
  },
})
const ORIGIN = 'https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com'
const PREFIX = '/submissions/'
const PASS_HEADERS = ['content-type', 'content-encoding', 'content-length', 'cache-control', 'etag', 'last-modified', 'accept-ranges', 'content-range']

const server = createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Headers', 'Range')
  res.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range')
  if (req.method === 'OPTIONS') return res.end()

  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname)
  if (path.startsWith(PREFIX)) {
    const local = normalize(join(values.staging, path.slice(PREFIX.length)))
    if (local.startsWith(values.staging)) {
      try {
        const stat = statSync(local)
        if (stat.isFile()) {
          res.writeHead(200, {
            'Content-Type': 'application/json',
            'Content-Encoding': 'gzip',
            'Content-Length': stat.size,
            'Cache-Control': path.includes('/simulations/') ? 'public, max-age=31536000, immutable' : 'public, max-age=3600',
            'X-Served-From': 'staging',
          })
          return createReadStream(local).pipe(res)
        }
      } catch { /* not staged: proxy */ }
    }
  }

  try {
    const upstream = await fetch(`${ORIGIN}${req.url}`, {
      headers: req.headers.range ? { range: req.headers.range } : {},
    })
    const headers = { 'X-Served-From': 's3' }
    for (const name of PASS_HEADERS) {
      const value = upstream.headers.get(name)
      if (value) headers[name] = value
    }
    // fetch() decodes gzip bodies; drop the stale encoding/length if it did.
    if (headers['content-encoding']) { delete headers['content-encoding']; delete headers['content-length'] }
    res.writeHead(upstream.status, headers)
    if (upstream.body) Readable.fromWeb(upstream.body).pipe(res)
    else res.end()
  } catch (error) {
    res.writeHead(502).end(String(error))
  }
})

server.listen(Number(values.port), () => {
  console.log(`submissions: http://localhost:${values.port}${PREFIX} (staging ${values.staging}, else ${ORIGIN})`)
})
