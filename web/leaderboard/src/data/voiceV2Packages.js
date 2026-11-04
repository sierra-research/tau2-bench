import { useEffect, useState } from 'react'
import { DOMAIN_BENCHMARKS, groupPackages } from './voiceV2Rows'
export { CORE_DOMAINS, DOMAIN_BENCHMARKS, LANGUAGES, PASS3_DOMAINS, domainScore, groupPackages, packageScore, toPercent } from './voiceV2Rows'

// 2.0 packages listed under `voice_v2_submissions` in manifest.json, plus the
// separately scored banking and entity packages listed under
// `voice_v2_domain_submissions`. Each directory holds the files
// `tau2 submit prepare` writes; see
// src/tau2/scripts/leaderboard/benchmark_package.py.
const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`
const NO_CACHE = { cache: 'no-cache' }
const FILES = {
  metadata: 'benchmark-metadata.json',
  submission: 'benchmark-submission.json',
  diagnostics: 'benchmark-diagnostics.json',
  quality: 'benchmark-quality.json',
  pass3: 'benchmark-pass3.json',
}
// Files a package may omit, requested only for the frames that can carry them
// (benchmark_package.py). A missing one loads as null; some static servers
// answer a missing file with the HTML index, which is treated the same way.
// Diagnostics are the largest file and only feed a breakdown, so they load on
// demand (loadDiagnostics), not with the package.
const OPTIONAL_BY_FRAME = {
  'voice-v2': ['quality'],
  'voice-v2-entity': ['pass3'],
}
const OPTIONAL = new Set(['diagnostics', ...Object.values(OPTIONAL_BY_FRAME).flat()])

const providerLogo = { gemini: 'Google__G__logo.svg.png', openai: 'openai.svg', openai_live: 'openai.svg', xai: 'xai-logo.svg', avartha_pipecat: 'avartha-logo.svg' }

async function loadFile(dir, key) {
  const file = FILES[key]
  const response = await fetch(`${SUBMISSIONS_BASE}/${dir}/${file}`, NO_CACHE)
  if (!response.ok) {
    if (OPTIONAL.has(key)) return [key, null]
    throw new Error(`${dir}/${file}: HTTP ${response.status}`)
  }
  if (!OPTIONAL.has(key)) return [key, await response.json()]
  try { return [key, await response.json()] } catch { return [key, null] }
}

// The required files first; the frame then names the optional files to request,
// so a package is never probed for a file its frame cannot carry.
async function loadPackage(dir, accepts) {
  const required = Object.keys(FILES).filter(key => !OPTIONAL.has(key))
  const pkg = Object.fromEntries(await Promise.all(required.map(key => loadFile(dir, key))))
  const { summary, frame } = pkg.submission
  if (!accepts(frame.benchmark_version) || !summary.complete) throw new Error(`${dir}: not a complete ${frame.benchmark_version} package of this list`)
  const optional = OPTIONAL_BY_FRAME[frame.benchmark_version] || []
  Object.assign(pkg, Object.fromEntries([...OPTIONAL].filter(key => key !== 'diagnostics').map(key => [key, null])))
  Object.assign(pkg, Object.fromEntries(await Promise.all(optional.map(key => loadFile(dir, key)))))
  return { dir, ...pkg }
}

// A voice-v2 package's benchmark-diagnostics.json, fetched once per package
// when a breakdown first needs it; resolves to null if the package has none.
const diagnosticsCache = new Map()
export function loadDiagnostics(dir) {
  if (!diagnosticsCache.has(dir)) {
    diagnosticsCache.set(dir, loadFile(dir, 'diagnostics').then(([, value]) => value, () => null))
  }
  return diagnosticsCache.get(dir)
}

// Loads every complete package in the manifest. `packages` is null while
// loading; `errors` names packages that were skipped.
export function useVoiceV2Packages() {
  const [packages, setPackages] = useState(null)
  const [errors, setErrors] = useState([])
  useEffect(() => {
    let cancelled = false
    const settle = results => [
      results.filter(result => result.status === 'fulfilled').map(result => result.value),
      results.filter(result => result.status === 'rejected').map(result => result.reason.message),
    ]
    const load = async () => {
      // One small same-origin snapshot, generated at every build, replaces the
      // S3 manifest → packages → quality waterfall. All score breakdowns are
      // included; only detailed judge evidence needs another request.
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}voice-leaderboard.json`, NO_CACHE)
        if (!response.ok) throw new Error(`Leaderboard snapshot: HTTP ${response.status}`)
        const snapshot = await response.json()
        if (snapshot.schema_version !== 1 || !Array.isArray(snapshot.rows) || !Array.isArray(snapshot.errors)) throw new Error('Invalid leaderboard snapshot')
        if (!cancelled) {
          setPackages(snapshot.rows)
          setErrors(snapshot.errors)
        }
        return
      } catch {
        // Older/self-hosted deployments can still serve packages directly.
      }
      const response = await fetch(`${SUBMISSIONS_BASE}/manifest.json`, NO_CACHE)
      const manifest = await response.json()
      const [coreResults, extraResults] = await Promise.all([
        Promise.allSettled((manifest.voice_v2_submissions || []).map(dir => loadPackage(dir, version => version === 'voice-v2'))),
        Promise.allSettled((manifest.voice_v2_domain_submissions || []).map(dir => loadPackage(dir, version => version in DOMAIN_BENCHMARKS))),
      ])
      const [cores, coreErrors] = settle(coreResults)
      const [extras, extraErrors] = settle(extraResults)
      if (cancelled) return
      const grouped = groupPackages(cores, extras)
      setPackages(grouped.rows)
      setErrors([...coreErrors, ...extraErrors, ...grouped.errors])
    }
    load().catch(error => !cancelled && (setPackages([]), setErrors([error.message])))
    return () => { cancelled = true }
  }, [])
  return { packages, errors }
}

// Interaction or Generation quality (`experience`) pass rate, in percent,
// pooled over the scored calls of the voice-v2 package's quality file:
// passed / (passed + failed). Unscored and excluded calls are left out. Scope
// is one language or 'all', and one core domain or 'all'; null if the package
// has no quality file or no scored call in scope.
const qualityCells = (pkg, language, domain) => (pkg.quality?.cells || [])
  .filter(cell => (language === 'all' || cell.language === language) && (domain === 'all' || cell.domain === domain))
export function packageQuality(pkg, measure, language = 'all', domain = 'all') {
  const cells = qualityCells(pkg, language, domain)
  const passed = cells.reduce((sum, cell) => sum + (cell[measure]?.passed || 0), 0)
  const failed = cells.reduce((sum, cell) => sum + (cell[measure]?.failed || 0), 0)
  return passed + failed ? 100 * passed / (passed + failed) : null
}

// Share of calls (percent) with one Interaction flag raised, over the calls
// where that flag is known.
export function interactionComponent(pkg, component, language = 'all', domain = 'all') {
  const cells = qualityCells(pkg, language, domain)
  const flagged = cells.reduce((sum, cell) => sum + (cell.interaction_components?.[component]?.flagged || 0), 0)
  const known = cells.reduce((sum, cell) => sum + (cell.interaction_components?.[component]?.known || 0), 0)
  return known ? 100 * flagged / known : null
}

// Calls counted toward a quality measure: scored, unscored and excluded.
export function qualityCounts(pkg, measure, language = 'all', domain = 'all') {
  const counts = { passed: 0, failed: 0, unscored: 0, excluded: 0 }
  for (const cell of qualityCells(pkg, language, domain)) for (const key of Object.keys(counts)) counts[key] += cell[measure]?.[key] || 0
  return counts
}

export const packageLogo = pkg => providerLogo[pkg.metadata.agent_provider]
