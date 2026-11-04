// Pure helpers over voice-v2 packages, shared by the site (voiceV2Packages.js)
// and the build-time visualizer index (scripts/generate-visualizer-index.mjs).
// No React and no import.meta here, so Node can import this module.

export const LANGUAGES = [
  { id: 'en', name: 'English' }, { id: 'es', name: 'Spanish' }, { id: 'pt', name: 'Portuguese' },
  { id: 'hi', name: 'Hindi' }, { id: 'ko', name: 'Korean' }, { id: 'zh', name: 'Mandarin' },
]
export const CORE_DOMAINS = ['airline', 'retail', 'telecom']

// Public names of the Interaction components and the Generation quality judges.
export const COMPONENT_NAMES = { nonresponse: 'Non-response', interruption: 'Interruptions', selectivity: 'Selectivity errors', monologue: 'Monologues', tool_misuse: 'Tool-use errors' }
export const MEASURE_NAMES = { naturalness: 'Naturalness', speech_fidelity: 'Speech fidelity' }

// Domain benchmarks score domains outside the voice-v2 Overall. Their packages
// attach to the same agent's voice-v2 row; they never change its Overall.
export const DOMAIN_BENCHMARKS = { 'voice-v2-banking': 'banking_knowledge', 'voice-v2-entity': 'intake' }
// Domains scored by crossed Pass^3 (benchmark-pass3.json) instead of pass@1:
// a task passes only if it passes in all three voice conditions.
export const PASS3_DOMAINS = ['intake']

export const toPercent = value => (value == null ? null : 100 * value)

// One submitted system: the same model, organization, agent and submission type.
const agentKey = ({ metadata }) => [metadata.model_name, metadata.model_organization, metadata.agent_provider, metadata.agent_model, metadata.submission_type || 'standard'].join('\u0000')

// Attach each domain package to the one voice-v2 package of the same agent.
// A domain package without one becomes its own row with no Overall. Rows are
// voice-v2-shaped: `submission` is the voice-v2 package (or null) and
// `extras` maps a domain to its separately scored package.
export function groupPackages(cores, domainPackages) {
  const errors = []
  // One row per agent: the first listed voice-v2 package wins.
  const byAgent = new Map()
  for (const pkg of cores) {
    const key = agentKey(pkg)
    if (byAgent.has(key)) errors.push(`${pkg.dir}: duplicate voice-v2 package for ${byAgent.get(key).dir}`)
    else byAgent.set(key, { ...pkg, extras: {} })
  }
  const rows = [...byAgent.values()]
  const standalone = new Map()
  for (const pkg of domainPackages) {
    const domain = DOMAIN_BENCHMARKS[pkg.submission.frame.benchmark_version]
    const key = agentKey(pkg)
    const core = byAgent.get(key)
    const row = core || standalone.get(key) || { dir: pkg.dir, metadata: pkg.metadata, submission: null, extras: {} }
    if (row.extras[domain]) {
      errors.push(`${pkg.dir}: duplicate ${domain} package for ${row.dir}`)
      continue
    }
    row.extras[domain] = pkg
    if (!core) standalone.set(key, row)
  }
  return { rows: [...rows, ...standalone.values()], errors }
}

// Task-completion pass@1 (percent) for one language or 'all'; null if the
// package does not cover that language or has no voice-v2 Overall.
export const packageScore = (pkg, language) => pkg.submission && toPercent(language === 'all' ? pkg.submission.summary.score : pkg.submission.summary.by_language[language]?.score)

// Task completion (percent) for one domain and language or 'all', from the
// domain's own package when it has one, else from the voice-v2 package.
// Pass@1, except crossed Pass^3 for PASS3_DOMAINS; null if a Pass^3 domain's
// package carries no benchmark-pass3.json.
export function domainScore(pkg, domain, language = 'all') {
  if (PASS3_DOMAINS.includes(domain)) {
    const pass3 = pkg.extras?.[domain]?.pass3
    if (!pass3) return null
    return toPercent(language === 'all' ? pass3.score : pass3.cells.find(cell => cell.language === language && cell.domain === domain)?.score)
  }
  const summary = pkg.extras?.[domain]?.submission.summary || pkg.submission?.summary
  if (!summary) return null
  if (language === 'all') return toPercent(summary.by_domain[domain]?.score)
  return toPercent(summary.cells.find(cell => cell.language === language && cell.domain === domain)?.score)
}
