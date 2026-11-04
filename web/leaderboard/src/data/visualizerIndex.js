// Data access for the trajectory visualizer. The list of submissions comes
// from one build-time file (scripts/generate-visualizer-index.mjs); a cell's
// task list from its small results.json index; a conversation from its own
// simulations/<id>.json, fetched only when opened.

export const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`
export const S3_SUBMISSIONS = 's3://sierra-tau-bench-public/submissions'

// Tab order follows the leaderboard: newest tracks first.
export const BENCHMARKS = [
  { id: 'voice-ml', label: 'τ³-Voice', sub: 'Multilingual', modality: 'voice' },
  { id: 'voice-en', label: 'τ³-Voice', sub: 'English', modality: 'voice' },
  { id: 'banking', label: 'τ³-Banking', sub: 'Text', modality: 'text' },
  { id: 'core', label: 'τ²-bench', sub: 'Text', modality: 'text' },
]

export const DOMAINS = {
  airline: { label: 'Airline', icon: '✈️', color: '#3b82f6' },
  retail: { label: 'Retail', icon: '🛍️', color: '#8b5cf6' },
  telecom: { label: 'Telecom', icon: '📱', color: '#059669' },
  banking_knowledge: { label: 'Banking', icon: '🏦', color: '#d97706' },
  intake: { label: 'Intake', icon: '📋', color: '#db2777' },
}

export { LANGUAGES } from './voiceV2Rows'

let indexPromise = null
export function loadVisualizerIndex() {
  if (!indexPromise) {
    indexPromise = fetch(`${import.meta.env.BASE_URL}visualizer-index.json`, { cache: 'no-cache' })
      .then(response => {
        if (!response.ok) throw new Error(`Failed to load the submission index (HTTP ${response.status})`)
        return response.json()
      })
      .then(index => index.entries)
    indexPromise.catch(() => { indexPromise = null })
  }
  return indexPromise
}

// Some static hosts answer a missing file with the HTML app shell; treat a
// non-JSON body like a 404.
async function fetchJson(url, init) {
  const response = await fetch(url, init)
  if (!response.ok) return { ok: false, status: response.status }
  try {
    return { ok: true, data: await response.json() }
  } catch {
    return { ok: false, status: 404 }
  }
}

// A dir-format index lists simulations without their messages; give the UI
// the simulations array it expects, each marked as a stub to be fetched.
function withSimulationStubs(data) {
  if ((!data.simulations || data.simulations.length === 0) && data.simulation_index) {
    data.simulations = data.simulation_index.map(entry => ({
      id: entry.id,
      task_id: entry.task_id,
      trial: entry.trial,
      reward_info: entry.reward != null ? { reward: entry.reward } : null,
      termination_reason: entry.termination_reason || null,
      agent_cost: entry.agent_cost ?? null,
      duration: entry.duration ?? null,
      stub: true,
    }))
  }
  return data
}

const cellCache = new Map()

// Loads a cell's tasks and simulation list. Text cells fall back to their
// monolithic file until the dir layout is published for them.
export function loadCell(cell) {
  const key = cell.path
  if (!cellCache.has(key)) {
    const promise = (async () => {
      const index = await fetchJson(`${SUBMISSIONS_BASE}/${cell.path}/results.json`)
      if (index.ok) return { data: withSimulationStubs(index.data), layout: 'dir' }
      if (!cell.flat) throw new Error(`Trajectories for this cell are not published yet (HTTP ${index.status}).`)
      const flat = await fetchJson(`${SUBMISSIONS_BASE}/${cell.flat}`)
      if (!flat.ok) throw new Error(`Failed to load trajectories (HTTP ${flat.status}).`)
      return { data: flat.data, layout: 'flat' }
    })()
    cellCache.set(key, promise)
    promise.catch(() => cellCache.delete(key))
  }
  return cellCache.get(key)
}

export const simulationUrl = (cell, id) => `${SUBMISSIONS_BASE}/${cell.path}/simulations/${id}.json`

export async function loadSimulation(cell, id) {
  const result = await fetchJson(simulationUrl(cell, id))
  if (!result.ok) throw new Error(`This conversation is not published yet (HTTP ${result.status}).`)
  return result.data
}

const qualityCache = new Map()

// A judged cell's per-call Interaction and Generation quality verdicts, by
// simulation id (from its package's benchmark-quality.json). Null when the
// cell is not judged.
export async function loadCellQuality(cell) {
  if (!cell.quality) return null
  if (!qualityCache.has(cell.quality)) {
    const promise = fetchJson(`${SUBMISSIONS_BASE}/${cell.quality}`).then(result => {
      if (!result.ok) throw new Error(`Failed to load judge results (HTTP ${result.status}).`)
      return result.data
    })
    qualityCache.set(cell.quality, promise)
    promise.catch(() => qualityCache.delete(cell.quality))
  }
  const quality = await qualityCache.get(cell.quality)
  const match = quality.cells.find(c => c.language === cell.language && c.domain === cell.domain)
  return new Map((match?.calls || []).map(call => [call.simulation_id, call]))
}

function parseEvidence(evidence) {
  try {
    return JSON.parse(evidence)
  } catch {
    return { reasoning: evidence }
  }
}

// The flagged findings of one judged unit (an agent utterance, or the call for
// tool use). Speech fidelity lists findings with a clip time range; tool use
// reports one verdict per factor; the other judges give one reasoning.
function unitFindings(unit) {
  const evidence = parseEvidence(unit.evidence)
  if (Array.isArray(evidence.findings)) {
    const kept = evidence.findings.filter(f => !f.excluded_as_truncation)
    return kept.map(f => ({ text: f.issue, category: f.category, severity: f.severity, time: f.time_range }))
  }
  if (typeof evidence.reasoning === 'string') {
    return [{ text: evidence.reasoning, quote: evidence.quote || null, severity: evidence.severity }]
  }
  return Object.entries(evidence)
    .filter(([, factor]) => factor && factor.violated)
    .map(([key, factor]) => ({ factor: key, text: factor.reasoning, quote: factor.quote || null }))
}

const verdictsCache = new Map()

// A submitted matrix cell's verdicts, all its calls in one file.
function loadCellVerdicts(path) {
  if (!verdictsCache.has(path)) {
    const promise = fetchJson(`${SUBMISSIONS_BASE}/${path}`).then(result => {
      if (!result.ok) throw new Error(`No judge results for this cell (HTTP ${result.status}).`)
      return result.data
    })
    verdictsCache.set(path, promise)
    promise.catch(() => verdictsCache.delete(path))
  }
  return verdictsCache.get(path)
}

// One conversation's saved verdict records, from its cell's file or its own.
async function loadVerdictRecords(cell, id) {
  if (cell.verdicts) {
    const records = (await loadCellVerdicts(cell.verdicts))
      .filter(record => record.identity.input.simulation_id === id)
    if (!records.length) throw new Error('No judge results for this conversation.')
    return records
  }
  const result = await fetchJson(`${SUBMISSIONS_BASE}/${cell.judges}/${id}.json`)
  if (!result.ok) throw new Error(`No judge results for this conversation (HTTP ${result.status}).`)
  return result.data
}

// One conversation's judge verdicts: per judge, how many units it judged and
// the flagged ones with their findings.
export async function loadJudgeFindings(cell, id) {
  const records = await loadVerdictRecords(cell, id)
  return records.map(record => {
    const units = record.verdict?.units || []
    return {
      measure: record.identity.measure,
      error: record.error || null,
      judged: units.filter(u => u.opportunity).length,
      // Units the judge could not assess (e.g. speech with no transcript)
      unitErrors: (record.verdict?.unit_errors || []).map(e => e.error),
      flagged: units
        .map((unit, index) => ({ index, findings: unit.violation ? unitFindings(unit) : [] }))
        .filter(unit => unit.findings.length),
      total: units.length,
    }
  })
}

export const formatEffort = effort => (effort ? effort.charAt(0).toUpperCase() + effort.slice(1) : '')
