// Generates public/visualizer-index.json: every submission with published
// trajectories, grouped by benchmark, with each language × domain cell's
// trajectory path and score. The trajectory visualizer loads this one file
// instead of fetching every submission's metadata.
//
// Runs before `npm run dev` and `npm run build`; the output is not committed.
//
// Cell layouts (paths are relative to the submissions base URL):
//   dir   — `<path>/results.json` holds tasks and a simulation_index;
//           conversations load from `<path>/simulations/<id>.json`.
//   Text cells also name `flat`, the monolithic results file: it is the
//   download, and the fallback while a text cell has no dir layout yet
//   (tau2.scripts.leaderboard.viewer_data writes it at trajectories/<domain>/).

import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { DOMAIN_BENCHMARKS, LANGUAGES, groupPackages, toPercent } from '../src/data/voiceV2Rows.js'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const submissionsDir = join(root, 'public', 'submissions')
const outFile = join(root, 'public', 'visualizer-index.json')

const readJson = (...parts) => JSON.parse(readFileSync(join(submissionsDir, ...parts), 'utf8'))
const manifest = readJson('manifest.json')

const languageOrder = Object.fromEntries(LANGUAGES.map((l, i) => [l.id, i]))
const domainOrder = { airline: 0, retail: 1, telecom: 2, banking_knowledge: 3, intake: 4 }
const byCell = (a, b) => (languageOrder[a.language] ?? 99) - (languageOrder[b.language] ?? 99)
  || (domainOrder[a.domain] ?? 99) - (domainOrder[b.domain] ?? 99)
const mean = values => (values.length ? values.reduce((s, v) => s + v, 0) / values.length : null)

// A 1.x entry's headline: the mean pass^1 over its core domains, as the
// leaderboard's Overall (banking is scored on its own), else its one domain.
const CORE = new Set(['airline', 'retail', 'telecom'])
function headline(cells) {
  const core = cells.filter(c => CORE.has(c.domain) && c.score != null)
  const scored = core.length ? core : cells.filter(c => c.score != null)
  return {
    score: mean(scored.map(c => c.score)),
    scoreLabel: core.length > 1 ? 'Overall pass^1' : 'pass^1',
  }
}

// Entries from 1.x submission.json files (text and English voice).
function legacyEntry(dir, modality) {
  if (!existsSync(join(submissionsDir, dir, 'submission.json'))) return null
  const sub = readJson(dir, 'submission.json')
  if (!sub.trajectories_available || !sub.trajectory_files) return null
  const cells = Object.entries(sub.trajectory_files).map(([domain, file]) => {
    const score = sub.results?.[domain]?.pass_1 ?? null
    return modality === 'voice'
      ? { language: 'en', domain, layout: 'dir', path: `${dir}/trajectories/${file}`, score }
      : { language: 'en', domain, layout: 'dir', path: `${dir}/trajectories/${domain}`, flat: `${dir}/trajectories/${file}`, score }
  }).sort(byCell)
  const domains = cells.map(c => c.domain)
  const benchmark = modality === 'voice' ? 'voice-en'
    : domains.every(d => d === 'banking_knowledge') ? 'banking' : 'core'
  return {
    id: dir,
    benchmark,
    model: sub.model_name,
    org: sub.model_organization || '',
    date: sub.submission_date || null,
    effort: sub.reasoning_effort || null,
    submissionType: sub.submission_type || 'standard',
    userSimulator: sub.methodology?.user_simulator || null,
    voiceConfig: sub.voice_config || null,
    ...headline(cells),
    cells,
  }
}

// voice-v2 entries: one per agent, its core package plus any banking and
// entity packages, grouped exactly as the leaderboard groups them.
function packageEntries() {
  const load = dir => ({
    dir,
    metadata: readJson(dir, 'benchmark-metadata.json'),
    submission: readJson(dir, 'benchmark-submission.json'),
  })
  const listed = key => (manifest[key] || []).filter(dir => existsSync(join(submissionsDir, dir, 'benchmark-submission.json')))
  const cores = listed('voice_v2_submissions').map(load)
    .filter(pkg => pkg.submission.frame.benchmark_version === 'voice-v2')
  const domainPackages = listed('voice_v2_domain_submissions').map(load)
    .filter(pkg => pkg.submission.frame.benchmark_version in DOMAIN_BENCHMARKS)
  const { rows, errors } = groupPackages(cores, domainPackages)
  for (const error of errors) console.warn(`visualizer index: ${error}`)

  return rows.map(row => {
    const packages = [row.submission ? row : null, ...Object.values(row.extras)].filter(Boolean)
    const cells = packages.flatMap(pkg => {
      // Judged packages name their per-call verdicts and their judge files:
      // a submitted matrix keeps one `diagnostics/verdicts.json` per cell; the
      // release export, one file per simulation under
      // `judges/<agent_provider>/<language>/`.
      const judged = existsSync(join(submissionsDir, pkg.dir, 'benchmark-quality.json'))
      const matrix = pkg.metadata.evidence === 'matrix'
      return pkg.submission.summary.cells.map(cell => ({
        language: cell.language,
        domain: cell.domain,
        layout: 'dir',
        path: `${pkg.dir}/trajectories/${cell.language}/${cell.domain}`,
        score: toPercent(cell.score),
        ...(judged && {
          quality: `${pkg.dir}/benchmark-quality.json`,
          ...(matrix
            ? { verdicts: `${pkg.dir}/trajectories/${cell.language}/${cell.domain}/diagnostics/verdicts.json` }
            : { judges: `${pkg.dir}/judges/${pkg.metadata.agent_provider}/${cell.language}` }),
        }),
      }))
    }).sort(byCell)
    const { metadata } = row
    return {
      id: row.dir,
      benchmark: 'voice-ml',
      model: metadata.model_name,
      org: metadata.model_organization || '',
      date: metadata.submission_date || null,
      effort: null,
      submissionType: metadata.submission_type || 'standard',
      userSimulator: null,
      voiceConfig: null,
      notes: metadata.notes || null,
      score: row.submission ? toPercent(row.submission.summary.score) : null,
      scoreLabel: 'Overall pass^1',
      cells,
    }
  })
}

const entries = [
  ...packageEntries(),
  ...(manifest.voice_submissions || []).map(dir => legacyEntry(dir, 'voice')),
  ...(manifest.submissions || []).map(dir => legacyEntry(dir, 'text')),
].filter(Boolean)

writeFileSync(outFile, JSON.stringify({ entries }) + '\n')
console.log(`visualizer index: ${entries.length} entries → ${outFile.replace(root + '/', '')}`)
