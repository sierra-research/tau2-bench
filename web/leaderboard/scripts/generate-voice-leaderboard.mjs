// Build the leaderboard snapshot from the same committed packages deployed by
// this revision. Per-call evidence stays in S3 and diagnostics remain lazy.
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { DOMAIN_BENCHMARKS, groupPackages } from '../src/data/voiceV2Rows.js'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const base = join(root, 'public', 'submissions')
const read = (dir, file) => JSON.parse(readFileSync(join(base, dir, file), 'utf8'))
const optional = (dir, file) => existsSync(join(base, dir, file)) ? read(dir, file) : null
const manifest = read('', 'manifest.json')
function load(dir, accepts) {
  const { frame, summary } = read(dir, 'benchmark-submission.json')
  if (!accepts(frame.benchmark_version) || !summary.complete) throw new Error(`${dir}: incomplete or unexpected benchmark package`)
  const quality = frame.benchmark_version === 'voice-v2' ? optional(dir, 'benchmark-quality.json') : null
  const pass3 = frame.benchmark_version === 'voice-v2-entity' ? optional(dir, 'benchmark-pass3.json') : null
  return {
    dir,
    metadata: read(dir, 'benchmark-metadata.json'),
    submission: { frame: { benchmark_version: frame.benchmark_version, cells: frame.cells.map(({ language, domain }) => ({ language, domain })) }, summary },
    quality: quality && { cells: quality.cells.map(cell => ({
      language: cell.language, domain: cell.domain,
      interaction: cell.interaction, experience: cell.experience,
      interaction_components: cell.interaction_components,
    })) },
    pass3: pass3 && { score: pass3.score, cells: pass3.cells.map(cell => ({ language: cell.language, domain: cell.domain, score: cell.score })) },
  }
}
const cores = (manifest.voice_v2_submissions || []).map(dir => load(dir, version => version === 'voice-v2'))
const extras = (manifest.voice_v2_domain_submissions || []).map(dir => load(dir, version => version in DOMAIN_BENCHMARKS))
const { rows, errors } = groupPackages(cores, extras)
const output = JSON.stringify({ schema_version: 1, rows, errors }) + '\n'
writeFileSync(join(root, 'public', 'voice-leaderboard.json'), output)
console.log(`voice leaderboard: ${rows.length} rows, ${Buffer.byteLength(output)} bytes`)
