import { useEffect, useState } from 'react'
import { loadJudgeFindings } from '../data/visualizerIndex'
import { COMPONENT_NAMES, MEASURE_NAMES } from '../data/voiceV2Rows'
import './JudgeFindings.css'

// Judges shown, grouped: Generation quality is built from the first two. The
// file's other judges (language-specific checks) are not shown.
const JUDGE_GROUPS = [
  ['Generation quality judges', ['naturalness', 'speech_fidelity']],
  ['Tool use judge', ['tool_use']],
]

const measureLabel = measure => MEASURE_NAMES[measure]
  || measure.charAt(0).toUpperCase() + measure.slice(1).replace(/_/g, ' ')
const factorLabel = factor => factor.replace(/_/g, ' ')

const verdictMark = outcome => ({ pass: '✓', fail: '✗' })[outcome] || outcome || '–'

// Per-call verdict chips: Interaction and Generation quality.
export function CallVerdicts({ call, compact = false }) {
  if (!call) return null
  // A call fails Interaction with two or more flags, so a passing call can
  // still carry one.
  const flagged = Object.entries(call.flags || {})
    .filter(([, value]) => value)
    .map(([key]) => COMPONENT_NAMES[key] || key)
  const interactionTitle = call.interaction === 'unscored'
    ? 'Interaction unscored: too many checks are unknown to decide the call'
    : `Interaction ${call.interaction === 'pass' ? 'passed' : 'failed'}${flagged.length ? ` · flagged: ${flagged.join(', ')}` : ''}`
  const experience = call.experience
  // Counts are shown for scored calls only; an unscored call may have none.
  const issues = experience !== 'unscored' && call.experience_issues != null && call.experience_utterances != null
    ? `${call.experience_issues} issue${call.experience_issues === 1 ? '' : 's'} / ${call.experience_utterances} utterances`
    : null
  return (
    <span className={`call-verdicts ${compact ? 'compact' : ''}`}>
      <span className={`verdict-chip ${call.interaction}`} title={interactionTitle}>
        Interaction {verdictMark(call.interaction)}
        {!compact && flagged.length > 0 && <span className="verdict-detail">{flagged.join(' · ')}</span>}
      </span>
      <span
        className={`verdict-chip ${experience}`}
        title={experience === 'unscored'
          ? `Generation quality unscored: ${call.experience_utterances === 0 ? 'no agent utterances were judged' : 'a judge did not return a complete verdict for this call'}, so it is neither passed nor failed`
          : `Generation quality ${experience}${issues ? ` · ${issues}` : ''}`}
      >
        Gen. quality {verdictMark(experience)}
        {issues && <span className="verdict-detail">{compact ? `${call.experience_issues}/${call.experience_utterances}` : issues}</span>}
      </span>
    </span>
  )
}

function Finding({ finding }) {
  return (
    <li className="judge-finding">
      {finding.quote && (/^tool_(call|result) /.test(finding.quote)
        ? <code className="judge-quote judge-quote-tool">{finding.quote}</code>
        : <q className="judge-quote">{finding.quote}</q>)}
      <p>
        {finding.factor && <span className="judge-tag">{factorLabel(finding.factor)}</span>}
        {finding.category && <span className="judge-tag">{factorLabel(finding.category)}</span>}
        {finding.text || 'No reasoning given.'}
      </p>
      {(finding.severity != null || finding.time) && (
        <span className="judge-meta">
          {finding.severity != null && `Severity ${finding.severity}`}
          {finding.severity != null && finding.time && ' · '}
          {finding.time && `${finding.time}s into the utterance`}
        </span>
      )}
    </li>
  )
}

// One judge: a summary line that expands to its flagged findings.
function MeasureSection({ result }) {
  const unit = result.measure === 'tool_use' ? 'call' : 'utterance'
  const header = (
    <span className="judge-measure-header">
      <span className="judge-measure-name">{measureLabel(result.measure)}</span>
      <span className={`judge-count ${result.error ? 'flagged' : result.flagged.length ? 'flagged' : 'clean'}`}>
        {result.error ? 'Judge error' : result.flagged.length
          ? `${result.flagged.length} of ${result.judged} ${unit}s flagged`
          : `${result.judged} ${unit}${result.judged === 1 ? '' : 's'} judged · no issues`}
      </span>
    </span>
  )
  const unjudged = result.unitErrors.length > 0 && (
    <p className="judge-unjudged">
      {result.unitErrors.length} {unit}{result.unitErrors.length === 1 ? '' : 's'} not judged: {[...new Set(result.unitErrors)].join('; ')}
    </p>
  )
  if (!result.error && !result.flagged.length) return <div className="judge-measure">{header}{unjudged}</div>
  return (
    <details className="judge-measure">
      <summary>{header}</summary>
      {unjudged}
      {result.error && <p className="judge-error">{String(result.error)}</p>}
      {result.flagged.map(unitResult => (
        <div key={unitResult.index} className="judge-unit">
          {result.total > 1 && <span className="judge-unit-label">Agent utterance {unitResult.index + 1} of {result.total}</span>}
          <ul>{unitResult.findings.map((finding, i) => <Finding key={i} finding={finding} />)}</ul>
        </div>
      ))}
    </details>
  )
}

/**
 * Judge verdicts for one conversation of a judged τ³-Voice Multilingual cell.
 * Judges are diagnostics: they never change the task reward.
 */
export default function JudgeFindings({ cell, simulationId, call }) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    let cancelled = false
    setState({ loading: true })
    loadJudgeFindings(cell, simulationId)
      .then(results => !cancelled && setState({ results }))
      .catch(err => !cancelled && setState({ error: err.message }))
    return () => { cancelled = true }
  }, [cell, simulationId])

  const groups = JUDGE_GROUPS
    .map(([title, measures]) => [title, measures.map(m => state.results?.find(r => r.measure === m)).filter(Boolean)])
    .filter(([, results]) => results.length)

  return (
    <section className="judge-findings">
      <div className="judge-findings-header">
        <div>
          <h4>Judge results</h4>
          <p className="judge-note">Diagnostics from LLM judges. They do not affect the task reward.</p>
        </div>
        <CallVerdicts call={call} />
      </div>
      {state.loading && <p className="judge-note">Loading judge results…</p>}
      {state.error && <p className="judge-error">{state.error}</p>}
      {groups.map(([title, results]) => (
        <div key={title} className="judge-group">
          <h5>{title}</h5>
          {results.map(result => <MeasureSection key={result.measure} result={result} />)}
        </div>
      ))}
    </section>
  )
}
