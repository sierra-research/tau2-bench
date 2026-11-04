import { useEffect, useState } from 'react'
import { heatScale, percent } from '../data/multilingual/metrics'
import { CORE_DOMAINS, LANGUAGES, PASS3_DOMAINS, domainScore, interactionComponent, loadDiagnostics, packageQuality, qualityCounts, toPercent } from '../data/voiceV2Packages'
import { COMPONENT_NAMES, MEASURE_NAMES } from '../data/voiceV2Rows'
import './MultilingualVoice.css'

// Breakdown of one submitted voice-v2 package and any banking or entity
// packages of the same agent, opened from its row in the multilingual
// leaderboard (MultilingualVoice).
const languageName = Object.fromEntries(LANGUAGES.map(item => [item.id, item.name]))
const tabs = [['task', 'Task completion'], ['interaction', 'Interaction'], ['experience', 'Generation quality']]
const componentName = COMPONENT_NAMES
const domainName = { airline: 'Airline', retail: 'Retail', telecom: 'Telecom', banking_knowledge: 'Banking', intake: 'Intake' }
const extraDomains = ['banking_knowledge', 'intake']
const measureName = MEASURE_NAMES
const label = (names, key) => names[key] || key.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())

// Pool each judged measure's units over the cells of one language (or all).
function measureTable(diagnostics) {
  if (!diagnostics) return { measures: [], value: () => null, coverage: () => null }
  const measures = [...new Set(diagnostics.cells.flatMap(cell => cell.summary.map(row => row.measure)))]
  const pooled = (measure, language) => {
    const rows = diagnostics.cells
      .filter(cell => language === 'all' || cell.language === language)
      .flatMap(cell => cell.summary.filter(row => row.measure === measure && row.applicable_calls > 0))
    if (!rows.length) return null
    const sum = key => rows.reduce((total, row) => total + row[key], 0)
    return { passed: sum('pass_units'), failed: sum('fail_units'), judged: sum('judged_calls'), applicable: sum('applicable_calls') }
  }
  return {
    measures,
    value: (measure, language) => {
      const totals = pooled(measure, language)
      return totals && totals.passed + totals.failed ? 100 * totals.passed / (totals.passed + totals.failed) : null
    },
    coverage: pooled,
  }
}

function Heatmap({ title, rowLabel, rows, columns, note, kind = 'pass' }) {
  // Failure rates shade on 100 - value so darker stays better.
  const shade = (row, value) => value == null ? null : row.failure ? 100 - value : value
  const heat = heatScale(rows.flatMap(row => columns.map(column => shade(row, row.value(column.key)))))
  return <div className="mv-factor-breakdown">
    <div className="mv-factor-heading"><h3>{title}</h3><span>{note}</span></div>
    <div className="mv-table-scroll" role="region" aria-label={title} tabIndex="0"><table className={`mv-factor-heatmap mv-${kind}-heatmap`} style={{ minWidth: `${180 + columns.length * 105}px` }}>
      <colgroup><col className="mv-factor-label-col" />{columns.map(column => <col key={column.key} />)}</colgroup>
      <thead><tr><th scope="col">{rowLabel}</th>{columns.map(column => <th scope="col" key={column.key} className={column.key === 'all' ? 'mv-overall' : ''}>{column.label}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.key} data-factor={row.key} className={row.key === 'all' ? 'mv-overall-row' : row.extra ? 'mv-extra-row' : ''}><th scope="row">{row.label}</th>{columns.map(column => {
        const value = row.value(column.key)
        return <td key={column.key} data-language={column.key} className={column.key === 'all' ? 'mv-overall' : ''}><span className={`mv-heat-cell${value == null ? ' is-missing' : ''}`} style={{ background: heat(shade(row, value)) }} title={`${row.label} · ${column.label}: ${value == null ? 'Not applicable' : percent(value)}${row.title ? row.title(column.key) : ''}`}>{percent(value)}</span></td>
      })}</tr>)}</tbody>
    </table></div>
  </div>
}

export default function PackageBreakdown({ pkg, metric = 'task', onMetricChange, onClose }) {
  const summary = pkg.submission?.summary
  const parts = [pkg.submission && pkg, ...Object.values(pkg.extras || {})].filter(Boolean)
  const languages = [...new Set(parts.flatMap(part => part.submission.frame.cells.map(cell => cell.language)))]
  const domains = summary ? Object.keys(summary.by_domain) : []
  const domainRow = domain => ({ key: domain, label: PASS3_DOMAINS.includes(domain) ? `${label(domainName, domain)} (pass^3)` : label(domainName, domain), extra: extraDomains.includes(domain), value: language => domainScore(pkg, domain, language) })
  const columns = [...languages.map(key => ({ key, label: label(languageName, key) })), { key: 'all', label: 'Overall' }]
  const taskRows = [
    ...domains.filter(domain => !extraDomains.includes(domain)).map(domainRow),
    { key: 'all', label: 'Overall', value: language => summary ? toPercent(language === 'all' ? summary.score : summary.by_language[language]?.score) : null },
    ...extraDomains.map(domainRow),
  ]
  // Judge diagnostics load when Generation quality is first opened.
  const [diagnostics, setDiagnostics] = useState(undefined)
  const wantsDiagnostics = metric === 'experience' && Boolean(pkg.submission)
  useEffect(() => {
    if (!wantsDiagnostics) return
    let cancelled = false
    loadDiagnostics(pkg.dir).then(value => { if (!cancelled) setDiagnostics(value) })
    return () => { cancelled = true }
  }, [wantsDiagnostics, pkg.dir])
  const judged = measureTable(diagnostics)
  // Only the two judge measures Generation quality is built from are shown;
  // the other diagnostics (tool use, language checks) are left out.
  const qualityMeasures = ['naturalness', 'speech_fidelity'].filter(measure => judged.measures.includes(measure))
  const judgeRows = qualityMeasures.map(measure => ({
    key: measure,
    label: label(measureName, measure),
    value: language => judged.value(measure, language),
    title: language => {
      const coverage = judged.coverage(measure, language)
      return coverage ? ` · ${coverage.judged}/${coverage.applicable} applicable calls judged` : ''
    },
  }))
  const coreColumns = [...(pkg.submission?.frame.cells || []).map(cell => cell.language).filter((language, index, all) => all.indexOf(language) === index).map(key => ({ key, label: label(languageName, key) })), { key: 'all', label: 'Overall' }]
  const qualityRows = measure => [
    ...CORE_DOMAINS.map(domain => ({ key: domain, label: label(domainName, domain), value: language => packageQuality(pkg, measure, language, domain) })),
    { key: 'all', label: 'Overall', value: language => packageQuality(pkg, measure, language), title: language => {
      const counts = qualityCounts(pkg, measure, language)
      return ` · ${counts.passed}/${counts.passed + counts.failed} scored calls pass${counts.unscored ? ` · ${counts.unscored} unscored` : ''}${counts.excluded ? ` · ${counts.excluded} excluded` : ''}`
    } },
  ]
  const componentRows = Object.entries(componentName).map(([key, name]) => ({ key, label: name, failure: true, value: language => interactionComponent(pkg, key, language) }))
  const headline = {
    task: summary ? toPercent(summary.score) : null,
    interaction: packageQuality(pkg, 'interaction'),
    experience: packageQuality(pkg, 'experience'),
  }
  const overall = measure => qualityCounts(pkg, measure)
  const { metadata } = pkg
  return <div className="mv-detail" id={`v2-breakdown-${pkg.dir}`}>
    <div className="mv-detail-heading"><span><strong>{metadata.model_name}</strong></span><div className="mv-breakdown-meta"><span>Submitted {parts.length > 1 ? `${parts.length} packages` : 'package'} · {parts.reduce((total, part) => total + part.submission.summary.scored_trials, 0)} scored calls · {parts.reduce((total, part) => total + part.submission.frame.cells.length, 0)} cells</span><button className="mv-close-breakdown" aria-label="Close breakdown" onClick={onClose}>×</button></div></div>
    {onMetricChange && <div className="mv-breakdown-tabs" aria-label="Metric breakdown">{tabs.map(([key, name]) => <button key={key} aria-pressed={metric === key} onClick={() => onMetricChange(key)}>{name}<strong>{percent(headline[key])}</strong></button>)}</div>}
    {metric === 'task' && <Heatmap title="Task completion (pass@1)" rowLabel="Domain" rows={taskRows} columns={columns} kind="task" note="Mean task reward · cells weighted equally" />}
    {metric === 'interaction' && (pkg.quality ? <>
      <Heatmap title="Interaction by language" rowLabel="Domain" rows={qualityRows('interaction')} columns={coreColumns} note="Share of scored calls that pass · higher is better" />
      <Heatmap title="Interaction failures" rowLabel="Failure" rows={componentRows} columns={coreColumns} kind="failure" note="Share of calls with the failure · lower is better" />
    </> : <p className="mv-footnote">This submission does not include Interaction scores.</p>)}
    {metric === 'experience' && (pkg.quality ? <Heatmap title="Generation quality by language" rowLabel="Domain" rows={qualityRows('experience')} columns={coreColumns} note="Share of scored calls that pass · higher is better" /> : <p className="mv-footnote">This submission does not include Generation quality scores.</p>)}
    {metric === 'experience' && pkg.submission && diagnostics === undefined && <p className="mv-footnote">Loading judge diagnostics…</p>}
    {metric === 'experience' && diagnostics && <Heatmap title="Generation quality judges" rowLabel="Judge" rows={judgeRows} columns={columns} note="Share of judged utterances without an issue · higher is better" />}
    <div className="mv-factor-notes">
      {metric === 'interaction' && pkg.quality && <p className="mv-footnote">A call fails Interaction with two or more of the five failures: no response, more than three interruptions, more than three selectivity errors, a monologue, or a tool-use error. It passes when it could not reach two even if every unknown check failed; otherwise it is unscored. {overall('interaction').unscored + overall('interaction').excluded} of {Object.values(overall('interaction')).reduce((a, b) => a + b, 0)} calls are unscored or excluded.</p>}
      {metric === 'experience' && pkg.quality && <p className="mv-footnote">A call passes Generation quality when at most 25% of the agent’s judged utterances have a naturalness or speech-fidelity issue. Utterances recorded without a transcript, or whose speech-fidelity judgment hit the judge’s output limit, are left out; calls with other incomplete judgments are unscored. {overall('experience').unscored + overall('experience').excluded} of {Object.values(overall('experience')).reduce((a, b) => a + b, 0)} calls are unscored or excluded.</p>}
    </div>
  </div>
}

