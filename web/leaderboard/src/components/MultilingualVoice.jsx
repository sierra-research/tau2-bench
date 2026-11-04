import { Fragment, useEffect, useRef, useState } from 'react'
import { frontier, percent, mean, heatScale } from '../data/multilingual/metrics'
import PackageBreakdown from './PackageBreakdown'
import { CORE_DOMAINS, LANGUAGES, PASS3_DOMAINS, domainScore, packageLogo, packageQuality, packageScore, useVoiceV2Packages } from '../data/voiceV2Packages'
import './MultilingualVoice.css'

const metrics = [['task', 'Task completion'], ['interaction', 'Interaction'], ['experience', 'Generation quality']]
// One color per agent provider; a provider's further packages take the palette.
const providerColors = { openai: '#19795b', openai_live: '#19795b', gemini: '#3b6fd8', xai: '#2f3437' }
const palette = ['#c26a2b', '#8a4fbf', '#2ea3b8', '#b8435a', '#6b8e23']

function Space({ rows, onSelect, selected, focused, onClearFocus }) {
  const [yaw, setYaw] = useState(32)
  const [tilt, setTilt] = useState(22)
  const [fit, setFit] = useState(true)
  const drag = useRef(null)
  const axes = [
    { key: 'task', label: 'Task completion' },
    { key: 'experience', label: 'Generation quality' },
    { key: 'interaction', label: 'Interaction' },
  ]
  const eligible = rows.filter(row => axes.every(axis => row[axis.key] != null))
  const ranges = axes.map(({ key }) => {
    const values = eligible.map(row => row[key])
    return fit && values.length ? [Math.max(0, Math.floor(Math.min(...values) / 5) * 5 - 5), Math.min(100, Math.ceil(Math.max(...values) / 5) * 5 + 5)] : [0, 100]
  })
  const coordinates = row => axes.map(({ key }, index) => (row[key] - ranges[index][0]) / (ranges[index][1] - ranges[index][0]))
  const radians = yaw * Math.PI / 180
  const elevation = tilt * Math.PI / 180
  const project = ([x, y, z]) => {
    const horizontal = (x - .5) * Math.cos(radians) + (z - .5) * Math.sin(radians)
    const depth = -(x - .5) * Math.sin(radians) + (z - .5) * Math.cos(radians)
    return [385 + horizontal * 350, 285 - ((y - .5) * Math.cos(elevation) + depth * Math.sin(elevation)) * 350]
  }
  const line = (from, to, props = {}) => {
    const a = project(from), b = project(to)
    return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} {...props} />
  }
  const polygon = points => points.map(point => project(point).join(',')).join(' ')
  const best = frontier(rows, axes.map(axis => axis.key))
  const active = rows.find(row => row.id === selected)
  const vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
  const reset = () => { setYaw(32); setTilt(22) }
  const move = event => {
    if (!drag.current) return
    setYaw(Math.max(12, Math.min(78, drag.current.yaw + (event.clientX - drag.current.x) * .3)))
    setTilt(Math.max(8, Math.min(45, drag.current.tilt - (event.clientY - drag.current.y) * .2)))
  }
  return <div className="mv-cube-layout">
    <div className="mv-cube-stage">
      <div className="mv-cube-toolbar"><div className="mv-range-switch" aria-label="Axis scale"><button aria-pressed={fit} onClick={() => setFit(true)}>Fit results</button><button aria-pressed={!fit} onClick={() => setFit(false)}>Full scale · 0–100</button></div><div className="mv-cube-actions">{focused && <button className="mv-reset" onClick={onClearFocus}>Show all models</button>}<button className="mv-reset" onClick={reset}>↺ Reset view</button></div></div>
      {eligible.length ? <>
        <svg className="mv-cube" viewBox="0 0 800 640" role="group" aria-label="Interactive 3D metric cube" onPointerDown={event => {
          if (event.button !== 0 || event.target.closest('[role="button"]')) return
          drag.current = { x: event.clientX, y: event.clientY, yaw, tilt }
          event.currentTarget.setPointerCapture(event.pointerId)
        }} onPointerMove={move} onPointerUp={() => { drag.current = null }} onPointerCancel={() => { drag.current = null }} onLostPointerCapture={() => { drag.current = null }}>
          <title>Task completion, generation quality, and interaction. Drag left or right to rotate, and up or down to tilt.</title>
          <polygon points={polygon([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])} fill="#eeeeee" fillOpacity=".85" />
          <polygon points={polygon([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])} fill="#f5f5f5" fillOpacity=".9" />
          <polygon points={polygon([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])} fill="#f8f8f8" fillOpacity=".8" />
          {[0, .25, .5, .75, 1].map(value => <g key={value} stroke="#d6d6d6" strokeWidth=".8">
            {line([value, 0, 0], [value, 0, 1])}{line([0, 0, value], [1, 0, value])}
            {line([value, 0, 1], [value, 1, 1])}{line([0, value, 1], [1, value, 1])}
            {line([0, value, 0], [0, value, 1])}{line([0, 0, value], [0, 1, value])}
          </g>)}
          {edges.map(([a, b]) => <g key={`${a}-${b}`}>{line(vertices[a], vertices[b], { stroke: '#adadad', strokeWidth: 1.1, opacity: .7 })}</g>)}
          {line([0, 0, 0], [1, 0, 0], { stroke: '#747474', strokeWidth: 2.5 })}
          {line([0, 0, 0], [0, 1, 0], { stroke: '#747474', strokeWidth: 2.5 })}
          {line([1, 0, 0], [1, 0, 1], { stroke: '#747474', strokeWidth: 2.5 })}
          {[0, .25, .5, .75, 1].map(value => <g key={value}>{axes.map((axis, index) => {
            const location = index === 0 ? [value, 0, 0] : index === 1 ? [0, value, 0] : [1, 0, value]
            const [x, y] = project(location)
            const tick = ranges[index][0] + value * (ranges[index][1] - ranges[index][0])
            return <text key={axis.key} x={x + (index === 1 ? -15 : index === 2 ? 17 : 0)} y={y + (index === 0 ? 22 : 4)} textAnchor={index === 1 ? 'end' : index === 2 ? 'start' : 'middle'} fill="#737373" className="mv-cube-tick">{Number(tick.toFixed(1))}</text>
          })}</g>)}
          {axes.map((axis, index) => {
            const location = index === 0 ? [.5, 0, 0] : index === 1 ? [0, 1, 0] : [1, 0, .5]
            const [x, y] = project(location)
            // The vertical axis's label sits above its top tick, right-aligned
            // with the ticks and on two lines, so it stays left of the cube.
            if (index === 1) {
              const [first, ...rest] = axis.label.split(' ')
              return <text key={axis.key} x={x - 15} y={y - 40} textAnchor="end" fill="#505050" className="mv-cube-axis">
                <tspan x={x - 15}>{first}</tspan><tspan x={x - 15} dy="1.25em">{[...rest, '(%) ↑'].join(' ')}</tspan>
              </text>
            }
            return <text key={axis.key} x={x + (index === 2 ? 34 : 0)} y={index === 0 ? Math.max(...vertices.map(vertex => project(vertex)[1])) + 48 : y + 36} textAnchor={index === 2 ? 'start' : 'middle'} fill="#505050" className="mv-cube-axis">{axis.label} (%) ↑</text>
          })}
          {active?.experience != null && active?.interaction != null && <g stroke={active.color} strokeWidth="1.6" strokeDasharray="4 5" opacity=".6">
            {line(coordinates(active), [coordinates(active)[0], 0, coordinates(active)[2]])}
            {line([coordinates(active)[0], 0, coordinates(active)[2]], [coordinates(active)[0], 0, 0])}
            {line([coordinates(active)[0], 0, coordinates(active)[2]], [1, 0, coordinates(active)[2]])}
            {line(coordinates(active), [0, coordinates(active)[1], coordinates(active)[2]])}
            {line([0, coordinates(active)[1], coordinates(active)[2]], [0, coordinates(active)[1], 0])}
          </g>}
          {[...eligible].sort((a, b) => {
            if (focused && a.id === selected) return 1
            if (focused && b.id === selected) return -1
            const depth = row => { const [x, y, z] = coordinates(row); return (-x * Math.sin(radians) + z * Math.cos(radians)) * Math.cos(elevation) - y * Math.sin(elevation) }
            return depth(b) - depth(a)
          }).map(row => {
            const [x, y] = project(coordinates(row)), index = rows.findIndex(item => item.id === row.id)
            const isBest = best.some(item => item.id === row.id)
            const muted = focused && selected !== row.id
            const pointColor = muted ? '#c5c9c7' : row.color
            return <g key={row.id} className={`mv-cube-point${muted ? ' is-muted' : ''}${focused && selected === row.id ? ' is-focused' : ''}`} role="button" tabIndex="0" aria-pressed={focused && selected === row.id} aria-label={`${row.name}, task ${percent(row.task)}, generation quality ${percent(row.experience)}, interaction ${percent(row.interaction)}`} onClick={() => onSelect(row.id)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(row.id) } }}>
              <title>{row.name} · task {percent(row.task)} · generation quality {percent(row.experience)} · interaction {percent(row.interaction)}{isBest ? ' · Pareto frontier' : ''}</title>
              <circle cx={x} cy={y} r={focused && selected === row.id ? 22 : 19} fill="none" stroke={pointColor} strokeWidth="1.5" opacity=".65" />
              <circle cx={x} cy={y} r={focused && selected === row.id ? 16 : 13} fill={pointColor} stroke="white" strokeWidth="2.5" />
              <text x={x} y={y + 4.5} textAnchor="middle" fill={muted ? '#6e7671' : 'white'} fontSize="12" fontWeight="700">{index + 1}</text>
            </g>
          })}
        </svg>
      </> : <div className="mv-cube-empty"><span>3D comparison unavailable</span><h3>No submission has all three scores yet.</h3><p>The cube needs Task completion, Interaction and Generation quality. Available scores are ranked in the leaderboard below.</p></div>}
    </div>
    <aside className="mv-cube-sidebar"><span className="mv-kicker">SELECT A MODEL</span><div className="mv-cube-models">{rows.map((row, index) => <button key={row.id} className={focused && selected !== row.id ? 'is-muted' : ''} aria-pressed={focused && selected === row.id} onClick={() => onSelect(row.id)}><span className="mv-model-number" style={{ background: focused && selected !== row.id ? '#c5c9c7' : row.color, color: 'white' }}>{index + 1}</span><span>{row.name}<small>{row.org}</small></span>{best.some(item => item.id === row.id) && <span className="mv-frontier-tag" title="On the observed 3D Pareto frontier">Pareto</span>}</button>)}</div><div className="mv-selected-metrics"><span className="mv-kicker">{active?.name}</span>{axes.map(axis => <div key={axis.key}><span>{axis.label}</span><strong>{percent(active?.[axis.key])}</strong></div>)}</div></aside>
  </div>
}

// Interaction and Generation quality come from each package's quality file and
// cover the three core domains; Banking and Intake are scored on task
// completion only.
const domainOptions = [
  { key: 'all', label: '📊 Overall' },
  { key: 'retail', label: '🛍️ Retail' },
  { key: 'airline', label: '✈️ Airline' },
  { key: 'telecom', label: '📱 Telecom' },
  { key: 'banking_knowledge', label: '🏦 Banking' },
  { key: 'intake', label: '📋 Intake' },
]
const qualityDomain = domain => domain === 'all' || CORE_DOMAINS.includes(domain) ? domain : null
const scoreFor = (pkg, metric, language, domain) => {
  if (metric === 'task') return domain === 'all' ? packageScore(pkg, language) : domainScore(pkg, domain, language)
  const scope = qualityDomain(domain)
  return scope ? packageQuality(pkg, metric, language, scope) : null
}

function DomainToggle({ domain, setDomain }) {
  const index = domainOptions.findIndex(option => option.key === domain)
  return <div className="domain-toggle-switch mv-domain-toggle">
    <div className="toggle-container domain-toggle-container" style={{ '--domain-count': domainOptions.length }}>
      {domainOptions.map(option => <button key={option.key} className={`toggle-option domain-toggle-option ${domain === option.key ? 'active' : ''}`} aria-pressed={domain === option.key} onClick={() => setDomain(option.key)}>{option.label}</button>)}
      <div className="toggle-slider domain-toggle-slider" style={{ width: `calc((100% - 8px) / ${domainOptions.length})`, transform: `translateX(${index * 100}%)` }} />
    </div>
  </div>
}

function LanguageMap({ rows, metrics, metric, setMetric, domain }) {
  const covered = new Set(rows.flatMap(row => [row.pkg.submission, ...Object.values(row.pkg.extras || {}).map(part => part.submission)].filter(Boolean).flatMap(part => part.frame.cells.map(cell => cell.language))))
  const columns = [...LANGUAGES.filter(language => covered.has(language.id)), { id: 'all', name: 'Overall' }]
  const table = rows.map(row => ({ ...row, values: Object.fromEntries(columns.map(language => [language.id, scoreFor(row.pkg, metric, language.id, domain)])) }))
  table.push({ id: 'overall', name: 'Overall', values: Object.fromEntries(columns.map(language => [language.id, mean(table.map(row => row.values[language.id]))])) })
  const metricLabel = metrics.find(([key]) => key === metric)[1]
  const heat = heatScale(table.flatMap(row => columns.map(language => row.values[language.id])))
  return <div className="mv-map">
    <div className="mv-map-heading"><div className="mv-range-switch mv-map-metrics" role="group" aria-label="Language map metric">{metrics.map(([key, label]) => <button key={key} type="button" aria-pressed={metric === key} onClick={() => setMetric(key)}>{label}</button>)}</div></div>
    <div className="mv-table-scroll"><table className="mv-heatmap" aria-label={`${metricLabel} by model and language`}>
      <thead><tr><th scope="col">Model</th>{columns.map(language => <th scope="col" key={language.id} className={language.id === 'all' ? 'mv-overall' : ''}>{language.name}</th>)}</tr></thead>
      <tbody>{table.map(row => <tr key={row.id} data-model={row.id} className={row.id === 'overall' ? 'mv-overall-row' : ''}><th scope="row">{row.color && <i style={{ background: row.color }} />}{row.name}{row.org && <small>{row.org}</small>}</th>{columns.map(language => {
        const value = row.values[language.id]
        return <td key={language.id} data-language={language.id} className={language.id === 'all' ? 'mv-overall' : ''}><span className="mv-heat-cell" style={{ background: heat(value) }} title={`${row.name} · ${language.name} · ${metricLabel}: ${percent(value)}`}>{percent(value)}</span></td>
      })}</tr>)}</tbody>
    </table></div>
    <p className="mv-footnote">Task completion: mean over cells, each language–domain cell weighted equally. Interaction and Generation quality: share of scored calls that pass, pooled over calls. Overall row: mean across models. “—” means not evaluated. Shading is relative to the lowest and highest values in the table.</p>
  </div>
}

function PackageRow({ row, index, columns, open, metric, onOpen, onClose }) {
  const { pkg } = row
  const logo = packageLogo(pkg)
  const toggle = () => open ? onClose() : onOpen('task')
  return <Fragment>
    <tr className={`model-row mv-package-row${open ? ' expanded' : ''}`} data-package={pkg.dir}>
      <td className="rank-cell"><span className={`rank-number ${['rank-gold', 'rank-silver', 'rank-bronze'][index] || ''}`}>#{index + 1}</span></td>
      <td className="model-info"><button className="mv-model-trigger" onClick={toggle} aria-expanded={open}><span className="model-name">{row.name}</span></button>{row.submissionType === 'custom' && <span className="legacy-badge">custom submission</span>}</td>
      <td className="organization-info"><div className="org-container">{logo && <div className="company-logo"><img src={`${import.meta.env.BASE_URL}${logo}`} alt="" className="logo-img" /></div>}<div className="org-text"><span className="org-name">{row.org}</span></div></div></td>
      {columns.map(([key, label]) => <td key={key} className="metric-cell score-cell" data-metric={key}><button className="score-bar-container mv-score-button" aria-label={`${row.name}: ${label} breakdown`} aria-expanded={open && metric === key} onClick={() => onOpen(key)}>
        {row[key] != null ? <><span className="score-bar-track"><span className="score-bar-fill" style={{ width: `${row[key]}%` }} /></span><span className="score-bar-value">{percent(row[key])}</span></> : <span className="no-data">—</span>}
      </button></td>)}
      <td className="expand-cell"><button className="mv-expand-trigger" aria-label={`${open ? 'Collapse' : 'Expand'} ${row.name} breakdown`} aria-expanded={open} onClick={toggle}><span className={`expand-caret${open ? ' open' : ''}`}>▶</span></button></td>
    </tr>
    {open && <tr className="domain-detail-row"><td colSpan={4 + columns.length} className="domain-detail-cell"><PackageBreakdown pkg={pkg} metric={metric} onMetricChange={onOpen} onClose={onClose} /></td></tr>}
  </Fragment>
}

export default function MultilingualVoice({ submissionTypeFilters, showStandard = true, showCustom = false }) {
  // ?domain= opens a domain tab (the banner links to Intake this way).
  const [domain, setDomain] = useState(() => {
    const requested = new URLSearchParams(window.location.search).get('domain')
    return domainOptions.some(option => option.key === requested) ? requested : 'all'
  })
  const domainLabel = domainOptions.find(option => option.key === domain).label.replace(/^\S+ /, '')
  const [view, setView] = useState(() => {
    const requested = new URLSearchParams(window.location.search).get('view')
    return requested === 'languages' ? 'languages' : 'space'
  })
  // Keep the chart view in the URL so a shared or refreshed link reopens it.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (view === 'languages') params.set('view', 'languages')
    else params.delete('view')
    if (domain !== 'all') params.set('domain', domain)
    else params.delete('domain')
    const next = `${window.location.pathname}?${params.toString()}`
    if (`${window.location.pathname}${window.location.search}` !== next) window.history.replaceState(null, '', next)
    // Leaving the τ-i18n view (English only, another benchmark) drops it.
    return () => {
      const left = new URLSearchParams(window.location.search)
      if (!left.has('view') && !left.has('domain')) return
      left.delete('view')
      left.delete('domain')
      window.history.replaceState(null, '', `${window.location.pathname}?${left.toString()}`)
    }
  }, [view, domain])
  const [mapMetric, setMapMetric] = useState('task')
  const [sort, setSort] = useState('task')
  // Every row is a submitted package. Banking and entity packages fill their
  // domain on the same agent's row; one without a voice-v2 package appears
  // only under its own domain.
  const { packages, errors: packageErrors } = useVoiceV2Packages()
  const [open, setOpen] = useState(null)
  const [breakdown, setBreakdown] = useState('task')
  const used = new Set()
  const colors = (packages || []).map((pkg, index) => {
    const own = providerColors[pkg.metadata.agent_provider]
    const color = own && !used.has(own) ? own : palette[index % palette.length]
    used.add(color)
    return color
  })
  const rowsFor = scope => (packages || []).map((pkg, index) => ({
    id: pkg.dir, pkg, color: colors[index],
    name: pkg.metadata.model_name,
    org: pkg.metadata.model_organization,
    submissionType: pkg.metadata.submission_type || 'standard',
    ...Object.fromEntries(metrics.map(([key]) => [key, scoreFor(pkg, key, 'all', scope)])),
  })).filter(row => (row.submissionType === 'custom' ? showCustom : showStandard) && (row.pkg.submission || row.task != null))
  const rows = rowsFor(domain)
  // Banking and Intake have task completion only, so their other columns are hidden.
  const extraDomain = domain === 'banking_knowledge' || domain === 'intake'
  // Intake's task completion is crossed Pass^3; the header says so.
  const shown = (extraDomain ? metrics.filter(([key]) => key === 'task') : metrics)
    .map(([key, label]) => key === 'task' && PASS3_DOMAINS.includes(domain) ? [key, `${label} (pass^3)`] : [key, label])
  const sortKey = shown.some(([key]) => key === sort) ? sort : 'task'
  const mapKey = shown.some(([key]) => key === mapMetric) ? mapMetric : 'task'
  const ranked = [...rows].sort((a, b) => a[sortKey] == null ? b[sortKey] == null ? 0 : 1 : b[sortKey] == null ? -1 : b[sortKey] - a[sortKey])
  const byTask = list => list.filter(row => row.task != null).sort((a, b) => b.task - a.task)
  const charted = byTask(rows)
  // The 3D explorer always shows Overall; the domain toggle drives the
  // leaderboard and the language map only.
  const spaceRows = byTask(rowsFor('all'))
  const panelRows = view === 'space' ? spaceRows : charted
  const panelLabel = view === 'space' || domain === 'all' ? '' : `${domainLabel} `
  const [picked, setPicked] = useState(null)
  const [focused, setFocused] = useState(false)
  const selected = spaceRows.some(row => row.id === picked) ? picked : spaceRows[0]?.id
  const selectModel = id => {
    setFocused(!(focused && selected === id))
    setPicked(id)
  }
  const openBreakdown = (dir, metric) => { setOpen(dir); setBreakdown(metric) }
  const coreParts = (packages || []).filter(pkg => pkg.submission)
  const calls = coreParts.reduce((sum, pkg) => sum + pkg.submission.summary.scored_trials, 0)
  const loading = packages == null
  return <section className="mv" aria-label="Multilingual voice benchmark">
    <div className="mv-toolbar"><div className="mv-view-tabs" aria-label="Chart views">{[['space', '01', '3D explorer'], ['languages', '02', 'Language map']].map(([id, number, label]) => <button key={id} aria-pressed={view === id} className={view === id ? 'active' : ''} onClick={() => setView(id)}><span>{number}</span>{label}</button>)}</div></div>
    <div className="mv-main-panel">
      {view === 'space' && <div className="mv-panel-heading"><h2>Three dimensions of a good conversation.</h2><span className="mv-core-note">Core domains only · Airline, Retail, Telecom</span></div>}
      {loading ? <div className="mv-cube-empty"><span>Loading</span><h3>Loading submissions…</h3></div>
        : !panelRows.length ? <div className="mv-cube-empty"><span>Not yet evaluated</span><h3>No {panelLabel}results yet.</h3><p>Submitted packages appear here once they are added to the leaderboard.</p></div>
        : view === 'languages' ? <LanguageMap rows={charted} metrics={shown} metric={mapKey} setMetric={setMapMetric} domain={domain} />
        : <Space rows={spaceRows} onSelect={selectModel} selected={selected} focused={focused} onClearFocus={() => setFocused(false)} />}
      <div className="mv-panel-footer"><span><strong>{calls.toLocaleString()}</strong> scored calls</span><span><strong>{coreParts.length}</strong> {coreParts.length === 1 ? 'submission' : 'submissions'}</span><span><strong>3</strong> core domains</span></div>
    </div>
    <div className="mv-leaderboard-heading"><h2>Leaderboard</h2><div className="mv-leaderboard-controls"><DomainToggle domain={domain} setDomain={setDomain} />{submissionTypeFilters}</div></div>
    <div className="mv-ranking-meta"><span>All languages{domain !== 'all' && ` · ${domainLabel}`}</span><span>Click a score to see its breakdown.</span></div>
    <div className="reliability-metrics mv-scores">
      <div className="metrics-table-container">
        <table className="reliability-table mv-leaderboard-table">
          <thead><tr>
            <th>Rank</th><th>Model</th><th>Provider</th>
            {shown.map(([key, label]) => <th key={key} className="mv-metric-header" aria-sort={sortKey === key ? 'descending' : 'none'}><button onClick={() => setSort(key)}>{label} ↑{sortKey === key && <span className="mv-sort-dot"> •</span>}</button></th>)}
            <th className="expand-header"><span className="mv-visually-hidden">Breakdown</span></th>
          </tr></thead>
          <tbody>{ranked.map((row, index) => <PackageRow key={row.id} row={row} index={index} columns={shown} open={open === row.id} metric={breakdown} onOpen={metric => openBreakdown(row.id, metric)} onClose={() => setOpen(null)} />)}
            {!loading && !ranked.length && <tr><td colSpan={4 + shown.length}><div className="filter-empty-state"><h3>No Results</h3><p>{!showStandard && !showCustom ? 'Select Standard or Custom to view results.' : showCustom && !showStandard ? 'No custom multilingual submissions yet.' : 'No submissions yet.'}</p></div></td></tr>}</tbody>
        </table>
      </div>
    </div>
    <div className="mv-factor-notes">
      <p className="mv-footnote"><strong>Task completion</strong>: pass@1 over the frame, each language–domain cell weighted equally; Overall covers Airline, Retail and Telecom. Intake is scored by pass^3: the share of tasks passed in all three voice conditions (regular, heavy channel noise, heavy speech effects). <strong>Interaction</strong>: share of calls with fewer than two of five conversational failures (non-response, more than three interruptions, more than three selectivity errors, a monologue, a tool-use error). <strong>Generation quality</strong>: share of calls where at most 25% of the agent’s judged utterances have a naturalness or speech-fidelity issue. Calls whose judgments are incomplete, and infrastructure errors, are left out of both.</p>
    </div>
    {packageErrors.length > 0 && <p className="mv-footnote" role="alert">Skipped submissions: {packageErrors.join('; ')}</p>}
    <nav className="mv-paper-links" aria-label="Research papers"><span>Papers</span>{[
      ['τ-i18n', '2609.35820', 'τ-Multilingual: Benchmarking Voice Agents Across Languages'],
      ['τ-Entity', '2609.13602', 'τ-Elicitation: Benchmarking multi-turn entity extraction in voice agents'],
      ['τ-Voice', '2603.13686', 'τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains'],
    ].map(([label, id, title]) => <a key={id} href={`https://arxiv.org/abs/${id}`} title={title} target="_blank" rel="noopener noreferrer">{label} ↗</a>)}</nav>
  </section>
}
