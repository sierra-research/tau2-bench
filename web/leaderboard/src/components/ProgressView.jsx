import React, { useMemo, useState } from 'react'
import './ProgressView.css'

const ORG_LOGOS = {
  'Anthropic': 'claude.png',
  'OpenAI': 'openai.svg',
  'Google': 'Google__G__logo.svg.png',
  'xAI': 'xai-logo.svg',
  'DeepSeek': 'DeepSeek_logo_icon.png',
  'Qwen': 'qwen-color.png',
  'Alibaba Cloud': 'qwen-color.png',
  'NVIDIA': 'Logo-nvidia-transparent-PNG.png',
  'Moonshot AI': null, // emoji fallback
}

const ORG_TINT = {
  'OpenAI': '#10A37F',
  'Anthropic': '#C26B43',
  'Google': '#4285F4',
  'xAI': '#111111',
  'DeepSeek': '#4D6BFE',
  'Qwen': '#615CED',
  'Alibaba Cloud': '#615CED',
  'NVIDIA': '#76B900',
  'Moonshot AI': '#000000',
  'Zhipu AI': '#3268FB',
  'Distyl AI': '#7C3AED',
}

const ORG_EMOJI = {
  'Moonshot AI': '🚀',
  'Zhipu AI': '🧠',
  'Distyl AI': '◆',
}

const formatMonth = (d) => d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
const formatDate = (d) => d.toISOString().slice(0, 10)

/**
 * Compute the score the chart should plot for a given model+domain pair.
 * For 'overall', match the leaderboard table semantics: average pass_1 across
 * all available core domains for the benchmark, but only when every core
 * domain has data. For a specific domain, just use that domain's pass_1.
 */
const computeScore = (model, domain, benchmark) => {
  if (domain === 'overall') {
    const overallDomains = benchmark === 'voice'
      ? ['retail', 'airline', 'telecom']
      : ['retail', 'airline', 'telecom', 'banking_knowledge']
    const vals = overallDomains
      .map((d) => model[d]?.[0])
      .filter((v) => v !== null && v !== undefined)
    if (vals.length !== overallDomains.length) return null
    return vals.reduce((a, b) => a + b, 0) / vals.length
  }
  const v = model[domain]?.[0]
  return v === null || v === undefined ? null : v
}

/**
 * Pull a "release date" for each entry from model_release.release_date when
 * available, otherwise fall back to methodology.evaluation_date or
 * submission_date and tag the entry as "eval" so we can render it differently.
 *
 * Filters mirror the leaderboard table: benchmark modality, submission type
 * (standard/custom), and legacy toggle. When the same model_name appears more
 * than once (e.g. a model has both a legacy and a current submission), the
 * higher-scoring entry wins.
 */
const extractEntries = (
  passKData,
  fullSubmissionData,
  benchmark,
  domain,
  { showStandard, showCustom, showLegacy }
) => {
  const passesFilters = (model) => {
    if (model.modality !== benchmark) return false
    if (model.isLegacy && !showLegacy) return false
    const isStandard = model.submissionType === 'standard' || !model.submissionType
    const isCustom = model.submissionType === 'custom'
    if (isStandard && !showStandard) return false
    if (isCustom && !showCustom) return false
    return true
  }

  const bestByModel = new Map()
  for (const [key, model] of Object.entries(passKData)) {
    if (!passesFilters(model)) continue
    const score = computeScore(model, domain, benchmark)
    if (score === null) continue
    const sub = fullSubmissionData[key] || {}
    const modelName = sub.model_name || model.modelName
    const prev = bestByModel.get(modelName)
    if (!prev || score > prev.score) {
      bestByModel.set(modelName, { key, score })
    }
  }

  const entries = []
  for (const [key, model] of Object.entries(passKData)) {
    if (!passesFilters(model)) continue
    const score = computeScore(model, domain, benchmark)
    if (score === null) continue
    const sub = fullSubmissionData[key] || {}
    const modelName = sub.model_name || model.modelName
    if (bestByModel.get(modelName)?.key !== key) continue
    const release = sub.model_release || null
    const releaseDate = release?.release_date || null
    const evalDate = sub.methodology?.evaluation_date || sub.submission_date || null
    const dateStr = releaseDate || evalDate
    if (!dateStr) continue
    entries.push({
      key,
      modelName,
      modelOrganization: sub.model_organization || model.organization,
      submittingOrganization: sub.submitting_organization || model.organization,
      submissionType: model.submissionType,
      isLegacy: model.isLegacy,
      score,
      date: new Date(dateStr + 'T00:00:00Z'),
      hasReleaseDate: !!releaseDate,
      announcementUrl: release?.announcement_url || null,
      announcementTitle: release?.announcement_title || null,
    })
  }
  entries.sort((a, b) => a.date - b.date)
  return entries
}

const computeFrontier = (entries) => {
  const frontierKeys = new Set()
  let running = -Infinity
  for (const e of entries) {
    if (e.score > running) {
      running = e.score
      frontierKeys.add(e.key)
    }
  }
  return frontierKeys
}

const DOMAIN_LABEL = {
  overall: 'Overall',
  retail: 'Retail',
  airline: 'Airline',
  telecom: 'Telecom',
  banking_knowledge: 'Banking',
}

const ProgressView = ({
  passKData,
  fullSubmissionData,
  benchmark,
  domain,
  showStandard,
  showCustom,
  showLegacy,
  baseUrl,
}) => {
  const [hoverKey, setHoverKey] = useState(null)
  const [showAllLabels, setShowAllLabels] = useState(false)

  const entries = useMemo(
    () => extractEntries(passKData, fullSubmissionData, benchmark, domain, {
      showStandard, showCustom, showLegacy,
    }),
    [passKData, fullSubmissionData, benchmark, domain, showStandard, showCustom, showLegacy]
  )
  const frontierKeys = useMemo(() => computeFrontier(entries), [entries])

  const benchmarkLabel = benchmark === 'voice' ? 'τ-voice' : 'τ-bench'
  const domainLabel = DOMAIN_LABEL[domain] || domain

  if (entries.length === 0) {
    return (
      <div className="progress-empty">
        <p>
          No submissions match the current filters for {benchmarkLabel} ·{' '}
          {domainLabel.toLowerCase()}.
          {!showStandard && !showCustom && ' Enable Standard or Custom to see results.'}
        </p>
      </div>
    )
  }

  // SVG layout
  const W = 1080
  const H = 560
  const marginTop = 56
  const marginBottom = 80
  const marginLeft = 64
  const marginRight = 32
  const plotX0 = marginLeft
  const plotX1 = W - marginRight
  const plotY0 = marginTop
  const plotY1 = H - marginBottom
  const plotW = plotX1 - plotX0
  const plotH = plotY1 - plotY0

  const yMax = 100

  const minDate = entries[0].date
  const maxDate = entries[entries.length - 1].date
  const axisStart = new Date(Date.UTC(minDate.getUTCFullYear(), minDate.getUTCMonth(), 1))
  const axisEnd = new Date(Date.UTC(maxDate.getUTCFullYear(), maxDate.getUTCMonth() + 1, 1))
  const totalDays = Math.max(1, (axisEnd - axisStart) / (1000 * 60 * 60 * 24))

  const xOf = (d) => plotX0 + ((d - axisStart) / (1000 * 60 * 60 * 24) / totalDays) * plotW
  const yOf = (v) => plotY1 - (v / yMax) * plotH

  // Compute positions; nudge same-date points
  const positioned = entries.map((e) => ({ ...e, x: xOf(e.date), y: yOf(e.score) }))
  const dateGroups = {}
  for (const p of positioned) {
    const k = p.date.toISOString().slice(0, 10)
    ;(dateGroups[k] ||= []).push(p)
  }
  for (const grp of Object.values(dateGroups)) {
    if (grp.length > 1) {
      const span = 14
      const step = (span * 2) / (grp.length - 1)
      grp.forEach((p, i) => {
        p.x += -span + i * step
      })
    }
  }

  // Frontier line points (only running-max entries)
  const frontierPts = positioned.filter((p) => frontierKeys.has(p.key))
  // Step path: rise vertically at each new point's x, then horizontal to next
  let frontierPath = ''
  if (frontierPts.length > 0) {
    frontierPath = `M ${frontierPts[0].x.toFixed(1)} ${frontierPts[0].y.toFixed(1)}`
    for (let i = 1; i < frontierPts.length; i++) {
      const prev = frontierPts[i - 1]
      const cur = frontierPts[i]
      frontierPath += ` L ${cur.x.toFixed(1)} ${prev.y.toFixed(1)} L ${cur.x.toFixed(1)} ${cur.y.toFixed(1)}`
    }
    // Extend horizontally to right edge
    frontierPath += ` L ${plotX1.toFixed(1)} ${frontierPts[frontierPts.length - 1].y.toFixed(1)}`
  }

  // Filled area under the frontier
  let areaPath = ''
  if (frontierPts.length > 0) {
    areaPath = `M ${frontierPts[0].x.toFixed(1)} ${plotY1.toFixed(1)}`
    areaPath += ` L ${frontierPts[0].x.toFixed(1)} ${frontierPts[0].y.toFixed(1)}`
    for (let i = 1; i < frontierPts.length; i++) {
      const prev = frontierPts[i - 1]
      const cur = frontierPts[i]
      areaPath += ` L ${cur.x.toFixed(1)} ${prev.y.toFixed(1)} L ${cur.x.toFixed(1)} ${cur.y.toFixed(1)}`
    }
    areaPath += ` L ${plotX1.toFixed(1)} ${frontierPts[frontierPts.length - 1].y.toFixed(1)}`
    areaPath += ` L ${plotX1.toFixed(1)} ${plotY1.toFixed(1)} Z`
  }

  // Y gridlines every 10%
  const yTicks = []
  for (let v = 0; v <= yMax; v += 10) yTicks.push(v)

  // X tick months (sample at most ~10 to avoid clutter)
  const allMonths = []
  let cur = new Date(axisStart)
  while (cur <= axisEnd) {
    allMonths.push(new Date(cur))
    cur = new Date(Date.UTC(cur.getUTCFullYear(), cur.getUTCMonth() + 1, 1))
  }
  const monthStride = Math.max(1, Math.ceil(allMonths.length / 12))
  const xTicks = allMonths.filter((_, i) => i % monthStride === 0)

  // Label decision: show all if toggled, otherwise frontier-only
  const labeledKeys = showAllLabels
    ? new Set(positioned.map((p) => p.key))
    : frontierKeys

  // Auto-pick anchor based on x position so labels never overflow horizontally.
  const pickAnchor = (x) => {
    if (x < plotX0 + 80) return 'start'
    if (x > plotX1 - 80) return 'end'
    return 'middle'
  }
  const labelOffsets = {}
  const frontierByX = [...frontierPts].sort((a, b) => a.x - b.x)
  frontierByX.forEach((p, i) => {
    labelOffsets[p.key] = { dy: i % 2 === 0 ? -34 : 38, anchor: pickAnchor(p.x), dx: 0 }
  })
  if (showAllLabels) {
    positioned.forEach((p, i) => {
      if (!labelOffsets[p.key]) {
        labelOffsets[p.key] = { dy: i % 2 === 0 ? -28 : 30, anchor: pickAnchor(p.x), dx: 0 }
      }
    })
  }

  const dotR = benchmark === 'voice' ? 16 : 13
  const logoSize = benchmark === 'voice' ? 20 : 16

  const renderLogo = (org, cx, cy) => {
    const file = ORG_LOGOS[org]
    if (file) {
      return (
        <image
          href={`${baseUrl}${file}`}
          x={cx - logoSize / 2}
          y={cy - logoSize / 2}
          width={logoSize}
          height={logoSize}
          preserveAspectRatio="xMidYMid meet"
        />
      )
    }
    const emoji = ORG_EMOJI[org] || org?.[0] || '?'
    return (
      <text
        x={cx}
        y={cy + logoSize / 3}
        textAnchor="middle"
        fontSize={logoSize - 2}
        fontWeight="700"
        fill={ORG_TINT[org] || '#1f2937'}
      >
        {emoji}
      </text>
    )
  }

  const subTitle = `Pass@1 on ${benchmarkLabel} · ${domainLabel.toLowerCase()} ` +
    `· plotted at each model's public release date. ${entries.length} model${entries.length === 1 ? '' : 's'}.`

  return (
    <div className="progress-view">
      <div className="progress-card">
        <div className="progress-card-head">
          <div>
            <div className="progress-card-title">{benchmarkLabel} {domainLabel} pass@1 by release date</div>
            <div className="progress-card-sub">{subTitle}</div>
          </div>
          <div className="progress-card-controls">
            <label className="checkbox-container progress-toggle">
              <input
                type="checkbox"
                checked={showAllLabels}
                onChange={(e) => setShowAllLabels(e.target.checked)}
              />
              <span className="checkbox-checkmark"></span>
              <span className="checkbox-label">Label every model</span>
            </label>
          </div>
        </div>

        <div className="progress-svg-wrap">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            xmlns="http://www.w3.org/2000/svg"
            role="img"
            aria-label={`${benchmarkLabel} progress over time`}
            className="progress-svg"
          >
            <defs>
              <linearGradient id="progress-frontier-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#047857" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#047857" stopOpacity="0.02" />
              </linearGradient>
              <filter id="progress-dot-shadow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceAlpha" stdDeviation="2" />
                <feOffset dx="0" dy="1" />
                <feComponentTransfer>
                  <feFuncA type="linear" slope="0.25" />
                </feComponentTransfer>
                <feMerge>
                  <feMergeNode />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* plot bg */}
            <rect
              x={plotX0}
              y={plotY0}
              width={plotW}
              height={plotH}
              fill="#ffffff"
              stroke="#e2e8f0"
              strokeWidth="1"
              rx="8"
            />

            {/* y gridlines + labels */}
            {yTicks.map((v) => (
              <g key={`y-${v}`}>
                <line
                  x1={plotX0}
                  y1={yOf(v)}
                  x2={plotX1}
                  y2={yOf(v)}
                  stroke="#eef2f7"
                  strokeWidth="1"
                />
                <text
                  x={plotX0 - 10}
                  y={yOf(v) + 4}
                  textAnchor="end"
                  fontSize="11"
                  fill="#64748b"
                >
                  {v}%
                </text>
              </g>
            ))}

            {/* y axis title */}
            <text
              x={plotX0 - 44}
              y={(plotY0 + plotY1) / 2}
              transform={`rotate(-90 ${plotX0 - 44} ${(plotY0 + plotY1) / 2})`}
              textAnchor="middle"
              fontSize="11"
              fill="#475569"
              fontWeight="600"
              letterSpacing="0.08em"
            >
              {domainLabel.toUpperCase()} PASS@1
            </text>

            {/* x ticks (months) */}
            {xTicks.map((m, i) => {
              const xx = xOf(m)
              return (
                <g key={`x-${i}`}>
                  <line
                    x1={xx}
                    y1={plotY1}
                    x2={xx}
                    y2={plotY1 + 5}
                    stroke="#cbd5e1"
                    strokeWidth="1"
                  />
                  <text
                    x={xx}
                    y={plotY1 + 20}
                    textAnchor="middle"
                    fontSize="11"
                    fill="#64748b"
                  >
                    {formatMonth(m)}
                  </text>
                </g>
              )
            })}

            {/* frontier */}
            {areaPath && <path d={areaPath} fill="url(#progress-frontier-fill)" />}
            {frontierPath && (
              <path
                d={frontierPath}
                fill="none"
                stroke="#047857"
                strokeWidth="2.5"
                strokeLinejoin="round"
                strokeLinecap="round"
                strokeDasharray="6,4"
              />
            )}

            {/* dots */}
            {positioned.map((p) => {
              const isFrontier = frontierKeys.has(p.key)
              const tint = ORG_TINT[p.modelOrganization] || '#888'
              const isHover = hoverKey === p.key
              return (
                <g
                  key={p.key}
                  className={`progress-dot${isFrontier ? ' frontier' : ''}${isHover ? ' hover' : ''}`}
                  onMouseEnter={() => setHoverKey(p.key)}
                  onMouseLeave={() => setHoverKey(null)}
                  style={{ cursor: 'pointer' }}
                >
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={dotR}
                    fill="#ffffff"
                    stroke="#e2e8f0"
                    strokeWidth="1.5"
                    filter="url(#progress-dot-shadow)"
                  />
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={dotR - 2}
                    fill="none"
                    stroke={tint}
                    strokeWidth="1.5"
                    strokeDasharray={p.hasReleaseDate ? undefined : '3,2'}
                    opacity={isFrontier ? 0.9 : 0.45}
                  />
                  {renderLogo(p.modelOrganization, p.x, p.y)}
                  <title>
                    {p.modelName} ({p.modelOrganization}) · {formatDate(p.date)}
                    {p.hasReleaseDate ? '' : ' (eval date)'} · {p.score.toFixed(1)}% {domainLabel.toLowerCase()}
                  </title>
                </g>
              )
            })}

            {/* labels */}
            {positioned
              .filter((p) => labeledKeys.has(p.key) || hoverKey === p.key)
              .map((p) => {
                const off = labelOffsets[p.key] || { dy: -34, anchor: 'middle', dx: 0 }
                const lx = p.x + (off.dx || 0)
                const ly = p.y + off.dy
                const isFrontier = frontierKeys.has(p.key)
                const isHover = hoverKey === p.key

                // Approximate text widths (avg glyph ~ 0.58em for our fonts)
                const scoreStr = `${p.score.toFixed(1)}%`
                const nameW = p.modelName.length * 12 * 0.58
                const scoreW = scoreStr.length * 11 * 0.58
                const textW = Math.max(nameW, scoreW)
                const padX = 6
                const rectW = textW + padX * 2
                const padTop = 11
                const padBottom = 7
                // model name baseline at ly, score baseline at ly+14
                const rectH = padTop + 14 + padBottom

                let rectX
                if (off.anchor === 'start') rectX = lx - padX
                else if (off.anchor === 'end') rectX = lx - rectW + padX
                else rectX = lx - rectW / 2
                const rectY = ly - padTop

                return (
                  <g key={`label-${p.key}`} className="progress-label">
                    {isHover && (
                      <rect
                        x={rectX}
                        y={rectY}
                        width={rectW}
                        height={rectH}
                        fill="white"
                        fillOpacity={0.95}
                        stroke="#e2e8f0"
                        rx="4"
                      />
                    )}
                    {/* Halo: render the same text twice with a thick white
                        stroke underneath the fill so labels stay legible on
                        top of gridlines and the frontier line/fill. */}
                    <text
                      x={lx}
                      y={ly}
                      textAnchor={off.anchor}
                      fontSize="12"
                      fontWeight={isFrontier ? 700 : 600}
                      fill="#0f172a"
                      stroke="#ffffff"
                      strokeWidth="3.5"
                      strokeLinejoin="round"
                      paintOrder="stroke"
                    >
                      {p.modelName}
                    </text>
                    <text
                      x={lx}
                      y={ly + 14}
                      textAnchor={off.anchor}
                      fontSize="11"
                      fontWeight={isFrontier ? 700 : 500}
                      fill={isFrontier ? '#047857' : '#334155'}
                      stroke="#ffffff"
                      strokeWidth="3"
                      strokeLinejoin="round"
                      paintOrder="stroke"
                    >
                      {scoreStr}
                    </text>
                  </g>
                )
              })}
          </svg>
        </div>

        <div className="progress-legend">
          {Array.from(new Set(positioned.map((p) => p.modelOrganization))).map((org) => (
            <span className="progress-legend-item" key={org}>
              <span className="progress-legend-logo">
                {ORG_LOGOS[org] ? (
                  <img src={`${baseUrl}${ORG_LOGOS[org]}`} alt={org} />
                ) : (
                  <span style={{ color: ORG_TINT[org] || '#1f2937' }}>
                    {ORG_EMOJI[org] || org?.[0] || '?'}
                  </span>
                )}
              </span>
              {org}
            </span>
          ))}
          <span className="progress-legend-item progress-legend-frontier">
            <span className="progress-legend-line" />
            Frontier (best so far)
          </span>
        </div>

        <div className="progress-footnote">
          Markers show each model's overall pass@1 plotted at its public release date.
          Hover any marker for details. The dashed line tracks the running best.
          Submissions without a published <code>model_release.release_date</code> fall back
          to the evaluation date.
        </div>
      </div>
    </div>
  )
}

export default ProgressView
