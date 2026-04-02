import { useMemo, useState } from 'react'
import { scoreColor, CLASS_COLORS, DIMENSION_COLORS } from '../utils/colors'
import { METRIC_DEFINITIONS, TASK_CLASS_EXPLANATIONS, generateRecommendations } from '../utils/metrics'

export default function Overview({ summary, tasks, conversations, faultData, safetyData, onDimensionClick, onViewConversations }) {
  if (!summary) return null
  const [expandedMetric, setExpandedMetric] = useState(null)
  const [expandedClass, setExpandedClass] = useState(null)

  const dims = summary.dimensions || {}
  const tc = summary.task_classes || {}
  const totalTasks = Object.values(tc).reduce((a, b) => a + b, 0) || 1
  const eff = summary.efficiency || {}

  const recommendations = useMemo(
    () => generateRecommendations(summary, tasks, conversations, faultData, safetyData),
    [summary, tasks, conversations, faultData, safetyData]
  )

  // Map dimension keys to metric definitions
  const dimCards = [
    ...Object.entries(dims).map(([key, dim]) => ({
      key, score: dim.score, label: dim.label, question: dim.question,
      dimKey: key === 'outcome_consistency' ? 'consistency' : key === 'policy_compliance' ? 'policy' : key.includes('action') || key.includes('sequence') ? 'consistency' : 'efficiency',
      definition: METRIC_DEFINITIONS[key],
    })),
    ...(summary.abstention ? [{
      key: 'abstention', score: null, label: 'Abstentions',
      question: `${summary.abstention.count}/${summary.abstention.total} conversations`,
      dimKey: 'efficiency', definition: METRIC_DEFINITIONS.abstention,
      customScore: `${summary.abstention.count}/${summary.abstention.total}`,
      customColor: summary.abstention.rate > 0.1 ? '#f59e0b' : '#10b981',
    }] : []),
    ...(faultData ? [{
      key: 'fault_tolerance', score: faultData.r_fault, label: 'Fault Tolerance',
      question: 'Handles API failures?', dimKey: 'robustness',
      definition: METRIC_DEFINITIONS.fault_tolerance,
    }] : []),
    ...(safetyData ? [{
      key: 'safety', score: safetyData.safety_score, label: 'Safety',
      question: 'Avoids harmful actions?', dimKey: 'safety',
      definition: METRIC_DEFINITIONS.safety,
    }] : []),
  ]

  return (
    <div className="overview">
      {/* Hero */}
      <div className="hero">
        <h1>{summary.model || 'Agent'} <span className="hero-domain">on {summary.domain}</span></h1>
        <p className="hero-meta">
          {summary.total_conversations} conversations · {summary.num_tasks} tasks × {summary.num_trials} trials · <strong>{(summary.accuracy * 100).toFixed(1)}% accuracy</strong>
        </p>
      </div>

      {/* Methodology note */}
      <div className="methodology-banner">
        Click any metric card to see what it measures, how it works, and how to improve it.
      </div>

      {/* Dimension Cards */}
      <div className="dim-grid">
        {dimCards.map(card => (
          <div key={card.key} className={`dim-card ${expandedMetric === card.key ? 'expanded' : ''}`}
            style={{ '--dim-color': DIMENSION_COLORS[card.key] || '#6b7280' }}>
            <div className="dim-card-top" onClick={() => onDimensionClick(card.dimKey)}>
              <div className="dim-score" style={{ color: card.customColor || scoreColor(card.score) }}>
                {card.customScore || (card.score != null ? (card.score * 100).toFixed(0) + '%' : 'N/A')}
              </div>
              <div className="dim-label">{card.label}</div>
              <div className="dim-question">{card.question}</div>
              <div className="dim-bar"><div className="dim-fill" style={{ width: `${(card.score || 0) * 100}%`, background: card.customColor || scoreColor(card.score) }} /></div>
            </div>
            {card.definition && (
              <button className="dim-info-btn" onClick={e => { e.stopPropagation(); setExpandedMetric(expandedMetric === card.key ? null : card.key) }}>
                {expandedMetric === card.key ? '✕' : '?'}
              </button>
            )}
            {expandedMetric === card.key && card.definition && (
              <div className="dim-explanation">
                <p className="explain-text">{card.definition.definition}</p>
                <div className="explain-methodology">
                  <strong>How it works:</strong> {card.definition.methodology}
                </div>
                <div className="explain-interp">
                  {card.score != null && card.score >= 0.8 && card.definition.interpretation?.high && (
                    <span className="interp good">{card.definition.interpretation.high}</span>
                  )}
                  {card.score != null && card.score >= 0.5 && card.score < 0.8 && card.definition.interpretation?.medium && (
                    <span className="interp mid">{card.definition.interpretation.medium}</span>
                  )}
                  {card.score != null && card.score < 0.5 && card.definition.interpretation?.low && (
                    <span className="interp bad">{card.definition.interpretation.low}</span>
                  )}
                </div>
                {card.definition.howToImprove && (
                  <div className="explain-improve">
                    <strong>How to improve:</strong> {card.definition.howToImprove}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Task Classification */}
      <div className="section">
        <h2>Task Classification</h2>
        <p className="section-desc">Tasks grouped by their reliability profile. Click a class to learn more.</p>
        <div className="class-bar">
          {Object.entries(tc).filter(([_, c]) => c > 0).map(([cls, count]) => (
            <div key={cls} className="class-seg" style={{ width: `${(count / totalTasks) * 100}%`, background: CLASS_COLORS[cls] || '#6b7280' }}
              onClick={() => setExpandedClass(expandedClass === cls ? null : cls)}>
              <span>{count} {cls.replace(/_/g, ' ')}</span>
            </div>
          ))}
        </div>
        {expandedClass && TASK_CLASS_EXPLANATIONS[expandedClass] && (
          <div className="class-explanation">
            <strong>{TASK_CLASS_EXPLANATIONS[expandedClass].name}</strong>
            <p>{TASK_CLASS_EXPLANATIONS[expandedClass].definition}</p>
            <p className="class-implication">{TASK_CLASS_EXPLANATIONS[expandedClass].implication}</p>
            <p className="class-action">{TASK_CLASS_EXPLANATIONS[expandedClass].action}</p>
          </div>
        )}
      </div>

      {/* Recommendations — evidence-based */}
      {recommendations.length > 0 && (
        <div className="section">
          <h2>Recommendations</h2>
          <p className="section-desc">Actionable improvements ranked by impact. Each backed by specific conversation evidence.</p>
          <div className="recs-list">
            {recommendations.map((r, i) => (
              <div key={i} className="rec-card" onClick={() => r.dimension && onDimensionClick(r.dimension)}>
                <div className="rec-header">
                  <span className="rec-priority">#{r.priority}</span>
                  <span className="rec-title">{r.title}</span>
                </div>
                <p className="rec-detail">{r.detail}</p>
                <div className="rec-action-box">
                  <strong>Action:</strong> {r.action}
                </div>
                {r.evidence && r.evidence.length > 0 && (
                  <div className="rec-evidence">
                    <strong>Evidence:</strong>
                    {r.evidence.map((e, j) => <span key={j} className="evidence-tag">{e}</span>)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CTA */}
      <button className="cta-btn" onClick={onViewConversations}>
        Explore All {summary.total_conversations} Conversations →
      </button>
    </div>
  )
}
