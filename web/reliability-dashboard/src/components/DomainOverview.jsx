import { DIMENSION_COLORS, CLASS_COLORS, scoreColor } from '../utils/colors'

export default function DomainOverview({ summary, tasks, onViewConversations }) {
  if (!summary) return null

  const dims = summary.dimensions || {}
  const tc = summary.task_classes || {}
  const totalTasks = Object.values(tc).reduce((a, b) => a + b, 0) || 1

  return (
    <div className="domain-overview">
      {/* Hero */}
      <div className="hero-section">
        <h1>{summary.model}</h1>
        <p className="hero-meta">
          {summary.domain} domain · {summary.total_conversations} conversations ·
          {summary.num_tasks} tasks × {summary.num_trials} trials ·
          <strong> {(summary.accuracy * 100).toFixed(1)}% accuracy</strong>
        </p>
      </div>

      {/* Dimension Cards */}
      <div className="dimension-grid">
        {Object.entries(dims).map(([key, dim]) => (
          <div key={key} className="dimension-card" style={{ borderTopColor: DIMENSION_COLORS[key] || '#6b7280' }}>
            <div className="dim-score" style={{ color: scoreColor(dim.score) }}>
              {dim.score != null ? (dim.score * 100).toFixed(0) + '%' : 'N/A'}
            </div>
            <div className="dim-label">{dim.label}</div>
            <div className="dim-question">{dim.question}</div>
            <div className="dim-bar">
              <div className="dim-bar-fill" style={{
                width: `${(dim.score || 0) * 100}%`,
                background: DIMENSION_COLORS[key] || '#6b7280',
              }} />
            </div>
          </div>
        ))}
      </div>

      {/* Task Class Distribution */}
      <div className="section">
        <h2>Task Classification</h2>
        <p className="section-desc">How tasks group by reliability profile — bimodal tasks are the reliability frontier.</p>
        <div className="class-bar">
          {Object.entries(tc).filter(([_, c]) => c > 0).map(([cls, count]) => (
            <div key={cls} className="class-segment" style={{
              width: `${(count / totalTasks) * 100}%`,
              background: CLASS_COLORS[cls] || '#6b7280',
            }} title={`${cls}: ${count} tasks`}>
              <span>{count}</span>
            </div>
          ))}
        </div>
        <div className="class-legend">
          {Object.entries(tc).filter(([_, c]) => c > 0).map(([cls, count]) => (
            <div key={cls} className="legend-item">
              <span className="legend-dot" style={{ background: CLASS_COLORS[cls] }} />
              <span className="legend-label">{cls.replace(/_/g, ' ')}</span>
              <span className="legend-count">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Efficiency Summary */}
      {summary.efficiency && (
        <div className="section">
          <h2>Efficiency</h2>
          <div className="stat-row">
            <div className="stat">
              <div className="stat-value">{summary.efficiency.total_redundant_calls}</div>
              <div className="stat-label">Redundant Calls</div>
            </div>
            <div className="stat">
              <div className="stat-value">{summary.efficiency.total_tool_errors}</div>
              <div className="stat-label">Tool Errors</div>
            </div>
            {summary.efficiency.avg_read_before_write != null && (
              <div className="stat">
                <div className="stat-value">{(summary.efficiency.avg_read_before_write * 100).toFixed(0)}%</div>
                <div className="stat-label">Read-Before-Write</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* CTA */}
      <div className="cta-section">
        <button className="cta-btn" onClick={onViewConversations}>
          View All {summary.total_conversations} Conversations →
        </button>
      </div>
    </div>
  )
}
