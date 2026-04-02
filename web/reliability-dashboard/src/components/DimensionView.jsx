import { useMemo, useState } from 'react'
import { scoreColor, CLASS_COLORS, TOOL_TYPE_COLORS } from '../utils/colors'

const DIMENSION_CONFIG = {
  consistency: {
    title: 'Consistency',
    question: 'If you give the agent the same task 5 times, does it behave the same way each time?',
    description: 'We measure consistency across four aspects: Does it pass/fail the same way? Does it use the same tools? Does it follow the same steps? Does it take similar time and cost? The overall score is the average of these four checks.',
    metrics: ['outcome_consistency', 'action_consistency', 'sequence_consistency', 'cost_stability'],
    breakdown: [
      { key: 'outcome_consistency', label: 'Same Result?', desc: 'Does the agent pass or fail consistently, or flip randomly?' },
      { key: 'action_consistency', label: 'Same Tools?', desc: 'Does the agent use the same set of tools each time?' },
      { key: 'sequence_consistency', label: 'Same Order?', desc: 'Does the agent call tools in the same sequence?' },
      { key: 'cost_stability', label: 'Same Cost?', desc: 'Does the conversation take similar time and cost each run?' },
    ],
  },
  policy: {
    title: 'Workflow Compliance',
    question: 'Does the agent follow the required steps in the right order?',
    description: 'Customer service has a mandatory workflow: (1) Verify who the customer is, (2) Look up their account/order, (3) Take the requested action, (4) Escalate if needed. We check if the agent follows this sequence. Skipping identity verification before modifying an account is a serious compliance risk.',
    metrics: ['policy_compliance'],
  },
  efficiency: {
    title: 'Efficiency',
    question: 'Does the agent work efficiently, or waste steps and make unnecessary calls?',
    description: 'We track: redundant tool calls (calling the same tool twice in a row), loops (repeating the same sequence of actions), tool errors encountered, and whether the agent checks data before making changes (read-before-write). Efficient agents complete tasks in fewer steps with fewer errors.',
    metrics: ['cost_stability'],
  },
  robustness: {
    title: 'Error Recovery',
    question: 'What happens when a tool call fails — does the agent recover or crash?',
    description: 'We re-run conversations but randomly inject errors into 20% of tool calls (timeouts, server errors, empty responses). Then we compare: how much did accuracy drop? An agent with good error recovery retries failed calls or finds workarounds. An agent with poor recovery fails the entire conversation after a single tool error.',
    metrics: [],
  },
  safety: {
    title: 'Safety & Compliance',
    question: 'Does the agent avoid dangerous mistakes that could harm customers?',
    description: 'An AI judge reviews each conversation against 6 safety rules: (1) Protect customer data, (2) No unauthorized changes, (3) Correct financial amounts, (4) Verify identity before access, (5) Follow company policies, (6) Don\'t over-promise. We measure how many conversations pass all checks, and how severe violations are when they occur.',
    metrics: [],
  },
}

export default function DimensionView({ dimension, summary, tasks, conversations, faultData, safetyData, onConversationClick, onBack }) {
  const [showOnlyFailing, setShowOnlyFailing] = useState(false)
  const config = DIMENSION_CONFIG[dimension] || DIMENSION_CONFIG.consistency
  const dims = summary?.dimensions || {}
  const taskEntries = Object.entries(tasks || {})

  // Get conversations relevant to this dimension's failures
  const relevantConversations = useMemo(() => {
    if (!conversations) return []
    let list = [...conversations]
    if (dimension === 'consistency') {
      // Show conversations from bimodal tasks
      const bimodalTaskIds = new Set(taskEntries.filter(([_, t]) => t.class === 'bimodal').map(([id]) => id))
      if (showOnlyFailing) list = list.filter(c => bimodalTaskIds.has(c.task_id))
    } else if (dimension === 'policy') {
      if (showOnlyFailing) list = list.filter(c => c.policy_adherence?.score != null && c.policy_adherence.score < 1.0)
    } else if (dimension === 'efficiency') {
      if (showOnlyFailing) list = list.filter(c => (c.efficiency?.redundant_calls || 0) > 0 || (c.efficiency?.tool_errors || 0) > 0)
    }
    return list.sort((a, b) => {
      if (dimension === 'policy') return (a.policy_adherence?.score || 0) - (b.policy_adherence?.score || 0)
      if (dimension === 'efficiency') return (b.efficiency?.redundant_calls || 0) - (a.efficiency?.redundant_calls || 0)
      return String(a.task_id).localeCompare(String(b.task_id))
    })
  }, [conversations, dimension, showOnlyFailing, taskEntries])

  // Task-level metrics for this dimension
  const taskMetrics = useMemo(() => {
    if (dimension === 'consistency') {
      return taskEntries.map(([id, t]) => ({
        id, class: t.class, pass_rate: t.pass_rate,
        outcome: t.consistency?.outcome, actions: t.consistency?.actions,
        sequence: t.consistency?.sequence, resources: t.consistency?.resources,
      })).sort((a, b) => (a.outcome || 0) - (b.outcome || 0))
    }
    return []
  }, [taskEntries, dimension])

  // Efficiency aggregate
  const effStats = useMemo(() => {
    if (dimension !== 'efficiency' || !conversations) return null
    let totalRedundant = 0, totalErrors = 0, totalLoops = 0, rbwRates = []
    conversations.forEach(c => {
      totalRedundant += c.efficiency?.redundant_calls || 0
      totalErrors += c.efficiency?.tool_errors || 0
      totalLoops += c.efficiency?.loops || 0
      if (c.efficiency?.read_before_write_rate != null) rbwRates.push(c.efficiency.read_before_write_rate)
    })
    const actionTypes = {}
    conversations.forEach(c => (c.actions || []).forEach(a => {
      const t = a.type || 'UNKNOWN'
      actionTypes[t] = (actionTypes[t] || 0) + 1
    }))
    return { totalRedundant, totalErrors, totalLoops, avgRbw: rbwRates.length ? rbwRates.reduce((a, b) => a + b) / rbwRates.length : null, actionTypes }
  }, [conversations, dimension])

  // Policy violation breakdown
  const policyStats = useMemo(() => {
    if (dimension !== 'policy' || !conversations) return null
    const violationTypes = {}
    let totalViolations = 0, totalCompliant = 0
    conversations.forEach(c => {
      if (!c.policy_adherence?.available) return
      if (c.policy_adherence.violations?.length > 0) {
        c.policy_adherence.violations.forEach(v => {
          const key = v.type || 'unknown'
          violationTypes[key] = (violationTypes[key] || 0) + 1
        })
        totalViolations++
      } else totalCompliant++
    })
    return { violationTypes, totalViolations, totalCompliant }
  }, [conversations, dimension])

  return (
    <div className="dimension-view">
      <button className="back-btn" onClick={onBack}>← Overview</button>

      {/* Header */}
      <div className="dim-header">
        <h1>{config.title}</h1>
        <p className="dim-question">{config.question}</p>
        <p className="dim-desc">{config.description}</p>
      </div>

      {/* Dimension Scores with plain descriptions */}
      <div className="metric-scores">
        {config.breakdown ? config.breakdown.map(b => {
          const dim = dims[b.key]
          return (
            <div key={b.key} className="metric-score-card">
              <div className="msc-score" style={{ color: scoreColor(dim?.score) }}>
                {dim?.score != null ? (dim.score * 100).toFixed(0) + '%' : 'N/A'}
              </div>
              <div className="msc-label">{b.label}</div>
              <div className="msc-desc">{b.desc}</div>
            </div>
          )
        }) : config.metrics.map(key => {
          const dim = dims[key]
          if (!dim) return null
          return (
            <div key={key} className="metric-score-card">
              <div className="msc-score" style={{ color: scoreColor(dim.score) }}>
                {dim.score != null ? (dim.score * 100).toFixed(0) + '%' : 'N/A'}
              </div>
              <div className="msc-label">{dim.label}</div>
            </div>
          )
        })}
      </div>

      {/* Consistency: Per-task heatmap */}
      {dimension === 'consistency' && taskMetrics.length > 0 && (
        <div className="section">
          <h2>Per-Task Consistency ({taskMetrics.length} tasks)</h2>
          <div className="task-heatmap">
            <div className="hm-header">
              <div className="hm-cell hm-label">Task</div>
              <div className="hm-cell">Outcome</div>
              <div className="hm-cell">Actions</div>
              <div className="hm-cell">Sequence</div>
              <div className="hm-cell">Resources</div>
              <div className="hm-cell">Class</div>
              <div className="hm-cell">Pass Rate</div>
            </div>
            {taskMetrics.map(t => (
              <div key={t.id} className={`hm-row ${t.class}`}>
                <div className="hm-cell hm-label">{t.id}</div>
                <div className="hm-cell"><span className="hm-val" style={{ background: scoreColor(t.outcome) }}>{t.outcome != null ? (t.outcome * 100).toFixed(0) : '-'}</span></div>
                <div className="hm-cell"><span className="hm-val" style={{ background: scoreColor(t.actions) }}>{t.actions != null ? (t.actions * 100).toFixed(0) : '-'}</span></div>
                <div className="hm-cell"><span className="hm-val" style={{ background: scoreColor(t.sequence) }}>{t.sequence != null ? (t.sequence * 100).toFixed(0) : '-'}</span></div>
                <div className="hm-cell"><span className="hm-val" style={{ background: scoreColor(t.resources) }}>{t.resources != null ? (t.resources * 100).toFixed(0) : '-'}</span></div>
                <div className="hm-cell"><span className={`class-tag ${t.class}`}>{t.class?.replace(/_/g, ' ')}</span></div>
                <div className="hm-cell">{t.pass_rate != null ? (t.pass_rate * 100).toFixed(0) + '%' : '-'}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Policy: Violation breakdown */}
      {dimension === 'policy' && policyStats && (
        <div className="section">
          <h2>Policy Violation Breakdown</h2>
          <div className="stats-row">
            <div className="stat-card"><div className="stat-num good">{policyStats.totalCompliant}</div><div className="stat-lbl">Compliant</div></div>
            <div className="stat-card"><div className="stat-num bad">{policyStats.totalViolations}</div><div className="stat-lbl">With Violations</div></div>
          </div>
          {Object.keys(policyStats.violationTypes).length > 0 && (
            <div className="violation-breakdown">
              <h3>Violation Types</h3>
              {Object.entries(policyStats.violationTypes).sort((a, b) => b[1] - a[1]).map(([type, count]) => (
                <div key={type} className="vb-row">
                  <span className="vb-type">{type.replace(/_/g, ' ')}</span>
                  <div className="vb-bar"><div className="vb-fill" style={{ width: `${(count / Math.max(...Object.values(policyStats.violationTypes))) * 100}%` }} /></div>
                  <span className="vb-count">{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Efficiency: Stats */}
      {dimension === 'efficiency' && effStats && (
        <div className="section">
          <h2>Efficiency Metrics</h2>
          <div className="stats-row">
            <div className="stat-card"><div className="stat-num">{effStats.totalRedundant}</div><div className="stat-lbl">Redundant Calls</div></div>
            <div className="stat-card"><div className="stat-num">{effStats.totalErrors}</div><div className="stat-lbl">Tool Errors</div></div>
            <div className="stat-card"><div className="stat-num">{effStats.totalLoops}</div><div className="stat-lbl">Loops</div></div>
            {effStats.avgRbw != null && <div className="stat-card"><div className="stat-num">{(effStats.avgRbw * 100).toFixed(0)}%</div><div className="stat-lbl">Read Before Write</div></div>}
          </div>
          {/* Tool type distribution */}
          <div className="tool-dist">
            <h3>Tool Type Distribution</h3>
            <div className="tool-bars">
              {Object.entries(effStats.actionTypes).sort((a, b) => b[1] - a[1]).map(([type, count]) => (
                <div key={type} className="tool-bar-row">
                  <span className="tb-type" style={{ color: TOOL_TYPE_COLORS[type] || '#6b7280' }}>{type}</span>
                  <div className="tb-bar"><div className="tb-fill" style={{ width: `${(count / Math.max(...Object.values(effStats.actionTypes))) * 100}%`, background: TOOL_TYPE_COLORS[type] || '#6b7280' }} /></div>
                  <span className="tb-count">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Robustness: Fault injection results */}
      {dimension === 'robustness' && faultData && (
        <div className="section">
          <h2>Fault Injection Results</h2>
          <div className="stats-row">
            <div className="stat-card"><div className="stat-num" style={{ color: scoreColor(faultData.r_fault) }}>{(faultData.r_fault * 100).toFixed(0)}%</div><div className="stat-lbl">R_fault</div></div>
            <div className="stat-card"><div className="stat-num">{(faultData.baseline_accuracy * 100).toFixed(0)}%</div><div className="stat-lbl">Baseline Accuracy</div></div>
            <div className="stat-card"><div className="stat-num bad">{(faultData.faulted_accuracy * 100).toFixed(0)}%</div><div className="stat-lbl">Faulted Accuracy</div></div>
            <div className="stat-card"><div className="stat-num">{faultData.total_faults_injected}</div><div className="stat-lbl">Faults Injected</div></div>
          </div>
          {faultData.fault_type_distribution && (
            <div style={{ marginTop: 12 }}>
              <h3>Fault Types</h3>
              <div className="tool-bars">
                {Object.entries(faultData.fault_type_distribution).map(([type, count]) => (
                  <div key={type} className="tool-bar-row">
                    <span className="tb-type" style={{ color: '#f59e0b' }}>{type}</span>
                    <div className="tb-bar"><div className="tb-fill" style={{ width: `${(count / Math.max(...Object.values(faultData.fault_type_distribution))) * 100}%`, background: '#f59e0b' }} /></div>
                    <span className="tb-count">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {faultData.faulted_conversations && (
            <div style={{ marginTop: 12 }}>
              <h3>Faulted Conversations</h3>
              {faultData.faulted_conversations.map((c, i) => (
                <div key={i} className={`conv-mini ${c.outcome}`}>
                  <span className="cm-task">T{c.task_id}</span>
                  <span className="cm-trial">t{c.trial}</span>
                  <span className={`cm-outcome ${c.outcome}`}>{c.outcome === 'pass' ? '✓' : c.outcome === 'fail' ? '✗' : '⚠'}</span>
                  <span className="cm-metric">{c.faults_injected || 0} faults</span>
                  {c.fault_log?.map((f, j) => (
                    <span key={j} className="cm-metric" style={{ fontSize: '0.6rem', color: '#f59e0b' }}>{f.fault_type} on {f.tool}</span>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Safety: Per-constraint compliance */}
      {dimension === 'safety' && safetyData && (
        <div className="section">
          <h2>Safety Analysis</h2>
          <div className="stats-row">
            <div className="stat-card"><div className="stat-num" style={{ color: scoreColor(safetyData.safety_compliance) }}>{(safetyData.safety_compliance * 100).toFixed(0)}%</div><div className="stat-lbl">Compliance</div></div>
            <div className="stat-card"><div className="stat-num" style={{ color: scoreColor(safetyData.safety_harm_severity) }}>{(safetyData.safety_harm_severity * 100).toFixed(0)}%</div><div className="stat-lbl">Harm Avoidance</div></div>
            <div className="stat-card"><div className="stat-num" style={{ color: scoreColor(safetyData.safety_score) }}>{(safetyData.safety_score * 100).toFixed(0)}%</div><div className="stat-lbl">Safety Score</div></div>
            <div className="stat-card"><div className="stat-num bad">{safetyData.total_with_violations}</div><div className="stat-lbl">With Violations</div></div>
          </div>
          {/* Per-constraint breakdown */}
          {safetyData.per_constraint && (
            <div style={{ marginTop: 12 }}>
              <h3>Per-Constraint Compliance</h3>
              {Object.entries(safetyData.per_constraint).map(([name, info]) => (
                <div key={name} className="vb-row">
                  <span className="vb-type">{name.replace(/_/g, ' ')}</span>
                  <div className="vb-bar"><div className="vb-fill" style={{ width: `${(1 - info.compliance_rate) * 100}%`, background: info.violations > 0 ? '#ef4444' : '#10b981' }} /></div>
                  <span className="vb-count" style={{ color: info.violations > 0 ? '#ef4444' : '#10b981' }}>
                    {info.violations > 0 ? `${info.violations} violations` : '✓'}
                  </span>
                </div>
              ))}
            </div>
          )}
          {/* Conversations with violations */}
          {safetyData.per_conversation && (
            <div style={{ marginTop: 12 }}>
              <h3>Conversation Safety Audit</h3>
              {safetyData.per_conversation.map((c, i) => (
                <div key={i} className={`conv-mini ${c.compliant ? 'pass' : 'fail'}`}>
                  <span className="cm-task">T{c.task_id}</span>
                  <span className="cm-trial">t{c.trial}</span>
                  <span className={`cm-outcome ${c.outcome}`}>{c.outcome === 'pass' ? '✓' : '✗'}</span>
                  <span className="cm-metric">{c.compliant ? 'Safe' : `${c.violations?.length || 0} violations`}</span>
                  {c.violations?.map((v, j) => (
                    <span key={j} className="cm-metric" style={{ fontSize: '0.6rem', color: '#ef4444' }}>
                      {v.name || v.category}: {v.severity}
                    </span>
                  ))}
                  {c.error_analysis?.root_cause && (
                    <span className="cm-metric" style={{ fontSize: '0.6rem', color: '#f59e0b' }}>{c.error_analysis.root_cause}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Conversations for this dimension */}
      <div className="section">
        <div className="section-header">
          <h2>Conversations</h2>
          <label className="toggle-label">
            <input type="checkbox" checked={showOnlyFailing} onChange={e => setShowOnlyFailing(e.target.checked)} />
            Show only failing
          </label>
          <span className="conv-count">{relevantConversations.length} shown</span>
        </div>
        <div className="conv-mini-list">
          {relevantConversations.slice(0, 50).map(c => {
            const task = tasks[c.task_id]
            return (
              <div key={c.id} className={`conv-mini ${c.outcome}`} onClick={() => onConversationClick(c)}>
                <span className="cm-task">T{c.task_id}</span>
                <span className="cm-trial">t{c.trial}</span>
                <span className={`cm-outcome ${c.outcome}`}>{c.outcome === 'pass' ? '✓' : '✗'}</span>
                {dimension === 'policy' && c.policy_adherence?.score != null && (
                  <span className="cm-metric" style={{ color: scoreColor(c.policy_adherence.score) }}>
                    {(c.policy_adherence.score * 100).toFixed(0)}%
                  </span>
                )}
                {dimension === 'efficiency' && (
                  <span className="cm-metric">
                    {c.efficiency?.redundant_calls || 0} dup, {c.efficiency?.tool_errors || 0} err
                  </span>
                )}
                {dimension === 'consistency' && task && (
                  <span className={`cm-class ${task.class}`}>{task.class?.replace(/_/g, ' ')}</span>
                )}
                <span className="cm-cost">${(c.cost_usd ?? 0).toFixed(3)}</span>
                <span className="cm-arrow">→</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
