import { useState } from 'react'
import { ROLE_COLORS, TOOL_TYPE_COLORS, OUTCOME_COLORS, scoreColor } from '../utils/colors'

export default function ConversationAudit({ conversation, task, allTrials, fullData, onBack, onSwitchTrial }) {
  const c = conversation
  const messages = c.messages || []
  const hasMessages = messages.length > 0

  return (
    <div className="conversation-audit">
      {/* Header */}
      <div className="audit-header">
        <button className="back-btn" onClick={onBack}>← Conversations</button>
        <h2>Task {c.task_id} · Trial {c.trial}</h2>
        <span className={`outcome-badge large ${c.outcome}`}>
          {c.outcome === 'pass' ? '✓ PASS' : '✗ FAIL'}
        </span>
      </div>

      {/* Trial switcher (for bimodal tasks) */}
      {allTrials && allTrials.length > 1 && (
        <div className="trial-switcher">
          {allTrials.map(t => (
            <button
              key={t.id}
              className={`trial-pill ${t.id === c.id ? 'active' : ''} ${t.outcome}`}
              onClick={() => onSwitchTrial(t)}
            >
              T{t.trial} {t.outcome === 'pass' ? '✓' : '✗'}
            </button>
          ))}
        </div>
      )}

      <div className="audit-content">
        {/* Left: Message trace */}
        <div className="trace-panel">
          {!hasMessages ? (
            <div className="no-messages">
              Full message trace not available. Load reliability_full.json for conversation audit.
            </div>
          ) : (
            <div className="message-list">
              {messages.map((msg, i) => (
                <MessageCard key={i} msg={msg} index={i} actions={c.actions} />
              ))}
            </div>
          )}

          {/* Action timeline */}
          {c.actions && c.actions.length > 0 && (
            <div className="action-timeline-section">
              <h3>Action Timeline</h3>
              <div className="action-timeline">
                {c.actions.map((a, i) => (
                  <div key={i} className={`action-node ${a.type?.toLowerCase()}`}>
                    <div className="action-dot" style={{ background: TOOL_TYPE_COLORS[a.type] || '#6b7280' }}>
                      {a.correct === true ? '✓' : a.correct === false ? '✗' : '·'}
                    </div>
                    <div className="action-name">{a.name}</div>
                    <div className="action-type">{a.type}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Metrics panel */}
        <div className="metrics-panel">
          {/* Reward Breakdown */}
          <div className="metric-section">
            <h3>Reward Breakdown</h3>
            <RewardBreakdown breakdown={c.reward_breakdown} />
          </div>

          {/* Key Metrics */}
          <div className="metric-section">
            <h3>Conversation Metrics</h3>
            <div className="metric-grid">
              <Metric label="Cost" value={`$${c.cost_usd?.toFixed(3)}`} />
              <Metric label="Duration" value={`${c.duration_sec?.toFixed(1)}s`} />
              <Metric label="Actions" value={c.actions?.length || 0} />
              <Metric label="Turns" value={c.num_turns} />
              <Metric label="Termination" value={c.termination} />
            </div>
          </div>

          {/* Policy Adherence */}
          {c.policy_adherence?.available && (
            <div className="metric-section">
              <h3>Policy Compliance</h3>
              <div className="policy-score" style={{ color: scoreColor(c.policy_adherence.score) }}>
                {(c.policy_adherence.score * 100).toFixed(0)}%
              </div>
              {c.policy_adherence.phases_followed?.length > 0 && (
                <div className="policy-phases">
                  <span className="phase-label">Followed: </span>
                  {c.policy_adherence.phases_followed.map(p => (
                    <span key={p} className="phase-tag good">{p}</span>
                  ))}
                </div>
              )}
              {c.policy_adherence.violations?.length > 0 && (
                <div className="violations">
                  {c.policy_adherence.violations.map((v, i) => (
                    <div key={i} className="violation-item">
                      <span className="violation-type">{v.type}</span>
                      <span className="violation-desc">{v.description}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Efficiency */}
          <div className="metric-section">
            <h3>Efficiency</h3>
            <div className="metric-grid">
              <Metric label="Redundant Calls" value={c.efficiency?.redundant_calls || 0}
                warn={c.efficiency?.redundant_calls > 0} />
              <Metric label="Loops" value={c.efficiency?.loops || 0}
                warn={c.efficiency?.loops > 0} />
              <Metric label="Tool Errors" value={c.efficiency?.tool_errors || 0}
                warn={c.efficiency?.tool_errors > 0} />
              <Metric label="Error Bursts" value={c.efficiency?.error_bursts || 0}
                warn={c.efficiency?.error_bursts > 0} />
              {c.efficiency?.read_before_write_rate != null && (
                <Metric label="Read Before Write"
                  value={`${(c.efficiency.read_before_write_rate * 100).toFixed(0)}%`}
                  warn={c.efficiency.read_before_write_rate < 0.5} />
              )}
            </div>
          </div>

          {/* Abstention */}
          {c.abstention && c.abstention.abstained && (
            <div className="metric-section">
              <h3>Abstention Detected</h3>
              <div className="metric-grid">
                <Metric label="Type" value={c.abstention.type} warn={true} />
                <Metric label="Strength" value={`${(c.abstention.strength * 100).toFixed(0)}%`} warn={c.abstention.strength > 0.5} />
              </div>
              {c.abstention.evidence?.length > 0 && (
                <div className="abstention-evidence">
                  {c.abstention.evidence.map((e, i) => (
                    <div key={i} className="evidence-quote">"{e.text}"</div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Task Context */}
          {task && (
            <div className="metric-section">
              <h3>Task Context</h3>
              <div className="task-context">
                <div>Class: <span className={`class-badge ${task.class}`}>{task.class?.replace(/_/g, ' ')}</span></div>
                <div>Pass Rate: {task.pass_rate != null ? (task.pass_rate * 100).toFixed(0) + '%' : 'N/A'}</div>
                <div>Outcome Consistency: {task.consistency?.outcome != null ? (task.consistency.outcome * 100).toFixed(0) + '%' : 'N/A'}</div>
                {task.decisive_action && (
                  <div>Decisive Action: <code>{task.decisive_action}</code></div>
                )}
                {task.divergence?.turn != null && (
                  <div>Divergence: Trials branch at step {task.divergence.turn}</div>
                )}
              </div>
            </div>
          )}

          {/* What does this mean — plain English summary */}
          <div className="metric-section summary-section">
            <h3>Summary</h3>
            <div className="plain-summary">
              {c.outcome === 'pass' ? (
                <p className="summary-good">This conversation completed successfully. The agent resolved the customer's request correctly.</p>
              ) : (
                <p className="summary-bad">This conversation failed.
                  {c.reward_breakdown?.db && !c.reward_breakdown.db.matched && ' The database state was incorrect after the conversation — the agent made wrong changes.'}
                  {c.reward_breakdown?.action?.checks?.some(a => !a.matched) && ' Some tool calls did not match the expected actions.'}
                  {c.reward_breakdown?.communicate?.checks?.some(a => !a.met) && ' The agent failed to communicate required information to the customer.'}
                </p>
              )}
              {c.policy_adherence?.violations?.length > 0 && (
                <p className="summary-warn">Policy violations detected: {c.policy_adherence.violations.map(v => v.description).join('. ')}.</p>
              )}
              {c.efficiency?.redundant_calls > 0 && (
                <p className="summary-info">{c.efficiency.redundant_calls} redundant tool call(s) made — the same tool was called consecutively.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function MessageCard({ msg, index, actions }) {
  const [expanded, setExpanded] = useState(true)
  const role = msg.role || 'unknown'
  const bgColor = role === 'assistant' ? 'var(--bg-hover)' : role === 'user' ? 'rgba(16,185,129,0.06)' : 'var(--bg-card)'
  const borderColor = ROLE_COLORS[role] || '#d1d5db'

  // Check if this message has tool calls
  const toolCalls = msg.tool_calls || []
  const isToolResult = role === 'tool'

  return (
    <div className="message-card" style={{ background: bgColor, borderLeftColor: borderColor }}>
      <div className="msg-header" onClick={() => setExpanded(!expanded)}>
        <span className="msg-role" style={{ color: borderColor }}>{role}</span>
        <span className="msg-index">#{index}</span>
        {msg.cost != null && <span className="msg-cost">${msg.cost.toFixed(4)}</span>}
        {!expanded && <span className="msg-collapsed">...</span>}
      </div>
      {expanded && (
        <div className="msg-body">
          {msg.content && <div className="msg-content">{msg.content}</div>}
          {toolCalls.length > 0 && (
            <div className="tool-calls">
              {toolCalls.map((tc, i) => {
                const actionInfo = actions?.find(a => a.name === tc.name)
                const toolType = actionInfo?.type || 'UNKNOWN'
                const correct = actionInfo?.correct
                return (
                  <div key={i} className={`tool-call ${toolType.toLowerCase()}`}
                    style={{ borderLeftColor: TOOL_TYPE_COLORS[toolType] }}>
                    <span className="tc-name">{tc.name}</span>
                    <span className="tc-type" style={{ color: TOOL_TYPE_COLORS[toolType] }}>{toolType}</span>
                    {correct === true && <span className="tc-correct">✓</span>}
                    {correct === false && <span className="tc-incorrect">✗</span>}
                    <pre className="tc-args">{JSON.stringify(tc.arguments, null, 2)}</pre>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function RewardBreakdown({ breakdown }) {
  if (!breakdown || Object.keys(breakdown).length === 0) return <div className="no-data">No breakdown available</div>

  return (
    <div className="reward-breakdown">
      {Object.entries(breakdown).map(([key, val]) => {
        const score = val?.score ?? (val?.matched ? 1.0 : val?.matched === false ? 0.0 : null)
        const passed = score != null ? score >= 0.99 : val?.matched
        return (
          <div key={key} className={`reward-item ${passed ? 'pass' : 'fail'}`}>
            <span className="reward-icon">{passed ? '✓' : '✗'}</span>
            <span className="reward-key">{key.replace(/_/g, ' ')}</span>
            {score != null && <span className="reward-score">{(score * 100).toFixed(0)}%</span>}
          </div>
        )
      })}
    </div>
  )
}

function Metric({ label, value, warn = false }) {
  return (
    <div className={`metric-item ${warn ? 'warn' : ''}`}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  )
}
