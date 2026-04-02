import { useState, useMemo } from 'react'
import { OUTCOME_COLORS, CLASS_COLORS, scoreColor } from '../utils/colors'

const PAGE_SIZE = 50

export default function ConversationList({ conversations, tasks, onSelectConversation, onBack }) {
  const [page, setPage] = useState(0)
  const [filterOutcome, setFilterOutcome] = useState('all')
  const [filterClass, setFilterClass] = useState('all')
  const [filterComponent, setFilterComponent] = useState('all')
  const [sortBy, setSortBy] = useState('task_id')
  const [sortDir, setSortDir] = useState('asc')

  const filtered = useMemo(() => {
    let list = [...conversations]

    if (filterOutcome !== 'all') list = list.filter(c => c.outcome === filterOutcome)
    if (filterClass !== 'all') {
      list = list.filter(c => {
        const task = tasks[c.task_id]
        return task?.class === filterClass
      })
    }
    if (filterComponent !== 'all') {
      list = list.filter(c => {
        const bd = c.reward_breakdown || {}
        if (filterComponent === 'db') return bd.db && !bd.db.matched
        if (filterComponent === 'action') return bd.action?.score != null && bd.action.score < 1
        if (filterComponent === 'communicate') return bd.communicate?.score != null && bd.communicate.score < 1
        if (filterComponent === 'policy') return c.policy_adherence?.score != null && c.policy_adherence.score < 0.7
        return true
      })
    }

    list.sort((a, b) => {
      let aVal, bVal
      if (sortBy === 'task_id') { aVal = a.task_id; bVal = b.task_id }
      else if (sortBy === 'cost') { aVal = a.cost_usd; bVal = b.cost_usd }
      else if (sortBy === 'duration') { aVal = a.duration_sec; bVal = b.duration_sec }
      else if (sortBy === 'policy') { aVal = a.policy_adherence?.score ?? 0; bVal = b.policy_adherence?.score ?? 0 }
      else if (sortBy === 'outcome') { aVal = a.outcome; bVal = b.outcome }
      else { aVal = a[sortBy]; bVal = b[sortBy] }

      if (aVal < bVal) return sortDir === 'asc' ? -1 : 1
      if (aVal > bVal) return sortDir === 'asc' ? 1 : -1
      return 0
    })

    return list
  }, [conversations, tasks, filterOutcome, filterClass, filterComponent, sortBy, sortDir])

  function toggleSort(col) {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortBy(col); setSortDir('asc') }
  }

  const sortIcon = (col) => sortBy === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''

  function failedComponent(c) {
    const bd = c.reward_breakdown || {}
    if (c.outcome === 'pass') return '—'
    const failed = []
    if (bd.db && !bd.db.matched) failed.push('Database')
    if (bd.action?.score != null && bd.action.score < 1) failed.push('Actions')
    if (bd.communicate?.score != null && bd.communicate.score < 1) failed.push('Communication')
    return failed.join(', ') || 'Unknown'
  }

  return (
    <div className="conversation-list">
      <div className="list-header">
        <button className="back-btn" onClick={onBack}>← Overview</button>
        <h2>All Conversations</h2>
        <span className="count">{filtered.length} of {conversations.length}</span>
      </div>

      {/* Filters */}
      <div className="filters">
        <div className="filter-group">
          <label>Outcome</label>
          <select value={filterOutcome} onChange={e => setFilterOutcome(e.target.value)}>
            <option value="all">All</option>
            <option value="pass">Pass</option>
            <option value="fail">Fail</option>
          </select>
        </div>
        <div className="filter-group">
          <label>Task Class</label>
          <select value={filterClass} onChange={e => setFilterClass(e.target.value)}>
            <option value="all">All</option>
            <option value="bimodal">Bimodal</option>
            <option value="stable_pass">Stable Pass</option>
            <option value="stable_fail">Stable Fail</option>
            <option value="fragile">Fragile</option>
          </select>
        </div>
        <div className="filter-group">
          <label>Failed Component</label>
          <select value={filterComponent} onChange={e => setFilterComponent(e.target.value)}>
            <option value="all">All</option>
            <option value="db">Database</option>
            <option value="action">Actions</option>
            <option value="communicate">Communication</option>
            <option value="policy">Policy Violations</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <table className="conv-table">
        <thead>
          <tr>
            <th className="sortable" onClick={() => toggleSort('task_id')}>Task{sortIcon('task_id')}</th>
            <th>Trial</th>
            <th className="sortable" onClick={() => toggleSort('outcome')}>Outcome{sortIcon('outcome')}</th>
            <th>Failed Component</th>
            <th className="sortable" onClick={() => toggleSort('cost')}>Cost{sortIcon('cost')}</th>
            <th className="sortable" onClick={() => toggleSort('duration')}>Duration{sortIcon('duration')}</th>
            <th>Actions</th>
            <th className="sortable" onClick={() => toggleSort('policy')}>Policy{sortIcon('policy')}</th>
            <th>Class</th>
          </tr>
        </thead>
        <tbody>
          {filtered.length === 0 ? (
            <tr><td colSpan={9} style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)' }}>No conversations match filters</td></tr>
          ) : filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(c => {
            const task = tasks[c.task_id]
            const cls = task?.class || 'moderate'
            return (
              <tr key={c.id} className={`conv-row ${c.outcome}`} onClick={() => onSelectConversation(c)}>
                <td className="task-id">{c.task_id}</td>
                <td>{c.trial}</td>
                <td>
                  <span className={`outcome-badge ${c.outcome}`}>
                    {c.outcome === 'pass' ? '✓ Pass' : '✗ Fail'}
                  </span>
                </td>
                <td className="failed-comp">{failedComponent(c)}</td>
                <td>${(c.cost_usd ?? 0).toFixed(3)}</td>
                <td>{(c.duration_sec ?? 0).toFixed(1)}s</td>
                <td>{c.actions?.length || 0}</td>
                <td>
                  {c.policy_adherence?.score != null ? (
                    <span style={{ color: scoreColor(c.policy_adherence.score) }}>
                      {(c.policy_adherence.score * 100).toFixed(0)}%
                    </span>
                  ) : '—'}
                </td>
                <td>
                  <span className={`class-badge ${cls}`}>{cls.replace(/_/g, ' ')}</span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {/* Pagination */}
      {filtered.length > PAGE_SIZE && (
        <div className="pagination">
          <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
          <span>Page {page + 1} of {Math.ceil(filtered.length / PAGE_SIZE)} ({filtered.length} total)</span>
          <button disabled={(page + 1) * PAGE_SIZE >= filtered.length} onClick={() => setPage(p => p + 1)}>Next →</button>
        </div>
      )}
    </div>
  )
}
