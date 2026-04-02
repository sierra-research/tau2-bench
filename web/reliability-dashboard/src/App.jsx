import { useState, useEffect, useMemo } from 'react'
import Overview from './components/Overview'
import DimensionView from './components/DimensionView'
import ConversationList from './components/ConversationList'
import ConversationAudit from './components/ConversationAudit'

export default function App() {
  const [data, setData] = useState(null)
  const [fullData, setFullData] = useState(null)
  const [selectedRun, setSelectedRun] = useState(0)
  const [view, setView] = useState('overview')
  const [selectedDimension, setSelectedDimension] = useState(null)
  const [selectedConv, setSelectedConv] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // URL routing: restore state from hash on load
  useEffect(() => {
    const hash = window.location.hash.slice(1)
    if (!hash) return
    const params = new URLSearchParams(hash)
    if (params.get('view')) setView(params.get('view'))
    if (params.get('dim')) setSelectedDimension(params.get('dim'))
    if (params.get('run')) setSelectedRun(Number(params.get('run')) || 0)
  }, [])

  // URL routing: sync state to hash
  useEffect(() => {
    const params = new URLSearchParams()
    if (view !== 'overview') params.set('view', view)
    if (selectedDimension && view === 'dimension') params.set('dim', selectedDimension)
    if (selectedRun > 0) params.set('run', String(selectedRun))
    if (selectedConv) params.set('conv', selectedConv.id || '')
    const hash = params.toString()
    window.history.replaceState(null, '', hash ? `#${hash}` : '#')
  }, [view, selectedDimension, selectedRun, selectedConv])

  const [faultData, setFaultData] = useState(null)
  const [safetyData, setSafetyData] = useState(null)

  useEffect(() => {
    Promise.all([
      fetch('/data/reliability_data.json').then(r => r.ok ? r.json() : Promise.reject('No data')),
      fetch('/data/reliability_full.json').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('/data/fault_results.json').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('/data/safety_results.json').then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([slim, full, fault, safety]) => {
      if (slim.runs) setData(slim)
      else setData({ runs: [{ id: 'default', domain_summary: slim.domain_summary, tasks: slim.tasks, conversations: slim.conversations }], num_runs: 1 })
      setFullData(full)
      setFaultData(fault)
      setSafetyData(safety)
      setLoading(false)
    }).catch(e => { setError(String(e)); setLoading(false) })
  }, [])

  const runs = data?.runs || []
  const currentRun = runs[selectedRun] || runs[0]
  const summary = currentRun?.domain_summary
  const conversations = currentRun?.conversations || []
  const tasks = currentRun?.tasks || {}

  function openConversation(conv) {
    if (fullData?.all_conversations) {
      const full = fullData.all_conversations.find(c => c.id === conv.id)
      setSelectedConv(full || conv)
    } else if (fullData?.conversations) {
      const full = fullData.conversations.find(c => c.id === conv.id)
      setSelectedConv(full || conv)
    } else setSelectedConv(conv)
    setView('audit')
  }

  function navigate(v, dim) {
    setView(v)
    if (dim) setSelectedDimension(dim)
    if (v !== 'audit') setSelectedConv(null)
  }

  if (loading) return <div className="loading-screen"><div className="spinner" /><p>Loading reliability data...</p></div>
  if (error) return (
    <div className="error-screen">
      <h1>No Data Found</h1>
      <pre>{`tau2 run --domain retail --agent-llm azure/gpt-5.4 --num-trials 5 --save-to my_eval
PYTHONPATH=src/experiments/tau2_reliability tau2 reliability analyze \\
  --results data/simulations/my_eval/results.json \\
  --output web/reliability-dashboard/public/data/`}</pre>
    </div>
  )

  return (
    <div className="app dark">
      <nav className="topnav">
        <div className="nav-brand" onClick={() => navigate('overview')}>
          <span className="tau">τ</span><span className="brand-text">reliability</span>
        </div>
        <div className="nav-tabs">
          <button className={view === 'overview' ? 'active' : ''} onClick={() => navigate('overview')}>Overview</button>
          {['consistency', 'policy', 'efficiency', 'robustness', 'safety'].map(d => (
            <button key={d} className={view === 'dimension' && selectedDimension === d ? 'active' : ''}
              onClick={() => navigate('dimension', d)}>{d.charAt(0).toUpperCase() + d.slice(1)}</button>
          ))}
          <button className={view === 'conversations' ? 'active' : ''} onClick={() => navigate('conversations')}>Conversations</button>
        </div>
        {runs.length > 1 && (
          <select className="run-select" value={selectedRun} onChange={e => { setSelectedRun(Number(e.target.value)); navigate('overview') }}>
            {runs.map((r, i) => <option key={i} value={i}>{r.domain_summary?.model} · {r.domain_summary?.domain}</option>)}
          </select>
        )}
      </nav>

      <main>
        {view === 'overview' && <Overview summary={summary} tasks={tasks} conversations={conversations}
          faultData={faultData} safetyData={safetyData}
          onDimensionClick={d => navigate('dimension', d)} onViewConversations={() => navigate('conversations')} />}
        {view === 'dimension' && <DimensionView dimension={selectedDimension} summary={summary} tasks={tasks}
          conversations={conversations} faultData={faultData} safetyData={safetyData}
          onConversationClick={openConversation} onBack={() => navigate('overview')} />}
        {view === 'conversations' && <ConversationList conversations={conversations} tasks={tasks}
          onSelectConversation={openConversation} onBack={() => navigate('overview')} />}
        {view === 'audit' && selectedConv && <ConversationAudit conversation={selectedConv}
          task={tasks[selectedConv.task_id]} allTrials={conversations.filter(c => c.task_id === selectedConv.task_id)}
          fullData={fullData} onBack={() => navigate('conversations')} onSwitchTrial={openConversation} />}
      </main>
    </div>
  )
}
