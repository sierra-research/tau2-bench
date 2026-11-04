import { useState, useEffect, useMemo, useRef } from 'react'
import VoiceViewer from './VoiceViewer'
import ModelPicker, { EntryChips } from './ModelPicker'
import JudgeFindings, { CallVerdicts } from './JudgeFindings'
import {
  BENCHMARKS, DOMAINS, LANGUAGES, S3_SUBMISSIONS, SUBMISSIONS_BASE,
  loadCell, loadCellQuality, loadSimulation, loadVisualizerIndex,
} from '../data/visualizerIndex'
import './TrajectoryVisualizer.css'

const TrajectoryVisualizer = () => {
  // --- Top-level selector state ---
  const [viewMode, setViewMode] = useState('trajectories') // 'trajectories' or 'tasks'
  const [entries, setEntries] = useState(null) // visualizer index; null while loading
  const [indexError, setIndexError] = useState(null)

  // Trajectory mode selectors: benchmark → model → language → domain
  const [benchmark, setBenchmark] = useState(BENCHMARKS[0].id)
  const [selectedId, setSelectedId] = useState('')
  const [language, setLanguage] = useState('en')
  const [selectedDomain, setSelectedDomain] = useState('')
  const [selectedTrialIdx, setSelectedTrialIdx] = useState(0)
  const [selectedTaskId, setSelectedTaskId] = useState(null)

  // Loaded trajectory data for the current cell
  const [trajectoryData, setTrajectoryData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  // Full text simulations fetched on open, by id: { loading } | { data } | { error }
  const [simDetails, setSimDetails] = useState({})
  // Judged cells: per-call verdicts by simulation id, and the task-list filter
  const [cellQuality, setCellQuality] = useState(null)
  const [verdictFilter, setVerdictFilter] = useState('all')

  // Task mode state
  const [taskData, setTaskData] = useState(null)
  const [selectedDomainTask, setSelectedDomainTask] = useState(null)
  const [selectedTaskDetail, setSelectedTaskDetail] = useState(null)

  // Configuration modal
  const [showConfigModal, setShowConfigModal] = useState(false)
  const [modalClosing, setModalClosing] = useState(false)

  // Download popover
  const [showDownload, setShowDownload] = useState(false)
  const [cliCopied, setCliCopied] = useState(false)

  // Document viewer state
  const [showDocModal, setShowDocModal] = useState(false)
  const [docModalClosing, setDocModalClosing] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [docLoading, setDocLoading] = useState(false)

  const handleCloseModal = () => {
    setModalClosing(true)
    setTimeout(() => {
      setShowConfigModal(false)
      setModalClosing(false)
    }, 200) // Match the CSS slideDown duration (0.2s)
  }

  const handleCloseDocModal = () => {
    setDocModalClosing(true)
    setTimeout(() => {
      setShowDocModal(false)
      setDocModalClosing(false)
      setSelectedDoc(null)
    }, 200)
  }

  const handleDocClick = async (docId) => {
    try {
      setDocLoading(true)
      setShowDocModal(true)
      setSelectedDoc(null)
      const domain = selectedDomainTask || 'banking_knowledge'
      const res = await fetch(`${import.meta.env.BASE_URL}task-data/domains/${domain}/documents/${docId}.json`)
      if (!res.ok) throw new Error(`Failed to load document: ${res.statusText}`)
      const doc = await res.json()
      setSelectedDoc(doc)
    } catch (err) {
      setSelectedDoc({ id: docId, title: 'Error', content: `Failed to load document: ${err.message}` })
    } finally {
      setDocLoading(false)
    }
  }

  // Task-definition domains (Tasks view)
  const domains = [
    { id: 'airline', label: 'Airline', icon: '✈️', color: '#3b82f6' },
    { id: 'retail', label: 'Retail', icon: '🛍️', color: '#8b5cf6' },
    { id: 'telecom', label: 'Telecom', icon: '📱', color: '#059669' },
    { id: 'banking_knowledge', label: 'Banking', icon: '🏦', color: '#d97706' }
  ]

  // --- Parse URL params for deep linking ---
  const getUrlParams = () => {
    const params = new URLSearchParams(window.location.search)
    const trialRaw = params.get('trial')
    return {
      bench: params.get('bench'),
      model: params.get('model'),
      lang: params.get('lang'),
      domain: params.get('domain'),
      task: params.get('task'),
      trial: trialRaw != null ? Number(trialRaw) : null,
      view: params.get('view'),
    }
  }

  // Ref to hold URL params that need deferred restoration (task/trial depend on data loading)
  const pendingUrlParams = useRef(null)
  // Suppress URL updates while we're restoring from URL
  const restoringFromUrl = useRef(false)

  // --- Derived selection ---
  const currentEntry = useMemo(
    () => entries?.find(e => e.id === selectedId) || null,
    [entries, selectedId]
  )
  const benchmarkEntries = useMemo(
    () => (entries || []).filter(e => e.benchmark === benchmark),
    [entries, benchmark]
  )
  const entryLanguages = useMemo(() => {
    if (!currentEntry) return []
    const ids = new Set(currentEntry.cells.map(c => c.language))
    return LANGUAGES.filter(l => ids.has(l.id))
  }, [currentEntry])
  const languageCells = useMemo(
    () => (currentEntry?.cells || []).filter(c => c.language === language),
    [currentEntry, language]
  )
  const currentCell = languageCells.find(c => c.domain === selectedDomain) || null
  const isVoice = BENCHMARKS.find(b => b.id === (currentEntry?.benchmark || benchmark))?.modality === 'voice'

  // Pick an entry, keeping the current language and domain when it has them.
  const selectEntry = (entry, prefer = {}) => {
    const langs = [...new Set(entry.cells.map(c => c.language))]
    const lang = [prefer.lang, language].find(l => l && langs.includes(l)) || langs[0]
    const cellDomains = entry.cells.filter(c => c.language === lang).map(c => c.domain)
    const domain = [prefer.domain, selectedDomain].find(d => d && cellDomains.includes(d)) || cellDomains[0]
    setBenchmark(entry.benchmark)
    setSelectedId(entry.id)
    setLanguage(lang)
    setSelectedDomain(domain || '')
    setSelectedTaskId(null)
    setSelectedTrialIdx(0)
    setShowDownload(false)
  }

  // --- Sync state → URL query (replaceState to avoid history clutter) ---
  useEffect(() => {
    if (restoringFromUrl.current) return
    if (!entries) return

    const params = new URLSearchParams()
    if (viewMode === 'trajectories') params.set('bench', benchmark)
    if (selectedId) params.set('model', selectedId)
    if (selectedId && entryLanguages.length > 1) params.set('lang', language)
    if (selectedDomain && selectedId) params.set('domain', selectedDomain)
    if (viewMode === 'tasks') params.set('view', 'tasks')
    if (selectedTaskId != null) params.set('task', String(selectedTaskId))
    if (selectedTrialIdx > 0) params.set('trial', String(selectedTrialIdx))

    const qs = params.toString()
    const newUrl = qs ? `/trajectory-visualizer?${qs}` : '/trajectory-visualizer'
    if (`${window.location.pathname}${window.location.search}` !== newUrl) {
      window.history.replaceState(null, '', newUrl)
    }
  }, [benchmark, selectedId, language, entryLanguages, selectedDomain, selectedTaskId, selectedTrialIdx, viewMode, entries])

  // --- Load the submission index on mount; restore a deep link ---
  useEffect(() => {
    let cancelled = false
    loadVisualizerIndex().then(loaded => {
      if (cancelled) return
      setEntries(loaded)
      const urlParams = getUrlParams()
      if (urlParams.view === 'tasks') setViewMode('tasks')
      const match = urlParams.model && loaded.find(e => e.id === urlParams.model)
      if (match) {
        restoringFromUrl.current = true
        selectEntry(match, { lang: urlParams.lang, domain: urlParams.domain })
        if (urlParams.task != null || urlParams.trial != null) {
          pendingUrlParams.current = { task: urlParams.task, trial: urlParams.trial }
        } else {
          restoringFromUrl.current = false
        }
      } else if (BENCHMARKS.some(b => b.id === urlParams.bench)) {
        setBenchmark(urlParams.bench)
      }
    }).catch(err => !cancelled && setIndexError(err.message))
    return () => { cancelled = true }
    // selectEntry reads state only to keep it; on mount there is none.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // --- Load a judged cell's per-call verdicts (alongside its task list) ---
  useEffect(() => {
    setCellQuality(null)
    setVerdictFilter('all')
    if (viewMode !== 'trajectories' || !currentCell?.quality) return
    let cancelled = false
    // Verdicts are optional: on failure the task list shows without them.
    loadCellQuality(currentCell).then(calls => !cancelled && setCellQuality(calls)).catch(() => {})
    return () => { cancelled = true }
  }, [currentCell, viewMode])

  // --- Load the selected cell's task list ---
  useEffect(() => {
    if (viewMode !== 'trajectories' || !currentCell) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setTrajectoryData(null)

    loadCell(currentCell).then(({ data }) => {
      if (cancelled) return
      setTrajectoryData(data)

      // Apply pending URL params (task/trial) now that data is loaded.
      if (pendingUrlParams.current) {
        const { task, trial } = pendingUrlParams.current
        pendingUrlParams.current = null
        if (task != null) {
          // Use the actual ID from the data to preserve its native type
          const actualTask = data.tasks?.find(t => String(t.id) === String(task))
          const actualSim = !actualTask && data.simulations?.find(s => String(s.task_id) === String(task))
          if (actualTask || actualSim) {
            setSelectedTaskId(actualTask ? actualTask.id : actualSim.task_id)
            if (trial != null && trial > 0) setSelectedTrialIdx(trial)
          }
        }
        restoringFromUrl.current = false
      }
    }).catch(err => {
      if (cancelled) return
      setError(err.message)
      if (pendingUrlParams.current) {
        pendingUrlParams.current = null
        restoringFromUrl.current = false
      }
    }).finally(() => !cancelled && setLoading(false))
    // A cancelled load must not leave the spinner up (e.g. in the Tasks view).
    return () => { cancelled = true; setLoading(false) }
  }, [currentCell, viewMode])

  // --- Derived data ---
  // All unique tasks from trajectory data
  const tasks = useMemo(() => {
    if (!trajectoryData?.tasks) return []
    return trajectoryData.tasks
  }, [trajectoryData])

  // Simulations grouped by task, each sorted by trial
  const simulationsByTask = useMemo(() => {
    const groups = new Map()
    for (const sim of trajectoryData?.simulations || []) {
      if (!groups.has(sim.task_id)) groups.set(sim.task_id, [])
      groups.get(sim.task_id).push(sim)
    }
    for (const sims of groups.values()) sims.sort((a, b) => a.trial - b.trial)
    return groups
  }, [trajectoryData])

  // Judged cells score one call per task (the earliest trial with a quality
  // record); its card chips and the task-list filter both read that call.
  const taskCalls = useMemo(() => {
    const calls = new Map()
    if (!cellQuality) return calls
    for (const [taskId, sims] of simulationsByTask) {
      const call = sims.map(sim => cellQuality.get(sim.id)).find(Boolean)
      if (call) calls.set(taskId, call)
    }
    return calls
  }, [cellQuality, simulationsByTask])
  const failedTasks = useMemo(() => {
    const failed = { interaction: new Set(), experience: new Set() }
    for (const [taskId, call] of taskCalls) {
      if (call.interaction === 'fail') failed.interaction.add(taskId)
      if (call.experience === 'fail') failed.experience.add(taskId)
    }
    return failed
  }, [taskCalls])
  const verdictCounts = { interaction: failedTasks.interaction.size, experience: failedTasks.experience.size }
  const shownTasks = verdictFilter === 'all' ? tasks : tasks.filter(task => failedTasks[verdictFilter].has(task.id))

  // Number of distinct trials available
  const numTrials = useMemo(() => {
    if (!trajectoryData?.simulations) return 0
    const trials = new Set(trajectoryData.simulations.map(s => s.trial))
    return trials.size
  }, [trajectoryData])

  // Simulations for the selected task, sorted by trial
  const taskSimulations = useMemo(() => {
    if (selectedTaskId === null) return []
    return simulationsByTask.get(selectedTaskId) || []
  }, [simulationsByTask, selectedTaskId])

  // The current simulation (for the selected trial)
  const currentSimulation = useMemo(() => {
    if (!taskSimulations.length) return null
    return taskSimulations[selectedTrialIdx] || taskSimulations[0]
  }, [taskSimulations, selectedTrialIdx])

  // A text conversation in the dir layout loads when opened; voice
  // conversations are loaded by VoiceViewer.
  useEffect(() => {
    if (isVoice || !currentCell || !currentSimulation?.stub) return
    const id = currentSimulation.id
    if (simDetails[id]) return
    setSimDetails(d => ({ ...d, [id]: { loading: true } }))
    loadSimulation(currentCell, id)
      .then(data => setSimDetails(d => ({ ...d, [id]: { data } })))
      .catch(err => setSimDetails(d => ({ ...d, [id]: { error: err.message } })))
  }, [isVoice, currentCell, currentSimulation, simDetails])

  const simDetail = currentSimulation?.stub ? simDetails[currentSimulation.id] : null
  // The full simulation when it is loaded, else its index entry.
  const shownSimulation = simDetail?.data || currentSimulation

  // Task metadata for the selected task
  const currentTask = useMemo(() => {
    if (selectedTaskId === null || !tasks.length) return null
    return tasks.find(t => t.id === selectedTaskId) || null
  }, [tasks, selectedTaskId])

  // Compute pass^k scores from trajectory data
  const passKScores = useMemo(() => {
    if (!trajectoryData?.simulations || !tasks.length) return null
    // Derive numTrials from actual data rather than info metadata (which can be wrong)
    const trialSet = new Set(trajectoryData.simulations.map(s => s.trial))
    const numTrials = trialSet.size
    if (numTrials === 0) return null

    // Helper: binomial coefficient C(n, k)
    const comb = (n, k) => {
      if (k < 0 || k > n) return 0
      if (k === 0 || k === n) return 1
      let result = 1
      for (let i = 0; i < Math.min(k, n - k); i++) {
        result = result * (n - i) / (i + 1)
      }
      return Math.round(result)
    }

    // Group simulations by task_id and count successes (reward ≈ 1.0)
    const taskSuccesses = {}
    const taskTrialCounts = {}
    for (const sim of trajectoryData.simulations) {
      const tid = sim.task_id
      if (!taskTrialCounts[tid]) { taskTrialCounts[tid] = 0; taskSuccesses[tid] = 0 }
      taskTrialCounts[tid]++
      const reward = sim.reward_info?.reward ?? 0
      if (Math.abs(reward - 1.0) < 1e-6) taskSuccesses[tid]++
    }

    const taskIds = Object.keys(taskTrialCounts)
    const n = taskIds.length
    if (n === 0) return null

    // Compute pass^k for k = 1..numTrials
    const maxK = Math.min(numTrials, Math.min(...Object.values(taskTrialCounts)))
    const scores = {}
    for (let k = 1; k <= maxK; k++) {
      let sum = 0
      for (const tid of taskIds) {
        sum += comb(taskSuccesses[tid], k) / comb(taskTrialCounts[tid], k)
      }
      scores[k] = (sum / n) * 100 // as percentage
    }
    return scores
  }, [trajectoryData, tasks])

  // The cell's own audio-native config, else the submission's voice_config.
  const voiceConfig = useMemo(() => {
    const native = trajectoryData?.info?.audio_native_config
    const config = currentEntry?.voiceConfig || {}
    return {
      ...config,
      provider: config.provider || native?.provider || '',
      model: config.model || native?.model || '',
      tick_duration_seconds: config.tick_duration_seconds || native?.tick_duration_seconds,
      domain: selectedDomain,
    }
  }, [trajectoryData, currentEntry, selectedDomain])

  // Build task info for the voice viewer from the current task's user scenario
  const voiceTaskInfo = useMemo(() => {
    if (!currentTask?.user_scenario?.instructions) return {}
    const instr = currentTask.user_scenario.instructions
    if (typeof instr === 'string') return { reason: instr }
    return {
      reason: instr.reason_for_call || '',
      knownInfo: instr.known_info || '',
      unknownInfo: instr.unknown_info || '',
      taskInstructions: instr.task_instructions || '',
    }
  }, [currentTask])

  // Text cells download as their monolithic file; dir cells (voice, and
  // anything else with per-simulation files) sync as a directory.
  const getDownloadCommand = () => {
    if (!currentCell || !currentEntry) return ''
    const slug = [currentEntry.model, language, selectedDomain].join('_').replace(/[^\w.-]+/g, '-')
    if (currentCell.flat) {
      return `aws s3 cp ${S3_SUBMISSIONS}/${currentCell.flat} ./${slug}.json --no-sign-request`
    }
    return `aws s3 sync ${S3_SUBMISSIONS}/${currentCell.path}/ ./${slug}/ --no-sign-request`
  }

  const handleCopyCliCommand = () => {
    navigator.clipboard.writeText(getDownloadCommand()).then(() => {
      setCliCopied(true)
      setTimeout(() => setCliCopied(false), 2000)
    })
  }

  // --- Handlers ---
  const handleBenchmarkChange = (id) => {
    if (id === benchmark) return
    setBenchmark(id)
    setSelectedId('')
    setSelectedDomain('')
    setSelectedTaskId(null)
    setSelectedTrialIdx(0)
    setTrajectoryData(null)
    setError(null)
  }

  const handleLanguageChange = (lang) => {
    const cellDomains = currentEntry.cells.filter(c => c.language === lang).map(c => c.domain)
    setLanguage(lang)
    if (!cellDomains.includes(selectedDomain)) setSelectedDomain(cellDomains[0] || '')
    setSelectedTaskId(null)
    setSelectedTrialIdx(0)
    setShowDownload(false)
  }

  const handleDomainChange = (domain) => {
    setSelectedDomain(domain)
    setSelectedTaskId(null)
    setSelectedTrialIdx(0)
    setShowDownload(false)
  }

  // --- Task mode ---
  const loadTaskData = async (domain) => {
    try {
      setLoading(true)
      setError(null)
      const [tasksRes, policyRes] = await Promise.all([
        fetch(`${import.meta.env.BASE_URL}task-data/domains/${domain}/tasks.json`),
        fetch(`${import.meta.env.BASE_URL}task-data/domains/${domain}/policy.md`)
      ])
      if (!tasksRes.ok) throw new Error(`Failed to load tasks: ${tasksRes.statusText}`)
      const tasksJson = await tasksRes.json()
      const policy = policyRes.ok ? await policyRes.text() : null
      setTaskData({ tasks: tasksJson, policy, domain })
      setSelectedDomainTask(domain)
      setSelectedTaskDetail(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // --- Helpers ---
  // Get a meaningful task description, preferring notes over generic "Task: task_XXX" purpose
  const getTaskDescription = (task) => {
    const purpose = task?.description?.purpose
    const notes = task?.description?.notes
    if (notes) return notes
    if (purpose && !/^Task:\s*task_\d+$/i.test(purpose)) return purpose
    return null
  }

  // Extract user scenario text from either string or object format
  const getUserScenarioText = (task) => {
    const instructions = task?.user_scenario?.instructions
    if (!instructions) return null
    if (typeof instructions === 'string') return instructions
    return instructions.reason_for_call || instructions.task_instructions || null
  }

  const getCleanTaskId = (taskId) => {
    if (!taskId && taskId !== 0) return 'Unknown'
    if (typeof taskId === 'number' || /^\d+$/.test(String(taskId))) return String(taskId)
    // Handle "task_001" format — extract just the number
    const taskNumMatch = String(taskId).match(/^task_0*(\d+)$/)
    if (taskNumMatch) return taskNumMatch[1]
    // Localized voice-v2 ids carry their language: "8_ko_identity" → "8"
    const localizedMatch = String(taskId).match(/^(\d+)_[a-z]{2}_identity$/)
    if (localizedMatch) return localizedMatch[1]
    if (String(taskId).length < 15) return String(taskId)
    const bracketMatch = String(taskId).match(/\[([^\]]+)\]/)
    if (bracketMatch) {
      return bracketMatch[1].replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    const cleaned = String(taskId).split(/[[|]/)[0].replace(/_/g, ' ')
    return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
  }

  const formatMessage = (message) => ({
    role: message.role,
    content: message.content,
    tool_calls: message.tool_calls,
    turn: message.turn_idx,
    timestamp: new Date(message.timestamp).toLocaleString(),
    cost: message.cost || 0,
    tokens: message.usage ? `${message.usage.prompt_tokens || 0}/${message.usage.completion_tokens || 0}` : 'N/A'
  })

  const getDisplayMessages = (sim) => {
    if (!sim?.messages) return []
    return sim.messages.slice(0, 60).map(formatMessage)
  }

  // --- Render ---
  return (
    <div className="trajectory-visualizer">
      <div className="visualizer-header">
        <h2>τ-bench Visualizer</h2>
        <p className="visualizer-description">
          Explore published trajectories: every task outcome and conversation from text and voice submissions, in every language they were run, or examine the underlying task definitions.
        </p>

        {/* View Mode Toggle */}
        <div className="view-toggle">
          <button
            className={`toggle-btn ${viewMode === 'trajectories' ? 'active' : ''}`}
            onClick={() => {
              setViewMode('trajectories')
              setTaskData(null)
              setSelectedTaskDetail(null)
              setSelectedDomainTask(null)
            }}
          >
            🔄 Trajectories
          </button>
          <button
            className={`toggle-btn ${viewMode === 'tasks' ? 'active' : ''}`}
            onClick={() => {
              setViewMode('tasks')
              setSelectedTaskId(null)
              setTrajectoryData(null)
            }}
          >
            📋 Tasks
          </button>
        </div>
      </div>

      {viewMode === 'trajectories' && (
        <>
          {/* ===== SELECTOR BAR ===== */}
          <div className="selector-bar">
            <div className="selector-group selector-group-full">
              <label className="selector-label">Benchmark</label>
              <div className="benchmark-tabs" role="tablist">
                {BENCHMARKS.map(b => {
                  const count = (entries || []).filter(e => e.benchmark === b.id).length
                  return (
                    <button
                      key={b.id}
                      role="tab"
                      aria-selected={benchmark === b.id}
                      className={`benchmark-tab ${benchmark === b.id ? 'active' : ''}`}
                      onClick={() => handleBenchmarkChange(b.id)}
                      disabled={!entries || !count}
                    >
                      <span className="benchmark-tab-label">{b.label}</span>
                      <span className="benchmark-tab-sub">{b.sub}{entries ? ` · ${count}` : ''}</span>
                    </button>
                  )
                })}
              </div>
            </div>

            {currentEntry && (
              <div className="selector-group">
                <label className="selector-label">Model</label>
                <ModelPicker entries={benchmarkEntries} selected={currentEntry} onSelect={entry => selectEntry(entry)} />
              </div>
            )}

            {entryLanguages.length > 1 && (
              <div className="selector-group">
                <label className="selector-label">Language</label>
                <div className="domain-pills">
                  {entryLanguages.map(l => (
                    <button
                      key={l.id}
                      className={`domain-pill language-pill ${language === l.id ? 'active' : ''}`}
                      onClick={() => handleLanguageChange(l.id)}
                    >
                      {l.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {currentEntry && (
              <div className="selector-group">
                <label className="selector-label">Domain</label>
                <div className="domain-pills">
                  {languageCells.map(cell => {
                    const d = DOMAINS[cell.domain] || { label: cell.domain, icon: '•', color: '#475569' }
                    const active = selectedDomain === cell.domain
                    return (
                      <button
                        key={cell.domain}
                        className={`domain-pill ${active ? 'active' : ''}`}
                        style={active ? { background: d.color, borderColor: d.color } : {}}
                        onClick={() => handleDomainChange(cell.domain)}
                        title={cell.score != null ? `pass^1 ${cell.score.toFixed(1)}%` : undefined}
                      >
                        {d.icon} {d.label}
                        {cell.score != null && <span className="domain-pill-score">{Math.round(cell.score)}%</span>}
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Config button */}
            {trajectoryData?.info && (
              <div className="selector-group selector-group-end">
                <button className="config-button" onClick={() => setShowConfigModal(true)}>
                  ⚙️ Config
                </button>
              </div>
            )}
          </div>

          {/* ===== MAIN CONTENT ===== */}
          <div className="viz-content">
            {indexError && (
              <div className="error-state"><p>⚠️ {indexError}</p></div>
            )}

            {!entries && !indexError && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Loading submissions...</p>
              </div>
            )}

            {/* Landing: no model chosen yet */}
            {entries && !currentEntry && (
              <div className="picker-landing">
                <div className="picker-landing-header">
                  <h3>Choose a model</h3>
                  <p>{BENCHMARKS.find(b => b.id === benchmark)?.label} · {BENCHMARKS.find(b => b.id === benchmark)?.sub} — {benchmarkEntries.length} submissions with published trajectories.</p>
                </div>
                <ModelPicker inline entries={benchmarkEntries} selected={null} onSelect={entry => selectEntry(entry)} />
              </div>
            )}

            {currentEntry && loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Loading tasks...</p>
              </div>
            )}

            {currentEntry && error && !loading && (
              <div className="error-state">
                <p>⚠️ {error}</p>
              </div>
            )}

            {/* Task list view — no task selected yet */}
            {currentEntry && !loading && !error && trajectoryData && selectedTaskId === null && (
              <div className="task-list-view">
                <div className="task-list-header">
                  <div className="task-list-header-left">
                    <h3>
                      {isVoice && <span className="voice-badge">🎙️ Voice</span>}
                      {currentEntry.model} <EntryChips entry={currentEntry} /> — {DOMAINS[selectedDomain]?.label || selectedDomain}
                      {entryLanguages.length > 1 && ` · ${LANGUAGES.find(l => l.id === language)?.name}`}
                    </h3>
                    <p className="task-list-subtitle">
                      {tasks.length} tasks · {numTrials} trial{numTrials !== 1 ? 's' : ''} each · Select a task to view conversations
                    </p>
                  </div>
                  <div className="task-list-header-right">
                    <div className="cli-download-wrapper">
                      <button
                        className="download-btn"
                        onClick={() => setShowDownload(!showDownload)}
                        title="Download the raw trajectory data"
                      >
                        ⬇ Download
                      </button>
                      {showDownload && (
                        <div className="cli-download-popover">
                          {currentCell?.flat ? (
                            <p className="cli-download-note">
                              The full results file for this domain (<a href={`${SUBMISSIONS_BASE}/${currentCell.flat}`} target="_blank" rel="noopener noreferrer">open JSON</a>), or with the AWS CLI:
                            </p>
                          ) : (
                            <p className="cli-download-note">Voice trajectories include audio and are large; download them with the AWS CLI:</p>
                          )}
                          <pre className="cli-download-command">{getDownloadCommand()}</pre>
                          <div className="cli-download-actions">
                            <button className="cli-copy-btn" onClick={handleCopyCliCommand}>
                              {cliCopied ? '✓ Copied' : 'Copy command'}
                            </button>
                            <button className="cli-close-btn" onClick={() => setShowDownload(false)}>
                              Close
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                    {passKScores && (
                      <div className="pass-k-scores">
                        {Object.entries(passKScores).map(([k, score]) => (
                          <div key={k} className="pass-k-item">
                            <span className="pass-k-label">pass^{k}</span>
                            <span className={`pass-k-value ${score >= 50 ? 'good' : score >= 25 ? 'mid' : 'low'}`}>
                              {score.toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {cellQuality && (
                  <div className="verdict-filter" role="group" aria-label="Filter by judge verdict">
                    <span>Show</span>
                    {[
                      ['all', 'All tasks'],
                      ['interaction', `Interaction fails (${verdictCounts.interaction})`],
                      ['experience', `Generation quality fails (${verdictCounts.experience})`],
                    ].map(([key, text]) => (
                      <button key={key} className={verdictFilter === key ? 'active' : ''} aria-pressed={verdictFilter === key} onClick={() => setVerdictFilter(key)}>{text}</button>
                    ))}
                  </div>
                )}

                <div className="task-grid">
                  {shownTasks.map((task, idx) => {
                    const sims = simulationsByTask.get(task.id) || []
                    const call = taskCalls.get(task.id)
                    const avgReward = sims.length > 0
                      ? (sims.reduce((sum, s) => sum + (s.reward_info?.reward || 0), 0) / sims.length)
                      : null
                    const desc = getTaskDescription(task)
                    const scenario = getUserScenarioText(task)

                    return (
                      <div
                        key={task.id ?? idx}
                        className="task-card"
                        onClick={() => { setSelectedTaskId(task.id); setSelectedTrialIdx(0) }}
                      >
                        <div className="task-header">
                          <span className="task-id">Task {getCleanTaskId(task.id)}</span>
                          {avgReward !== null && (
                            <span className={`task-reward ${avgReward >= 0.5 ? 'good' : 'bad'}`}>
                              {avgReward.toFixed(2)}
                            </span>
                          )}
                        </div>
                        <div className="task-description">
                          <p>{desc ? desc.slice(0, 120) + (desc.length > 120 ? '…' : '') : (task.description?.purpose || 'No description')}</p>
                        </div>
                        <div className="task-stats">
                          <span>{sims.length} trial{sims.length !== 1 ? 's' : ''}</span>
                          <span>{scenario ? '📞 ' + scenario.slice(0, 50) + (scenario.length > 50 ? '…' : '') : ''}</span>
                        </div>
                        {call && <div className="task-verdicts"><CallVerdicts call={call} compact /></div>}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Conversation view — task selected */}
            {currentEntry && !loading && !error && trajectoryData && selectedTaskId !== null && currentSimulation && (
              <div className="conversation-view">
                <div className="conversation-header">
                  <div className="conversation-meta">
                    <button className="back-button" onClick={() => setSelectedTaskId(null)}>
                      ← All Tasks
                    </button>
                    <h3>Task {getCleanTaskId(selectedTaskId)}</h3>
                    <div className="trial-indicator">
                      Trial {selectedTrialIdx + 1} of {numTrials}
                    </div>
                  </div>

                  {/* Trial selector pills inline */}
                  {numTrials > 1 && (
                    <div className="trial-selector-inline">
                      {Array.from({ length: numTrials }, (_, i) => {
                        const sim = taskSimulations[i]
                        const reward = sim?.reward_info?.reward
                        return (
                          <button
                            key={i}
                            className={`trial-pill-inline ${selectedTrialIdx === i ? 'active' : ''} ${reward !== undefined ? (reward >= 0.5 ? 'success' : 'fail') : ''}`}
                            onClick={() => setSelectedTrialIdx(i)}
                          >
                            Trial {i + 1}
                            {reward !== undefined && <span className="trial-reward">{reward.toFixed(1)}</span>}
                          </button>
                        )
                      })}
                    </div>
                  )}

                  {/* Task context */}
                  {currentTask && (
                    <div className="task-context">
                      <h4>Task Context</h4>
                      <p><strong>Purpose:</strong> {currentTask.description?.purpose}</p>
                      {typeof currentTask.user_scenario?.instructions === 'string' ? (
                        <p><strong>User Scenario:</strong> {currentTask.user_scenario.instructions.slice(0, 300)}{currentTask.user_scenario.instructions.length > 300 ? '…' : ''}</p>
                      ) : (
                        <>
                          <p><strong>User Scenario:</strong> {currentTask.user_scenario?.instructions?.reason_for_call}</p>
                          <p><strong>Known Info:</strong> {currentTask.user_scenario?.instructions?.known_info}</p>
                        </>
                      )}
                    </div>
                  )}

                  {/* Simulation results */}
                  <div className="simulation-results">
                    <h4>Results</h4>
                    <div className="results-grid">
                      <div className="result-item">
                        <span className="result-label">Reward</span>
                        <span className={`result-value ${(shownSimulation.reward_info?.reward ?? 0) >= 0.5 ? 'good' : 'bad'}`}>
                          {shownSimulation.reward_info?.reward?.toFixed(2) ?? 'N/A'}
                        </span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Termination</span>
                        <span className="result-value">{shownSimulation.termination_reason || '—'}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Agent Cost</span>
                        <span className="result-value">${shownSimulation.agent_cost?.toFixed(4) ?? '—'}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Duration</span>
                        <span className="result-value">{shownSimulation.duration ? `${Math.round(shownSimulation.duration)}s` : '—'}</span>
                      </div>
                    </div>

                    {shownSimulation.reward_info?.nl_assertions?.length > 0 && (
                      <div className="assertions">
                        <h5>Evaluation Assertions</h5>
                        <div className="assertion-list">
                          {shownSimulation.reward_info.nl_assertions.map((a, i) => (
                            <div key={i} className={`assertion ${a.met ? 'passed' : 'failed'}`}>
                              <span className="assertion-status">{a.met ? '✅' : '❌'}</span>
                              <span className="assertion-text">{a.nl_assertion}</span>
                              {a.justification && <p className="assertion-justification">{a.justification}</p>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {(currentCell.judges || currentCell.verdicts) && (
                  <JudgeFindings
                    key={currentSimulation.id}
                    cell={currentCell}
                    simulationId={currentSimulation.id}
                    call={cellQuality?.get(currentSimulation.id)}
                  />
                )}

                {/* Voice: timeline + conversation table; Text: message bubbles */}
                {isVoice ? (
                  <VoiceViewer
                    key={`${currentSimulation.id}-${selectedTrialIdx}`}
                    cellUrl={`${SUBMISSIONS_BASE}/${currentCell.path}`}
                    simulationId={currentSimulation.id}
                    taskId={selectedTaskId}
                    voiceConfig={voiceConfig}
                    taskInfo={voiceTaskInfo}
                  />
                ) : simDetail?.error ? (
                  <div className="error-state">
                    <p>⚠️ {simDetail.error}</p>
                    {/* Dropping the entry makes the load effect fetch it again. */}
                    <button className="retry-btn" onClick={() => setSimDetails(d => {
                      const next = { ...d }
                      delete next[currentSimulation.id]
                      return next
                    })}>Retry</button>
                  </div>
                ) : !shownSimulation.messages ? (
                  <div className="loading-state">
                    <div className="loading-spinner"></div>
                    <p>Loading conversation...</p>
                  </div>
                ) : (
                  <div className="conversation-messages">
                    {getDisplayMessages(shownSimulation).map((msg, i) => (
                      <div key={i} className={`message ${msg.role}`}>
                        <div className="message-header">
                          <span className="message-role">
                            {msg.role === 'assistant' ? '🤖 Agent' : msg.role === 'tool' ? '🔧 Tool' : '👤 User'}
                          </span>
                          <span className="message-turn">Turn {msg.turn}</span>
                          {msg.cost > 0 && <span className="message-cost">${msg.cost.toFixed(4)}</span>}
                          <span className="message-tokens">{msg.tokens} tokens</span>
                        </div>
                        <div className="message-content">{msg.content}</div>
                        {msg.tool_calls && (
                          <div className="message-tools">
                            <strong>Tool Calls:</strong>
                            <pre>{JSON.stringify(msg.tool_calls, null, 2)}</pre>
                          </div>
                        )}
                      </div>
                    ))}
                    {shownSimulation.messages.length > 60 && (
                      <div className="message-truncated">
                        <p>… and {shownSimulation.messages.length - 60} more messages (showing first 60)</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {/* ===== TASK MODE ===== */}
      {viewMode === 'tasks' && (
        <div className="task-mode-container">
          <div className="selector-bar">
            <div className="selector-group">
              <label className="selector-label">Domain</label>
              <div className="domain-pills">
                {domains.map(d => (
                  <button
                    key={d.id}
                    className={`domain-pill ${selectedDomainTask === d.id ? 'active' : ''}`}
                    style={selectedDomainTask === d.id ? { background: d.color, borderColor: d.color } : {}}
                    onClick={() => loadTaskData(d.id)}
                  >
                    {d.icon} {d.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="viz-content">
            {loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Loading task data...</p>
              </div>
            )}

            {error && !loading && (
              <div className="error-state"><p>⚠️ {error}</p></div>
            )}

            {!loading && !error && !taskData && (
              <div className="empty-state">
                <h3>Select a Domain</h3>
                <p>Choose a domain above to explore task definitions and agent policies.</p>
              </div>
            )}

            {/* Task overview list */}
            {!loading && taskData && !selectedTaskDetail && (
              <div className="task-list-view">
                <div className="task-list-header">
                  <h3>{domains.find(d => d.id === taskData.domain)?.label || taskData.domain} Tasks</h3>
                  <p className="task-list-subtitle">{taskData.tasks?.length || 0} task definitions</p>
                </div>
                <div className="task-grid">
                  {taskData.tasks?.map((task, idx) => {
                    const scenarioText = getUserScenarioText(task)
                    const description = getTaskDescription(task)
                    return (
                      <div key={task.id || idx} className="task-card" onClick={() => setSelectedTaskDetail(task)}>
                        <div className="task-header">
                          <span className="task-id">Task {getCleanTaskId(task.id)}</span>
                          {task.required_documents?.length > 0 && (
                            <span className="task-badge docs-badge">📄 {task.required_documents.length} docs</span>
                          )}
                        </div>
                        <div className="task-description">
                          {description ? (
                            <p>{description.slice(0, 180)}{description.length > 180 ? '…' : ''}</p>
                          ) : (
                            <p><strong>Scenario:</strong> {scenarioText ? scenarioText.slice(0, 120) + (scenarioText.length > 120 ? '…' : '') : '—'}</p>
                          )}
                        </div>
                        <div className="task-stats">
                          <span>{task.evaluation_criteria?.actions?.length || 0} actions</span>
                          <span>{task.evaluation_criteria?.nl_assertions?.length || 0} assertions</span>
                          {task.evaluation_criteria?.communicate_info?.length > 0 && (
                            <span>{task.evaluation_criteria.communicate_info.length} info items</span>
                          )}
                          {task.user_tools?.length > 0 && (
                            <span>🔧 {task.user_tools.length} user tools</span>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Task detail */}
            {!loading && taskData && selectedTaskDetail && (
              <div className="task-detail-view">
                <div className="task-detail-header">
                  <button className="back-button" onClick={() => setSelectedTaskDetail(null)}>← Back to Tasks</button>
                  <h3>Task {getCleanTaskId(selectedTaskDetail.id)}</h3>
                </div>

                <div className="task-detail-content">
                  <div className="task-section">
                    <h4>Task Description</h4>
                    <div className="task-info">
                      {selectedTaskDetail.description?.notes ? (
                        <p>{selectedTaskDetail.description.notes}</p>
                      ) : (
                        <p>{selectedTaskDetail.description?.purpose || '—'}</p>
                      )}
                    </div>
                  </div>

                  <div className="task-section">
                    <h4>User Scenario</h4>
                    <div className="task-info">
                      {typeof selectedTaskDetail.user_scenario?.instructions === 'string' ? (
                        <>
                          {selectedTaskDetail.user_scenario?.persona && (
                            <p><strong>Persona:</strong> {selectedTaskDetail.user_scenario.persona}</p>
                          )}
                          <pre className="instructions-text">{selectedTaskDetail.user_scenario.instructions}</pre>
                        </>
                      ) : (
                        <>
                          <p><strong>Domain:</strong> {selectedTaskDetail.user_scenario?.instructions?.domain}</p>
                          <p><strong>Reason for Call:</strong> {selectedTaskDetail.user_scenario?.instructions?.reason_for_call}</p>
                          <p><strong>Known Information:</strong> {selectedTaskDetail.user_scenario?.instructions?.known_info}</p>
                          {selectedTaskDetail.user_scenario?.instructions?.unknown_info && (
                            <p><strong>Unknown Information:</strong> {selectedTaskDetail.user_scenario.instructions.unknown_info}</p>
                          )}
                          {selectedTaskDetail.user_scenario?.instructions?.task_instructions && (
                            <div>
                              <p><strong>Task Instructions:</strong></p>
                              <pre className="instructions-text">{selectedTaskDetail.user_scenario.instructions.task_instructions}</pre>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>

                  {selectedTaskDetail.user_tools?.length > 0 && (
                    <div className="task-section">
                      <h4>User Tools</h4>
                      <div className="user-tools-list">
                        {selectedTaskDetail.user_tools.map((tool, i) => (
                          <span key={i} className="user-tool-badge">{tool}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedTaskDetail.required_documents?.length > 0 && (
                    <div className="task-section">
                      <h4>Required Documents ({selectedTaskDetail.required_documents.length})</h4>
                      <p className="doc-hint">Click a document to view its contents</p>
                      <div className="required-docs-list">
                        {selectedTaskDetail.required_documents.map((doc, i) => (
                          <button key={i} className="doc-badge clickable" onClick={() => handleDocClick(doc)}>
                            <span className="doc-icon">📄</span>
                            <span className="doc-name">{doc.replace(/^doc_/, '').replace(/_/g, ' ')}</span>
                            <span className="doc-arrow">→</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedTaskDetail.evaluation_criteria && (
                    <div className="task-section">
                      <h4>Evaluation Criteria</h4>
                      {selectedTaskDetail.evaluation_criteria.reward_basis?.length > 0 && (
                        <div className="criteria-subsection">
                          <h5>Reward Basis</h5>
                          <div className="reward-basis-list">
                            {selectedTaskDetail.evaluation_criteria.reward_basis.map((basis, i) => (
                              <span key={i} className="reward-basis-badge">{basis}</span>
                            ))}
                          </div>
                        </div>
                      )}
                      {selectedTaskDetail.evaluation_criteria.actions && (
                        <div className="criteria-subsection">
                          <h5>Expected Actions ({selectedTaskDetail.evaluation_criteria.actions.length})</h5>
                          <div className="actions-list">
                            {selectedTaskDetail.evaluation_criteria.actions.length > 0 ? (
                              selectedTaskDetail.evaluation_criteria.actions.map((action, i) => (
                                <div key={i} className="action-item">
                                  <p><strong>Action:</strong> {action.name}</p>
                                  {action.requestor && <p><strong>Requestor:</strong> {action.requestor}</p>}
                                  {action.arguments && <pre className="action-args">{JSON.stringify(action.arguments, null, 2)}</pre>}
                                </div>
                              ))
                            ) : (
                              <div className="no-actions-message"><p>Agent should not take any action</p></div>
                            )}
                          </div>
                        </div>
                      )}
                      {selectedTaskDetail.evaluation_criteria.communicate_info?.length > 0 && (
                        <div className="criteria-subsection">
                          <h5>Required Communication ({selectedTaskDetail.evaluation_criteria.communicate_info.length})</h5>
                          <div className="communicate-info-list">
                            {selectedTaskDetail.evaluation_criteria.communicate_info.map((info, i) => (
                              <div key={i} className="communicate-info-item">
                                <p>{typeof info === 'string' ? info : JSON.stringify(info)}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {selectedTaskDetail.evaluation_criteria.nl_assertions?.length > 0 && (
                        <div className="criteria-subsection">
                          <h5>Natural Language Assertions ({selectedTaskDetail.evaluation_criteria.nl_assertions.length}) <span className="experimental-badge-container"><span className="experimental-badge">experimental</span><div className="experimental-tooltip">These assertions are experimental and not used to compute benchmark scores</div></span></h5>
                          <div className="assertions-list">
                            {selectedTaskDetail.evaluation_criteria.nl_assertions.map((a, i) => (
                              <div key={i} className="assertion-item"><p>{a}</p></div>
                            ))}
                          </div>
                        </div>
                      )}
                      {selectedTaskDetail.evaluation_criteria.env_assertions?.length > 0 && (
                        <div className="criteria-subsection">
                          <h5>Environment Assertions ({selectedTaskDetail.evaluation_criteria.env_assertions.length})</h5>
                          <div className="env-assertions-list">
                            {selectedTaskDetail.evaluation_criteria.env_assertions.map((a, i) => (
                              <div key={i} className="env-assertion-item">
                                <p><strong>Function:</strong> {a.func_name}</p>
                                <pre className="assertion-args">{JSON.stringify(a.arguments, null, 2)}</pre>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {selectedTaskDetail.initial_state?.initialization_actions && (
                    <div className="task-section">
                      <h4>Initial State</h4>
                      <pre className="initial-actions">{JSON.stringify(selectedTaskDetail.initial_state.initialization_actions, null, 2)}</pre>
                    </div>
                  )}

                  {taskData?.policy && (
                    <div className="task-section">
                      <h4>Domain Policy</h4>
                      <pre className="policy-text">{taskData.policy}</pre>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ===== DOCUMENT VIEWER MODAL ===== */}
      {showDocModal && (
        <div className="modal-overlay" onClick={handleCloseDocModal}>
          <div className={`modal-content doc-modal ${docModalClosing ? 'closing' : ''}`} onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{selectedDoc?.title || 'Loading document...'}</h3>
              <button className="modal-close" onClick={handleCloseDocModal}>✕</button>
            </div>
            <div className="doc-modal-body">
              {docLoading && (
                <div className="doc-loading">
                  <div className="loading-spinner"></div>
                  <p>Loading document...</p>
                </div>
              )}
              {!docLoading && selectedDoc && (
                <>
                  <div className="doc-id-label">
                    <code>{selectedDoc.id}</code>
                  </div>
                  <div className="doc-content-rendered">
                    {selectedDoc.content.split('\n').map((line, i) => {
                      // Simple markdown rendering for headings, lists, and text
                      if (line.startsWith('## ')) {
                        return <h3 key={i} className="doc-heading">{line.slice(3)}</h3>
                      }
                      if (line.startsWith('### ')) {
                        return <h4 key={i} className="doc-subheading">{line.slice(4)}</h4>
                      }
                      if (/^\d+\.\s/.test(line)) {
                        return <p key={i} className="doc-ordered-item">{line}</p>
                      }
                      if (line.startsWith('- ')) {
                        return <p key={i} className="doc-list-item">{line}</p>
                      }
                      if (line.startsWith('   - ')) {
                        return <p key={i} className="doc-list-item nested">{line.trim()}</p>
                      }
                      if (line.trim() === '') {
                        return <br key={i} />
                      }
                      return <p key={i} className="doc-paragraph">{line}</p>
                    })}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ===== CONFIG MODAL ===== */}
      {showConfigModal && trajectoryData && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className={`modal-content config-modal ${modalClosing ? 'closing' : ''}`} onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Configuration</h3>
              <button className="modal-close" onClick={handleCloseModal}>✕</button>
            </div>
            <div className="config-table-body">
              <table className="config-table">
                <tbody>
                  {trajectoryData.info?.agent_info && (
                    <>
                      <tr className="config-section-header"><td colSpan="2">Agent</td></tr>
                      <tr><td>Implementation</td><td><code>{trajectoryData.info.agent_info.implementation}</code></td></tr>
                      <tr><td>Model</td><td><code>{trajectoryData.info.agent_info.llm}</code></td></tr>
                      {trajectoryData.info.agent_info.llm_args && Object.entries(trajectoryData.info.agent_info.llm_args).map(([k, v]) => (
                        <tr key={k}><td className="config-indent">{k}</td><td><code>{JSON.stringify(v)}</code></td></tr>
                      ))}
                    </>
                  )}
                  {trajectoryData.info?.user_info && (
                    <>
                      <tr className="config-section-header"><td colSpan="2">User Simulator</td></tr>
                      <tr><td>Implementation</td><td><code>{trajectoryData.info.user_info.implementation}</code></td></tr>
                      <tr><td>Model</td><td><code>{trajectoryData.info.user_info.llm}</code></td></tr>
                    </>
                  )}
                  {trajectoryData.info && (
                    <>
                      <tr className="config-section-header"><td colSpan="2">Evaluation</td></tr>
                      {trajectoryData.info.num_trials && <tr><td>Trials</td><td><code>{trajectoryData.info.num_trials}</code></td></tr>}
                      {trajectoryData.info.max_steps && <tr><td>Max Steps</td><td><code>{trajectoryData.info.max_steps}</code></td></tr>}
                      {trajectoryData.info.max_errors && <tr><td>Max Errors</td><td><code>{trajectoryData.info.max_errors}</code></td></tr>}
                      {trajectoryData.info.seed !== undefined && <tr><td>Seed</td><td><code>{trajectoryData.info.seed}</code></td></tr>}
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default TrajectoryVisualizer
