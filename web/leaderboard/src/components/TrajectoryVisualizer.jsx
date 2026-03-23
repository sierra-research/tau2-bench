import { useState, useEffect, useMemo } from 'react'
import './TrajectoryVisualizer.css'

const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`

const TrajectoryVisualizer = () => {
  // --- Top-level selector state ---
  const [viewMode, setViewMode] = useState('trajectories') // 'trajectories' or 'tasks'
  const [submissions, setSubmissions] = useState([])
  const [submissionsLoading, setSubmissionsLoading] = useState(true)

  // Trajectory mode selectors
  const [selectedModelDir, setSelectedModelDir] = useState('')
  const [selectedDomain, setSelectedDomain] = useState('')
  const [selectedTrialIdx, setSelectedTrialIdx] = useState(0)
  const [selectedTaskId, setSelectedTaskId] = useState(null)

  // Loaded trajectory data for current model+domain
  const [trajectoryData, setTrajectoryData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Task mode state
  const [taskData, setTaskData] = useState(null)
  const [selectedDomainTask, setSelectedDomainTask] = useState(null)
  const [selectedTaskDetail, setSelectedTaskDetail] = useState(null)

  // Configuration modal
  const [showConfigModal, setShowConfigModal] = useState(false)
  const [modalClosing, setModalClosing] = useState(false)

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

  // Check if a submission has any trajectory files
  const checkSubmissionHasTrajectories = async (submission) => {
    // Use the declared trajectories_available field from the submission
    // This is much more reliable than trying to guess file patterns
    return submission.trajectories_available === true
  }

  // Load submissions data from the manifest
  const loadSubmissions = async () => {
    try {
      setSubmissionsLoading(true)
      setError(null)
      
      // Load the manifest file to get list of submissions
      const manifestResponse = await fetch(`${SUBMISSIONS_BASE}/manifest.json`)
      if (!manifestResponse.ok) {
        throw new Error('Failed to load submissions manifest')
      }
      
      const manifest = await manifestResponse.json()
      const submissionDirs = [
        ...(manifest.submissions || []),
        ...(manifest.legacy_submissions || [])
      ]
      
      const loadedSubmissions = []
      
      // Load each submission from its directory
      for (const submissionDir of submissionDirs) {
        try {
          const response = await fetch(`${SUBMISSIONS_BASE}/${submissionDir}/submission.json`)
          if (!response.ok) {
            console.warn(`Failed to load ${submissionDir}: ${response.status}`)
            continue
          }
          
          const submission = await response.json()
          
          // Check if this submission has any trajectory files
          const hasTrajectories = await checkSubmissionHasTrajectories({
            ...submission,
            submissionDir
          })
          
          // Store submission data with directory info and trajectory availability
          loadedSubmissions.push({
            ...submission,
            submissionDir, // Include directory name for trajectory access
            hasTrajectories // Flag indicating if trajectories are available
          })
        } catch (error) {
          console.warn(`Error loading ${submissionDir}:`, error)
        }
      }
      
      // Sort submissions: new first, then those with trajectories, then alphabetically
      const sortedSubmissions = loadedSubmissions.sort((a, b) => {
        // New submissions come first
        if (a.is_new !== b.is_new) {
          return (b.is_new ? 1 : 0) - (a.is_new ? 1 : 0)
        }
        // Then sort by trajectory availability
        if (a.hasTrajectories !== b.hasTrajectories) {
          return b.hasTrajectories - a.hasTrajectories
        }
        // Finally sort by model name
        return a.model_name.localeCompare(b.model_name)
      })
      
      setSubmissions(sortedSubmissions)
    } catch (error) {
      console.error('Error loading submissions:', error)
      setError(error.message)
    } finally {
      setSubmissionsLoading(false)
    }
  }

  // Load available trajectories for a selected submission
  const loadSubmissionTrajectories = async (submission) => {
    try {
      setLoading(true)
      setError(null)
      
      const submissionDir = submission.submissionDir
      const domains = ['airline', 'retail', 'telecom']
      const trajectories = []
      
      // Read trajectory filenames directly from submission.json
      const trajectoryFiles = submission.trajectory_files || {}
      
      for (const domain of domains) {
        const fileName = trajectoryFiles[domain]
        if (fileName) {
          trajectories.push({
            name: `${submission.model_name} - ${domain.charAt(0).toUpperCase() + domain.slice(1)}`,
            file: fileName,
            domain: domain,
            model: submission.model_name,
            submissionDir: submissionDir
          })
        }
      }
      
      setAvailableTrajectories(trajectories)
      setSelectedSubmission(submission)
    } catch (error) {
      setError(`Error loading trajectories: ${error.message}`)
      console.error('Error loading trajectories:', error)
    } finally {
      setLoading(false)
    }
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

  // Available domains
  const domains = [
    { id: 'airline', label: 'Airline', icon: '✈️', color: '#3b82f6' },
    { id: 'retail', label: 'Retail', icon: '🛍️', color: '#8b5cf6' },
    { id: 'telecom', label: 'Telecom', icon: '📱', color: '#059669' },
    { id: 'banking_knowledge', label: 'Banking', icon: '🏦', color: '#d97706' }
  ]

  // --- Parse URL params for deep linking (e.g. from leaderboard) ---
  const getUrlParams = () => {
    const hash = window.location.hash || ''
    const qIdx = hash.indexOf('?')
    if (qIdx === -1) return {}
    const params = new URLSearchParams(hash.slice(qIdx + 1))
    return {
      model: params.get('model'),
      domain: params.get('domain')
    }
  }

  // --- Load submissions on mount ---
  useEffect(() => {
    const loadSubmissions = async () => {
      try {
        setSubmissionsLoading(true)
        const res = await fetch(`${SUBMISSIONS_BASE}/manifest.json`)
        if (!res.ok) throw new Error('Failed to load manifest')
        const manifest = await res.json()
        const dirs = manifest.submissions || []

        const loaded = []
        for (const dir of dirs) {
          try {
            const r = await fetch(`${SUBMISSIONS_BASE}/${dir}/submission.json`)
            if (!r.ok) continue
            const sub = await r.json()
            if (sub.trajectories_available && sub.trajectory_files) {
              loaded.push({
                dir,
                model_name: sub.model_name,
                model_organization: sub.model_organization || '',
                reasoning_effort: sub.reasoning_effort || null,
                trajectory_files: sub.trajectory_files,
                availableDomains: Object.keys(sub.trajectory_files)
              })
            }
          } catch { /* skip */ }
        }
        loaded.sort((a, b) => a.model_name.localeCompare(b.model_name))
        setSubmissions(loaded)

        // Check URL params for deep linking from leaderboard
        const urlParams = getUrlParams()
        if (urlParams.model) {
          const match = loaded.find(s => s.dir === urlParams.model)
          if (match) {
            setSelectedModelDir(match.dir)
            const dom = urlParams.domain && match.availableDomains.includes(urlParams.domain)
              ? urlParams.domain
              : match.availableDomains[0]
            setSelectedDomain(dom || '')
            setViewMode('trajectories')
          } else {
            // Fallback to first model
            if (loaded.length > 0) {
              setSelectedModelDir(loaded[0].dir)
              setSelectedDomain(loaded[0].availableDomains[0] || '')
            }
          }
        } else if (loaded.length > 0) {
          // Default: auto-select first model
          setSelectedModelDir(loaded[0].dir)
          setSelectedDomain(loaded[0].availableDomains[0] || '')
        }
      } catch (err) {
        setError(err.message)
      } finally {
        setSubmissionsLoading(false)
      }
    }
    loadSubmissions()
  }, [])

  // --- Load trajectory data when model+domain changes ---
  useEffect(() => {
    if (viewMode !== 'trajectories' || !selectedModelDir || !selectedDomain) return

    const sub = submissions.find(s => s.dir === selectedModelDir)
    if (!sub) return
    const fileName = sub.trajectory_files[selectedDomain]
    if (!fileName) return

    const loadData = async () => {
      try {
        setLoading(true)
        setError(null)
        setTrajectoryData(null)
        setSelectedTaskId(null)
        setSelectedTrialIdx(0)

        const res = await fetch(
          `${SUBMISSIONS_BASE}/${sub.dir}/trajectories/${fileName}`
        )
        if (!res.ok) throw new Error(`Failed to load trajectory: ${res.statusText}`)
        const data = await res.json()
        setTrajectoryData(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [selectedModelDir, selectedDomain, viewMode, submissions])

  // --- Derived data ---
  const currentSubmission = useMemo(
    () => submissions.find(s => s.dir === selectedModelDir),
    [submissions, selectedModelDir]
  )

  // All unique tasks from trajectory data
  const tasks = useMemo(() => {
    if (!trajectoryData?.tasks) return []
    return trajectoryData.tasks
  }, [trajectoryData])

  // Number of distinct trials available
  const numTrials = useMemo(() => {
    if (!trajectoryData?.simulations) return 0
    const trials = new Set(trajectoryData.simulations.map(s => s.trial))
    return trials.size
  }, [trajectoryData])

  // Simulations for the selected task, sorted by trial
  const taskSimulations = useMemo(() => {
    if (!trajectoryData?.simulations || selectedTaskId === null) return []
    return trajectoryData.simulations
      .filter(s => s.task_id === selectedTaskId)
      .sort((a, b) => a.trial - b.trial)
  }, [trajectoryData, selectedTaskId])

  // The current simulation (for the selected trial)
  const currentSimulation = useMemo(() => {
    if (!taskSimulations.length) return null
    return taskSimulations[selectedTrialIdx] || taskSimulations[0]
  }, [taskSimulations, selectedTrialIdx])

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

  // --- Handlers ---
  const handleModelChange = (dir) => {
    setSelectedModelDir(dir)
    setSelectedTaskId(null)
    setSelectedTrialIdx(0)
    const sub = submissions.find(s => s.dir === dir)
    if (sub && sub.availableDomains.length > 0) {
      // Keep current domain if available, else pick first
      if (!sub.availableDomains.includes(selectedDomain)) {
        setSelectedDomain(sub.availableDomains[0])
      }
    }
  }

  const handleDomainChange = (domain) => {
    setSelectedDomain(domain)
    setSelectedTaskId(null)
    setSelectedTrialIdx(0)
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
    if (String(taskId).length < 15) return String(taskId)
    const bracketMatch = String(taskId).match(/\[([^\]]+)\]/)
    if (bracketMatch) {
      return bracketMatch[1].replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    const cleaned = String(taskId).split(/[[\|]/)[0].replace(/_/g, ' ')
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
          Explore τ-bench dataset: view conversation trajectories showing AI agent interactions with users,
          or examine the underlying task definitions across airline, retail, telecom, and banking domains.
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
            {/* Model dropdown */}
            <div className="selector-group">
              <label className="selector-label">Model</label>
              <select
                className="selector-dropdown"
                value={selectedModelDir}
                onChange={e => handleModelChange(e.target.value)}
                disabled={submissionsLoading}
              >
                {submissionsLoading && <option value="">Loading...</option>}
                {submissions.map(s => (
                  <option key={s.dir} value={s.dir}>
                    {s.model_name}{s.reasoning_effort ? ` [${s.reasoning_effort}]` : ''} ({s.model_organization})
                  </option>
                ))}
              </select>
            </div>

            {/* Domain dropdown */}
            <div className="selector-group">
              <label className="selector-label">Domain</label>
              <div className="domain-pills">
                {domains.map(d => {
                  const available = currentSubmission?.availableDomains?.includes(d.id)
                  return (
                    <button
                      key={d.id}
                      className={`domain-pill ${selectedDomain === d.id ? 'active' : ''} ${!available ? 'disabled' : ''}`}
                      style={selectedDomain === d.id ? { background: d.color, borderColor: d.color } : {}}
                      onClick={() => available && handleDomainChange(d.id)}
                      disabled={!available}
                    >
                      {d.icon} {d.label}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Trial selector */}
            {numTrials > 0 && selectedTaskId !== null && (
              <div className="selector-group">
                <label className="selector-label">Trial</label>
                <div className="trial-pills">
                  {Array.from({ length: numTrials }, (_, i) => (
                    <button
                      key={i}
                      className={`trial-pill ${selectedTrialIdx === i ? 'active' : ''}`}
                      onClick={() => setSelectedTrialIdx(i)}
                    >
                      {i + 1}
                    </button>
                  ))}
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
            {loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Loading trajectory data...</p>
                <p className="loading-note">Large files may take a moment</p>
              </div>
            )}

            {error && !loading && (
              <div className="error-state">
                <p>⚠️ {error}</p>
              </div>
            )}

            {!loading && !error && !trajectoryData && !submissionsLoading && (
              <div className="empty-state">
                <h3>Select a Model &amp; Domain</h3>
                <p>Choose a model and domain above to explore trajectory data.</p>
              </div>
            )}

            {/* Task list view — no task selected yet */}
            {!loading && !error && trajectoryData && selectedTaskId === null && (
              <div className="task-list-view">
                <div className="task-list-header">
                  <div className="task-list-header-left">
                    <h3>{currentSubmission?.model_name}{currentSubmission?.reasoning_effort ? ` [${currentSubmission.reasoning_effort}]` : ''} — {selectedDomain.charAt(0).toUpperCase() + selectedDomain.slice(1)}</h3>
                    <p className="task-list-subtitle">
                      {tasks.length} tasks · {numTrials} trial{numTrials !== 1 ? 's' : ''} each · Select a task to view conversations
                    </p>
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

                <div className="task-grid">
                  {tasks.map((task, idx) => {
                    // Find simulations for this task to show summary
                    const sims = trajectoryData.simulations?.filter(s => s.task_id === task.id) || []
                    const avgReward = sims.length > 0
                      ? (sims.reduce((sum, s) => sum + (s.reward_info?.reward || 0), 0) / sims.length)
                      : null

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
                          <p>{getTaskDescription(task) ? getTaskDescription(task).slice(0, 120) + (getTaskDescription(task).length > 120 ? '…' : '') : (task.description?.purpose || 'No description')}</p>
                        </div>
                        <div className="task-stats">
                          <span>{sims.length} trial{sims.length !== 1 ? 's' : ''}</span>
                          <span>{getUserScenarioText(task) ? '📞 ' + getUserScenarioText(task).slice(0, 50) + (getUserScenarioText(task).length > 50 ? '…' : '') : ''}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Conversation view — task selected */}
            {!loading && !error && trajectoryData && selectedTaskId !== null && currentSimulation && (
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
                        <span className={`result-value ${(currentSimulation.reward_info?.reward ?? 0) >= 0.5 ? 'good' : 'bad'}`}>
                          {currentSimulation.reward_info?.reward?.toFixed(2) ?? 'N/A'}
                        </span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Termination</span>
                        <span className="result-value">{currentSimulation.termination_reason || '—'}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Agent Cost</span>
                        <span className="result-value">${currentSimulation.agent_cost?.toFixed(4) ?? '—'}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Duration</span>
                        <span className="result-value">{currentSimulation.duration ? `${Math.round(currentSimulation.duration)}s` : '—'}</span>
                      </div>
                    </div>

                    {currentSimulation.reward_info?.nl_assertions?.length > 0 && (
                      <div className="assertions">
                        <h5>Evaluation Assertions</h5>
                        <div className="assertion-list">
                          {currentSimulation.reward_info.nl_assertions.map((a, i) => (
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

                {/* Messages */}
                <div className="conversation-messages">
                  {getDisplayMessages(currentSimulation).map((msg, i) => (
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
                  {currentSimulation.messages?.length > 60 && (
                    <div className="message-truncated">
                      <p>… and {currentSimulation.messages.length - 60} more messages (showing first 60)</p>
                    </div>
                  )}
                </div>
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
