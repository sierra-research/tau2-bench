import { useState, useEffect } from 'react'
import './TrajectoryVisualizer.css'

const TrajectoryVisualizer = () => {
  const [selectedTrajectory, setSelectedTrajectory] = useState(null)
  const [selectedTask, setSelectedTask] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // New state for view mode and task data
  const [viewMode, setViewMode] = useState('trajectories') // 'trajectories' or 'tasks'
  const [taskData, setTaskData] = useState(null)
  const [selectedTaskDetail, setSelectedTaskDetail] = useState(null)
  const [selectedDomain, setSelectedDomain] = useState(null)

  // New state for submission-based trajectory selection
  const [submissions, setSubmissions] = useState([])
  const [selectedSubmission, setSelectedSubmission] = useState(null)
  const [availableTrajectories, setAvailableTrajectories] = useState([])
  const [submissionsLoading, setSubmissionsLoading] = useState(false)

  // Modal state for configuration display
  const [showConfigModal, setShowConfigModal] = useState(false)
  const [modalClosing, setModalClosing] = useState(false)

  // Handle modal close with animation
  const handleCloseModal = () => {
    setModalClosing(true)
    setTimeout(() => {
      setShowConfigModal(false)
      setModalClosing(false)
    }, 300) // Match the CSS animation duration
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
      const manifestResponse = await fetch(`${import.meta.env.BASE_URL}submissions/manifest.json`)
      if (!manifestResponse.ok) {
        throw new Error('Failed to load submissions manifest')
      }
      
      const manifest = await manifestResponse.json()
      const submissionDirs = manifest.submissions || []
      
      const loadedSubmissions = []
      
      // Load each submission from its directory
      for (const submissionDir of submissionDirs) {
        try {
          const response = await fetch(`${import.meta.env.BASE_URL}submissions/${submissionDir}/submission.json`)
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
      
      // Sort submissions: those with trajectories first, then those without
      const sortedSubmissions = loadedSubmissions.sort((a, b) => {
        // If both have trajectories or both don't have trajectories, sort by model name
        if (a.hasTrajectories === b.hasTrajectories) {
          return a.model_name.localeCompare(b.model_name)
        }
        // Submissions with trajectories come first (true > false)
        return b.hasTrajectories - a.hasTrajectories
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
      
      // Map of exact trajectory file patterns based on actual file structure
      const trajectoryPatterns = {
        'claude-3.7-sonnet': [
          'claude-3-7-sonnet-20250219_{domain}_default_gpt-4.1-2025-04-14_4trials.json'
        ],
        'gpt-4.1': [
          'gpt-4.1-2025-04-14_{domain}_default_gpt-4.1-2025-04-14_4trials.json'
        ],
        'gpt-4.1-mini': [
          'gpt-4.1-mini-2025-04-14_{domain}_base_gpt-4.1-2025-04-14_4trials.json'
        ],
        'o4-mini': [
          'o4-mini-2025-04-16_{domain}_default_gpt-4.1-2025-04-14_4trials.json'
        ],
        'gpt-5': [
          'gpt-5_{domain}_default_gpt-4.1-2025-04-14_4trials.json'
        ]
      }
      
      // Get patterns for this exact model name (case-insensitive lookup)
      const modelKey = Object.keys(trajectoryPatterns).find(key => 
        key.toLowerCase() === submission.model_name.toLowerCase()
      )
      const patterns = modelKey ? trajectoryPatterns[modelKey] : []
      
      for (const domain of domains) {
        for (const pattern of patterns) {
          const fileName = pattern.replace('{domain}', domain)
          
          try {
            const response = await fetch(`${import.meta.env.BASE_URL}submissions/${submissionDir}/trajectories/${fileName}`, { method: 'HEAD' })
            if (response.ok) {
              trajectories.push({
                name: `${submission.model_name} - ${domain.charAt(0).toUpperCase() + domain.slice(1)}`,
                file: fileName,
                domain: domain,
                model: submission.model_name,
                submissionDir: submissionDir
              })
              break // Found a file for this domain, move to next domain
            }
          } catch {
            // File doesn't exist, try next pattern
          }
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

  // Available domains for task exploration
  const domains = [
    { name: 'Airline', id: 'airline', color: '#3b82f6' },
    { name: 'Retail', id: 'retail', color: '#8b5cf6' },
    { name: 'Telecom', id: 'telecom', color: '#059669' }
  ]

  // Load submissions on component mount
  useEffect(() => {
    if (viewMode === 'trajectories') {
      loadSubmissions()
    }
  }, [viewMode])

  const loadTrajectoryData = async (trajectoryInfo) => {
    try {
      setLoading(true)
      setError(null)
      
      // Construct the path based on submission directory and file
      const filePath = `${import.meta.env.BASE_URL}submissions/${trajectoryInfo.submissionDir}/trajectories/${trajectoryInfo.file}`
      
      // Fetch the JSON file from the submissions directory
      const response = await fetch(filePath)
      
      if (!response.ok) {
        throw new Error(`Failed to load trajectory data: ${response.statusText}`)
      }
      
      const data = await response.json()
      setSelectedTrajectory(data)
      setSelectedTask(null)
      setSelectedFile(trajectoryInfo.file)
      
    } catch (err) {
      setError(`Error loading trajectory: ${err.message}`)
      console.error('Error loading trajectory:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadTaskData = async (domain) => {
    try {
      setLoading(true)
      setError(null)
      
      // Load both tasks and policy for the domain
      const [tasksResponse, policyResponse] = await Promise.all([
        fetch(`${import.meta.env.BASE_URL}task-data/domains/${domain}/tasks.json`),
        fetch(`${import.meta.env.BASE_URL}task-data/domains/${domain}/policy.md`)
      ])
      
      if (!tasksResponse.ok) {
        throw new Error(`Failed to load tasks: ${tasksResponse.statusText}`)
      }
      
      const tasks = await tasksResponse.json()
      let policy = null
      
      if (policyResponse.ok) {
        policy = await policyResponse.text()
      }
      
      setTaskData({ tasks: tasks.slice(0, 50), policy, domain }) // Limit to first 50 for performance
      setSelectedDomain(domain)
      setSelectedTaskDetail(null)
      
    } catch (err) {
      setError(`Error loading task data: ${err.message}`)
      console.error('Error loading task data:', err)
    } finally {
      setLoading(false)
    }
  }

  const formatMessage = (message) => {
    const { role, content, tool_calls, turn_idx, timestamp, cost, usage } = message
    
    return {
      role,
      content,
      tool_calls,
      turn: turn_idx,
      timestamp: new Date(timestamp).toLocaleString(),
      cost: cost || 0,
      tokens: usage ? `${usage.prompt_tokens || 0}/${usage.completion_tokens || 0}` : 'N/A'
    }
  }

  const getDisplayMessages = (simulation) => {
    if (!simulation || !simulation.messages) return []
    
    // Limit to first 60 messages for performance
    const messages = simulation.messages.slice(0, 60)
    return messages.map(formatMessage)
  }

  const getCleanTaskId = (taskId) => {
    if (!taskId) return 'Unknown'
    
    // If it's a simple numeric or short string, return as is
    if (/^\d+$/.test(taskId) || taskId.length < 10) {
      return taskId
    }
    
    // For complex telecom task IDs like [mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]
    // Extract the main issue type from brackets
    const bracketMatch = taskId.match(/\[([^\]]+)\]/)
    if (bracketMatch) {
      const issueType = bracketMatch[1]
      // Convert snake_case to readable format
      return issueType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    
    // Fallback: just return the first part before any special characters
    const cleaned = taskId.split(/[[|]/)[0].replace(/_/g, ' ')
    return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
  }

  const getDomainColor = (domain) => {
    const colors = {
      airline: '#3b82f6',
      telecom: '#059669', 
      retail: '#8b5cf6'
    }
    return colors[domain] || '#6b7280'
  }

  return (
    <div className="trajectory-visualizer">
        <div className="visualizer-header">
          <h2>τ²-bench Visualizer</h2>
          <p className="visualizer-description">
            Explore τ²-bench dataset: view conversation trajectories showing AI agent interactions with users, 
            or examine the underlying task definitions that drive these conversations across airline, retail, and telecom domains.
          </p>
          
          {/* View Mode Toggle */}
          <div className="view-toggle">
            <button 
              className={`toggle-btn ${viewMode === 'trajectories' ? 'active' : ''}`}
              onClick={() => {
                setViewMode('trajectories')
                setTaskData(null)
                setSelectedTaskDetail(null)
                setSelectedDomain(null)
                setSelectedSubmission(null)
                setAvailableTrajectories([])
                setSelectedTrajectory(null)
                setSelectedTask(null)
              }}
            >
              🔄 Trajectories
            </button>
            <button 
              className={`toggle-btn ${viewMode === 'tasks' ? 'active' : ''}`}
              onClick={() => {
                setViewMode('tasks')
                setSelectedTrajectory(null)
                setSelectedTask(null)
                setSelectedFile(null)
              }}
            >
              📋 Tasks
            </button>
          </div>
        </div>

        <div className="trajectory-grid">
          {/* Selection Panel - Changes based on view mode */}
          <div className="trajectory-selection">
            {viewMode === 'trajectories' ? (
              <>
                {!selectedSubmission ? (
                  <>
                    <h3>Available Submissions</h3>
                    <p className="selection-description">
                      Select a submission to explore its conversation trajectories:
                    </p>
                    
                    {submissionsLoading && (
                      <div className="loading-state">
                        <div className="loading-spinner"></div>
                        <p>Loading submissions...</p>
                      </div>
                    )}
                    
                    {!submissionsLoading && submissions.length === 0 && (
                      <div className="empty-state">
                        <p>No submissions available.</p>
                      </div>
                    )}
                    
                    {!submissionsLoading && submissions.map((submission, index) => (
                      <div 
                        key={`${submission.submissionDir}-${index}`}
                        className={`submission-item ${!submission.hasTrajectories ? 'no-trajectories' : ''}`}
                        onClick={() => submission.hasTrajectories ? loadSubmissionTrajectories(submission) : null}
                      >
                        <div className="submission-info">
                          <div className="submission-title">{submission.model_name}</div>
                          <div className="submission-org">{submission.model_organization}</div>
                          <div className="submission-meta">
                            <span className="submission-date">{submission.submission_date}</span>
                            {submission.is_new && <span className="new-badge">NEW</span>}
                            {!submission.hasTrajectories && <span className="no-trajectories-badge">No Trajectories</span>}
                          </div>
                          {!submission.hasTrajectories && (
                            <div className="no-trajectories-message">
                              No trajectory files available for this submission
                            </div>
                          )}
                          {submission.hasTrajectories && (
                            <div className="has-trajectories-message">
                              Click to view available trajectory files
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </>
                ) : (
                  <>
                    <button 
                      className="back-button"
                      onClick={() => {
                        setSelectedSubmission(null)
                        setAvailableTrajectories([])
                        setSelectedTrajectory(null)
                        setSelectedTask(null)
                      }}
                    >
                      ← Back to Submissions
                    </button>
                    
                    <h3>{selectedSubmission.model_name} Trajectories</h3>
                    <p className="selection-description">
                      {availableTrajectories.length > 0 
                        ? `Found ${availableTrajectories.length} trajectory file${availableTrajectories.length === 1 ? '' : 's'}. Select a domain to explore conversation details:`
                        : 'Loading trajectory information...'
                      }
                    </p>
                    
                    {availableTrajectories.length === 0 && !loading && (
                      <div className="empty-state">
                        <h4>No Trajectories Available</h4>
                        <p>This submission doesn't have any trajectory files for the standard domains (Airline, Retail, Telecom).</p>
                        <p>This could mean:</p>
                        <ul style={{ textAlign: 'left', marginTop: '0.5rem' }}>
                          <li>The evaluation is still in progress</li>
                          <li>The trajectory files use a different naming convention</li>
                          <li>The submission only contains performance results</li>
                        </ul>
                      </div>
                    )}
                    
                    <div className="trajectory-list">
                      {availableTrajectories.map((traj, index) => (
                        <div 
                          key={`${traj.submissionDir}-${traj.file}-${index}`}
                          className={`trajectory-item ${selectedFile === traj.file ? 'selected' : ''}`}
                          onClick={() => loadTrajectoryData(traj)}
                        >
                          <div className="trajectory-info">
                            <div className="trajectory-title">{traj.domain}</div>
                            <div className="trajectory-meta">
                              <span 
                                className="domain-badge" 
                                style={{ backgroundColor: getDomainColor(traj.domain) }}
                              >
                                {traj.domain}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </>
            ) : (
              <>
                <h3>Task Domains</h3>
                <p className="selection-description">
                  Select a domain to explore task definitions and agent policies:
                </p>
                
                <div className="domain-list">
                  {domains.map((domain) => (
                    <div 
                      key={domain.id}
                      className={`domain-item ${selectedDomain === domain.id ? 'selected' : ''}`}
                      onClick={() => loadTaskData(domain.id)}
                    >
                      <div className="domain-info">
                        <div className="domain-title">{domain.name}</div>
                        <div className="domain-meta">
                          <span 
                            className="domain-badge" 
                            style={{ backgroundColor: domain.color }}
                          >
                            {domain.name}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Main Content Panel */}
          <div className="trajectory-content">
            {loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Loading {viewMode === 'trajectories' ? 'trajectory' : 'task'} data...</p>
                <p className="loading-note">Large files may take a moment to load</p>
              </div>
            )}

            {error && (
              <div className="error-state">
                <p>⚠️ {error}</p>
                <p className="error-note">
                  Note: Some files are quite large and may take a moment to load.
                  In a production environment, these would be streamed or paginated for better performance.
                </p>
              </div>
            )}

            {!loading && !error && !selectedTrajectory && !taskData && (
              <div className="empty-state">
                <h3>Select {viewMode === 'trajectories' ? 'a Trajectory' : 'a Domain'}</h3>
                <p>
                  {viewMode === 'trajectories' 
                    ? 'Choose a trajectory from the list to explore detailed conversation flows and agent interactions.'
                    : 'Choose a domain from the list to explore task definitions and agent policies.'
                  }
                </p>
              </div>
            )}

            {/* Trajectory View Content */}
            {viewMode === 'trajectories' && selectedTrajectory && !selectedTask && (
              <div className="task-selection">
                <div className="task-selection-header">
                  <h3>Available Simulations</h3>
                  <button 
                    className="config-button"
                    onClick={() => setShowConfigModal(true)}
                    title="View reproduction configuration"
                  >
                    ⚙️ Configuration
                  </button>
                </div>
                <p>This trajectory contains {selectedTrajectory.simulations?.length || 0} simulations across {selectedTrajectory.tasks?.length || 0} tasks. Select a simulation to view the conversation:</p>
                
                <div className="task-grid">
                  {selectedTrajectory.simulations?.slice(0, 12).map((simulation, index) => {
                    const task = selectedTrajectory.tasks?.find(t => t.id === simulation.task_id) || {}
                    const domain = task.user_scenario?.instructions?.domain || 'Unknown'
                    
                    return (
                      <div 
                        key={simulation.id || index}
                        className="task-card"
                        onClick={() => setSelectedTask(simulation)}
                      >
                        <div className="task-header">
                          <span className="task-id">Task {getCleanTaskId(simulation.task_id)} - Trial {simulation.trial}</span>
                          <span className="task-domain" data-domain={domain}>{domain}</span>
                        </div>
                        <div className="task-description">
                          <p><strong>Purpose:</strong> {task.description?.purpose || 'No description available'}</p>
                          <p><strong>Scenario:</strong> {task.user_scenario?.instructions?.reason_for_call || 'No scenario provided'}</p>
                          <p><strong>Result:</strong> Reward: {simulation.reward_info?.reward?.toFixed(2) || 'N/A'}</p>
                          <p><strong>Termination:</strong> {simulation.termination_reason || 'Unknown'}</p>
                        </div>
                        <div className="task-stats">
                          <span className="message-count">
                            {simulation.messages?.length || 0} messages
                          </span>
                          <span className="duration-count">
                            {simulation.duration ? `${Math.round(simulation.duration)}s` : 'N/A'}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Task View Content */}
            {viewMode === 'tasks' && taskData && !selectedTaskDetail && (
              <div className="task-overview">
                <h3>{taskData.domain.charAt(0).toUpperCase() + taskData.domain.slice(1)} Domain Tasks</h3>
                <p>This domain contains {taskData.tasks?.length || 0} task definitions. Select a task to view its details:</p>
                
                <div className="task-grid">
                  {taskData.tasks?.map((task, index) => (
                    <div 
                      key={task.id || index}
                      className="task-card"
                      onClick={() => setSelectedTaskDetail(task)}
                    >
                      <div className="task-header">
                        <span className="task-id">Task {getCleanTaskId(task.id)}</span>
                        <span className="task-domain" data-domain={taskData.domain}>{taskData.domain}</span>
                      </div>
                      <div className="task-description">
                        <p><strong>Purpose:</strong> {task.description?.purpose || 'No description available'}</p>
                        <p><strong>Scenario:</strong> {task.user_scenario?.instructions?.reason_for_call || 'No scenario provided'}</p>
                        <p><strong>User:</strong> {task.user_scenario?.instructions?.known_info || 'No user info'}</p>
                      </div>
                      <div className="task-stats">
                        <span className="action-count">
                          {task.evaluation_criteria?.actions?.length || 0} expected actions
                        </span>
                        <span className="assertion-count">
                          {task.evaluation_criteria?.nl_assertions?.length || 0} assertions
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Trajectory Conversation View */}
            {viewMode === 'trajectories' && selectedTask && (
              <div className="conversation-view">
                <div className="conversation-header">
                  <div className="conversation-meta">
                    <button 
                      className="back-button"
                      onClick={() => setSelectedTask(null)}
                    >
                      ← Back to Simulations
                    </button>
                    <h3>Task {getCleanTaskId(selectedTask.task_id)} - Trial {selectedTask.trial} Conversation</h3>
                    <div className="conversation-stats">
                      <span>Total Messages: {selectedTask.messages?.length || 0}</span>
                      <span>Showing: {Math.min(60, selectedTask.messages?.length || 0)} messages</span>
                      <span>Duration: {selectedTask.duration ? `${Math.round(selectedTask.duration)}s` : 'N/A'}</span>
                      <span>Result: {selectedTask.reward_info?.reward?.toFixed(2) || 'N/A'}</span>
                    </div>
                  </div>
                  
                  <div className="task-context">
                    <h4>Task Context</h4>
                    {(() => {
                      const task = selectedTrajectory.tasks?.find(t => t.id === selectedTask.task_id) || {}
                      return (
                        <>
                          <p><strong>Purpose:</strong> {task.description?.purpose}</p>
                          <p><strong>User Scenario:</strong> {task.user_scenario?.instructions?.reason_for_call}</p>
                          <p><strong>Known Info:</strong> {task.user_scenario?.instructions?.known_info}</p>
                        </>
                      )
                    })()}
                  </div>
                  
                  <div className="simulation-results">
                    <h4>Simulation Results</h4>
                    <div className="results-grid">
                      <div className="result-item">
                        <span className="result-label">Overall Reward:</span>
                        <span className="result-value">{selectedTask.reward_info?.reward?.toFixed(2) || 'N/A'}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Termination:</span>
                        <span className="result-value">{selectedTask.termination_reason || 'Unknown'}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Agent Cost:</span>
                        <span className="result-value">${selectedTask.agent_cost?.toFixed(4) || 'N/A'}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">User Cost:</span>
                        <span className="result-value">${selectedTask.user_cost?.toFixed(4) || 'N/A'}</span>
                      </div>
                    </div>
                    
                    {selectedTask.reward_info?.nl_assertions && selectedTask.reward_info.nl_assertions.length > 0 && (
                      <div className="assertions">
                        <h5>Evaluation Assertions</h5>
                        <div className="assertion-list">
                          {selectedTask.reward_info.nl_assertions.map((assertion, index) => (
                            <div key={index} className={`assertion ${assertion.met ? 'passed' : 'failed'}`}>
                              <span className="assertion-status">{assertion.met ? '✅' : '❌'}</span>
                              <span className="assertion-text">{assertion.nl_assertion}</span>
                              {assertion.justification && (
                                <p className="assertion-justification">{assertion.justification}</p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="conversation-messages">
                  {getDisplayMessages(selectedTask).map((message, index) => (
                    <div 
                      key={index}
                      className={`message ${message.role}`}
                    >
                      <div className="message-header">
                        <span className="message-role">
                          {message.role === 'assistant' ? '🤖 Agent' : message.role === 'tool' ? '🔧 Tool Output' : '👤 User'}
                        </span>
                        <span className="message-turn">Turn {message.turn}</span>
                        <span className="message-timestamp">{message.timestamp}</span>
                        {message.cost > 0 && (
                          <span className="message-cost">${message.cost.toFixed(4)}</span>
                        )}
                        <span className="message-tokens">{message.tokens} tokens</span>
                      </div>
                      
                      <div className="message-content">
                        {message.content}
                      </div>

                      {message.tool_calls && (
                        <div className="message-tools">
                          <strong>Tool Calls:</strong>
                          <pre>{JSON.stringify(message.tool_calls, null, 2)}</pre>
                        </div>
                      )}
                    </div>
                  ))}
                  
                  {selectedTask.messages?.length > 60 && (
                    <div className="message-truncated">
                      <p>... and {selectedTask.messages.length - 60} more messages</p>
                      <p>Showing first 60 messages for performance. In a full implementation, pagination would be added.</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Task Detail View */}
            {viewMode === 'tasks' && selectedTaskDetail && (
              <div className="task-detail-view">
                <div className="task-detail-header">
                  <button 
                    className="back-button"
                    onClick={() => setSelectedTaskDetail(null)}
                  >
                    ← Back to Tasks
                  </button>
                  <h3>Task {getCleanTaskId(selectedTaskDetail.id)} Details</h3>
                </div>

                <div className="task-detail-content">
                  <div className="task-section">
                    <h4>Task Description</h4>
                    <div className="task-info">
                      <p><strong>Purpose:</strong> {selectedTaskDetail.description?.purpose || 'No description available'}</p>
                      {selectedTaskDetail.description?.notes && (
                        <p><strong>Notes:</strong> {selectedTaskDetail.description.notes}</p>
                      )}
                    </div>
                  </div>

                  <div className="task-section">
                    <h4>User Scenario</h4>
                    <div className="task-info">
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
                    </div>
                  </div>

                  {selectedTaskDetail.evaluation_criteria && (
                    <div className="task-section">
                      <h4>Evaluation Criteria</h4>
                      
                      {selectedTaskDetail.evaluation_criteria.actions && selectedTaskDetail.evaluation_criteria.actions.length > 0 && (
                        <div className="criteria-subsection">
                          <h5>Expected Actions ({selectedTaskDetail.evaluation_criteria.actions.length})</h5>
                          <div className="actions-list">
                            {selectedTaskDetail.evaluation_criteria.actions.map((action, index) => (
                              <div key={index} className="action-item">
                                <p><strong>Action:</strong> {action.name}</p>
                                {action.arguments && (
                                  <pre className="action-args">{JSON.stringify(action.arguments, null, 2)}</pre>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {selectedTaskDetail.evaluation_criteria.nl_assertions && selectedTaskDetail.evaluation_criteria.nl_assertions.length > 0 && (
                        <div className="criteria-subsection">
                          <h5>Natural Language Assertions ({selectedTaskDetail.evaluation_criteria.nl_assertions.length})</h5>
                          <div className="assertions-list">
                            {selectedTaskDetail.evaluation_criteria.nl_assertions.map((assertion, index) => (
                              <div key={index} className="assertion-item">
                                <p>{assertion}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {selectedTaskDetail.evaluation_criteria.env_assertions && selectedTaskDetail.evaluation_criteria.env_assertions.length > 0 && (
                        <div className="criteria-subsection">
                          <h5>Environment Assertions ({selectedTaskDetail.evaluation_criteria.env_assertions.length})</h5>
                          <div className="env-assertions-list">
                            {selectedTaskDetail.evaluation_criteria.env_assertions.map((assertion, index) => (
                              <div key={index} className="env-assertion-item">
                                <p><strong>Function:</strong> {assertion.func_name}</p>
                                <pre className="assertion-args">{JSON.stringify(assertion.arguments, null, 2)}</pre>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {selectedTaskDetail.initial_state && (
                    <div className="task-section">
                      <h4>Initial State</h4>
                      <div className="initial-state">
                        {selectedTaskDetail.initial_state.initialization_actions && (
                          <div>
                            <h5>Initialization Actions</h5>
                            <pre className="initial-actions">{JSON.stringify(selectedTaskDetail.initial_state.initialization_actions, null, 2)}</pre>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {taskData?.policy && (
                    <div className="task-section">
                      <h4>Domain Policy</h4>
                      <div className="policy-content">
                        <pre className="policy-text">{taskData.policy}</pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Configuration Modal */}
        {showConfigModal && selectedTrajectory && (
          <div className="modal-overlay" onClick={handleCloseModal}>
            <div className={`modal-content ${modalClosing ? 'closing' : ''}`} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Testing Configs</h3>
              <button 
                className="modal-close"
                onClick={handleCloseModal}
                title="Close"
              >
                ✕
              </button>
            </div>
              
              <div className="modal-body">
                {selectedTrajectory.info?.agent_info && (
                  <div className="config-section">
                    <h4>🤖 Agent Configuration</h4>
                    <div className="config-details">
                      <div className="config-item">
                        <span className="config-label">Implementation:</span>
                        <span className="config-value">{selectedTrajectory.info.agent_info.implementation}</span>
                      </div>
                      <div className="config-item">
                        <span className="config-label">Model:</span>
                        <span className="config-value">{selectedTrajectory.info.agent_info.llm}</span>
                      </div>
                      {selectedTrajectory.info.agent_info.llm_args && Object.keys(selectedTrajectory.info.agent_info.llm_args).length > 0 && (
                        <div className="config-item">
                          <span className="config-label">LLM Args:</span>
                          <div className="config-args">
                            {Object.entries(selectedTrajectory.info.agent_info.llm_args).map(([key, value]) => (
                              <span key={key} className="arg-item">
                                <code>{key}:</code> <code>{JSON.stringify(value)}</code>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {selectedTrajectory.info?.user_info && (
                  <div className="config-section">
                    <h4>👤 User Simulator Configuration</h4>
                    <div className="config-details">
                      <div className="config-item">
                        <span className="config-label">Implementation:</span>
                        <span className="config-value">{selectedTrajectory.info.user_info.implementation}</span>
                      </div>
                      <div className="config-item">
                        <span className="config-label">Model:</span>
                        <span className="config-value">{selectedTrajectory.info.user_info.llm}</span>
                      </div>
                      {selectedTrajectory.info.user_info.llm_args && Object.keys(selectedTrajectory.info.user_info.llm_args).length > 0 && (
                        <div className="config-item">
                          <span className="config-label">LLM Args:</span>
                          <div className="config-args">
                            {Object.entries(selectedTrajectory.info.user_info.llm_args).map(([key, value]) => (
                              <span key={key} className="arg-item">
                                <code>{key}:</code> <code>{JSON.stringify(value)}</code>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {selectedTrajectory.info && (
                  <div className="config-section">
                    <h4>📊 Evaluation Configuration</h4>
                    <div className="config-details">
                      {selectedTrajectory.info.num_trials && (
                        <div className="config-item">
                          <span className="config-label">Trials:</span>
                          <span className="config-value">{selectedTrajectory.info.num_trials}</span>
                        </div>
                      )}
                      {selectedTrajectory.info.max_steps && (
                        <div className="config-item">
                          <span className="config-label">Max Steps:</span>
                          <span className="config-value">{selectedTrajectory.info.max_steps}</span>
                        </div>
                      )}
                      {selectedTrajectory.info.max_errors && (
                        <div className="config-item">
                          <span className="config-label">Max Errors:</span>
                          <span className="config-value">{selectedTrajectory.info.max_errors}</span>
                        </div>
                      )}
                      {selectedTrajectory.info.seed && (
                        <div className="config-item">
                          <span className="config-label">Seed:</span>
                          <span className="config-value">{selectedTrajectory.info.seed}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
  )
}

export default TrajectoryVisualizer 