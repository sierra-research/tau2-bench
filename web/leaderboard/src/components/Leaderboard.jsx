import { useState, useEffect } from 'react'
import './Leaderboard.css'

const Leaderboard = () => {
  // Chart state for leaderboard
  const [chartInstance, setChartInstance] = useState(null)
  // Add leaderboard view state with localStorage persistence
  const [leaderboardView, setLeaderboardView] = useState(() => {
    return localStorage.getItem('leaderboardView') || 'table'
  })
  // Add unified domain selection state with localStorage persistence
  const [domain, setDomain] = useState(() => {
    return localStorage.getItem('domain') || 'overall'
  })
  // Add sorting state for table with localStorage persistence
  const [sortColumn, setSortColumn] = useState(() => {
    return localStorage.getItem('sortColumn') || 'pass1'
  })
  const [sortDirection, setSortDirection] = useState(() => {
    return localStorage.getItem('sortDirection') || 'desc'
  })
  
  // Add state for dynamically loaded data
  const [passKData, setPassKData] = useState({})
  const [fullSubmissionData, setFullSubmissionData] = useState({}) // Store full submission.json data
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState(null)
  
  // Modal state for submission details
  const [showModal, setShowModal] = useState(false)
  const [selectedSubmission, setSelectedSubmission] = useState(null)
  const [modalClosing, setModalClosing] = useState(false)

  // Function to handle model click and show details
  const handleModelClick = (modelName) => {
    const submissionData = fullSubmissionData[modelName]
    if (submissionData) {
      setSelectedSubmission(submissionData)
      setShowModal(true)
    }
  }

  // Function to close modal with animation
  const closeModal = () => {
    setModalClosing(true)
    setTimeout(() => {
      setShowModal(false)
      setSelectedSubmission(null)
      setModalClosing(false)
    }, 300) // Match the CSS animation duration
  }

  // Function to load submission data from JSON files
  const loadSubmissionData = async () => {
    try {
      setIsLoading(true)
      setLoadError(null)
      
      // Load the manifest file to get list of submissions from new directory structure
      const manifestResponse = await fetch(`${import.meta.env.BASE_URL}submissions/manifest.json`)
      if (!manifestResponse.ok) {
        throw new Error('Failed to load submissions manifest')
      }
      
      const manifest = await manifestResponse.json()
      const submissionDirs = manifest.submissions || []
      
      const loadedData = {}
      const fullSubmissions = {}
      
      // Load each submission from its directory
      for (const submissionDir of submissionDirs) {
        try {
          const response = await fetch(`${import.meta.env.BASE_URL}submissions/${submissionDir}/submission.json`)
          if (!response.ok) {
            console.warn(`Failed to load ${submissionDir}: ${response.status}`)
            continue
          }
          
          const submission = await response.json()
          
          // Store full submission data for modal display
          fullSubmissions[submission.model_name] = {
            ...submission,
            submissionDir // Include directory name for potential trajectory access
          }
          
          // Convert JSON format to internal format
          const retailData = [
            submission.results.retail?.pass_1 || null,
            submission.results.retail?.pass_2 || null,
            submission.results.retail?.pass_3 || null,
            submission.results.retail?.pass_4 || null
          ]
          const airlineData = [
            submission.results.airline?.pass_1 || null,
            submission.results.airline?.pass_2 || null,
            submission.results.airline?.pass_3 || null,
            submission.results.airline?.pass_4 || null
          ]
          const telecomData = [
            submission.results.telecom?.pass_1 || null,
            submission.results.telecom?.pass_2 || null,
            submission.results.telecom?.pass_3 || null,
            submission.results.telecom?.pass_4 || null
          ]
          
          // Calculate overall averages (only if all 3 domains have data)
          const hasRetailData = submission.results.retail?.pass_1 !== null && submission.results.retail?.pass_1 !== undefined
          const hasAirlineData = submission.results.airline?.pass_1 !== null && submission.results.airline?.pass_1 !== undefined
          const hasTelecomData = submission.results.telecom?.pass_1 !== null && submission.results.telecom?.pass_1 !== undefined
          
          const overallData = (hasRetailData && hasAirlineData && hasTelecomData) 
            ? [0, 1, 2, 3].map(passIndex => {
                const values = [retailData[passIndex], airlineData[passIndex], telecomData[passIndex]].filter(val => val !== null)
                return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : null
              })
            : [null, null, null, null] // No overall score if missing any domain
          
          const modelData = {
            retail: retailData,
            airline: airlineData,
            telecom: telecomData,
            overall: overallData,
            // Cost information for each domain
            costs: {
              retail: submission.results.retail?.cost || null,
              airline: submission.results.airline?.cost || null,
              telecom: submission.results.telecom?.cost || null
            },
            isNew: submission.is_new || false,
            organization: submission.submitting_organization,
            userSimulator: submission.methodology?.user_simulator || null,
            // Add verification status
            isVerified: submission.trajectories_available && 
                       submission.methodology?.verification?.modified_prompts === false && 
                       submission.methodology?.verification?.omitted_questions === false,
            verificationDetails: submission.methodology?.verification || null
          }
          
          loadedData[submission.model_name] = modelData
        } catch (error) {
          console.warn(`Error loading ${submissionDir}:`, error)
        }
      }
      
      setPassKData(loadedData)
      setFullSubmissionData(fullSubmissions)
    } catch (error) {
      console.error('Error loading submission data:', error)
      setLoadError(error.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Load data on component mount
  useEffect(() => {
    loadSubmissionData()
  }, [])

  // Initialize chart when leaderboard view is active and chart view is selected
  useEffect(() => {
    if (leaderboardView === 'chart' && !isLoading && Object.keys(passKData).length > 0) {
      // Small delay to ensure DOM is ready
      const timer = setTimeout(() => {
        initializeChart()
      }, 200)
      
      return () => {
        clearTimeout(timer)
      }
    }
  }, [leaderboardView, domain, isLoading, passKData]) // eslint-disable-line react-hooks/exhaustive-deps

  // Save leaderboard state to localStorage
  useEffect(() => {
    localStorage.setItem('leaderboardView', leaderboardView)
  }, [leaderboardView])

  useEffect(() => {
    localStorage.setItem('domain', domain)
  }, [domain])

  useEffect(() => {
    localStorage.setItem('sortColumn', sortColumn)
  }, [sortColumn])

  useEffect(() => {
    localStorage.setItem('sortDirection', sortDirection)
  }, [sortDirection])

  const initializeChart = () => {
    const canvas = document.getElementById('passKChart')
    if (!canvas) return

    // Destroy existing chart if it exists
    if (chartInstance) {
      if (chartInstance.cleanupListeners) {
        chartInstance.cleanupListeners()
      }
      chartInstance.destroy()
    }

    const ctx = canvas.getContext('2d')
    
    const modelColors = {
      'Claude-3.7-Sonnet': '#059669',
      'GPT-4.1': '#3b82f6',
      'o4-mini': '#8b5cf6',
      'GPT-4.1-mini': '#f59e0b',
      'Claude Opus 4.1': '#06b6d4',
      'GPT-5': '#991b1b',
      'Kimi-k2': '#7c3aed',
      'o3': '#1e40af',
      'Claude Opus 4': '#0891b2',
      'Claude Sonnet 4': '#047857',
      'DeepSeek-V3-0324': '#dc2626',
      'Qwen3-235B-A22B': '#ea580c',
      'Gemini-2.5-Flash': '#16a34a'
    }

    const createDatasets = () => {
      const datasets = []

      Object.keys(passKData).forEach(model => {
        const modelData = passKData[model]
        const domainData = modelData[domain]
        
        // Skip models that don't have data for this domain or only have pass^1 data
        if (!domainData || domainData[0] === null || domainData.every((val, index) => index === 0 ? val !== null : val === null)) {
          return
        }
        
        const color = modelColors[model]
        const chartData = domainData.map(val => val === null ? NaN : val)
        
        datasets.push({
          label: `${model}${modelData.isNew ? ' 🆕' : ''}`,
          data: chartData,
          borderColor: color,
          backgroundColor: color + '20',
          fill: false,
          tension: 0.1,
          pointRadius: modelData.isNew ? 8 : 6,
          pointHoverRadius: modelData.isNew ? 10 : 8,
          borderWidth: modelData.isNew ? 4 : 3,
          spanGaps: false
        })
      })

      return datasets
    }

    // Calculate dynamic max value for y-axis
    const calculateMaxValue = (datasets) => {
      let maxValue = 0
      datasets.forEach(dataset => {
        dataset.data.forEach(value => {
          if (!isNaN(value) && value > maxValue) {
            maxValue = value
          }
        })
      })
      
      // Add 10% padding above the max value, with a minimum of 80, but cap at 100
      const paddedMax = Math.max(80, Math.ceil(maxValue * 1.1))
      // Cap at 100 since success rates can't exceed 100%
      const cappedMax = Math.min(100, paddedMax)
      // Round up to nearest 10 for cleaner axis
      return Math.ceil(cappedMax / 10) * 10
    }

    const datasets = createDatasets()
    const dynamicMaxValue = calculateMaxValue(datasets)

    const chart = new window.Chart(ctx, {
      type: 'line',
      data: {
        labels: ['Pass^1', 'Pass^2', 'Pass^3', 'Pass^4'],
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          title: {
            display: true,
            text: 'Pass^k Performance Analysis',
            font: {
              size: 18,
              weight: 'bold'
            },
            padding: 20
          },
          legend: {
            display: true,
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 15,
              font: {
                size: 12
              }
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: 'white',
            bodyColor: 'white',
            borderColor: '#065f46',
            borderWidth: 1,
            callbacks: {
              label: function(context) {
                const drop = context.parsed.y
                const baseline = context.dataset.data[0]
                const dropPercent = ((baseline - drop) / baseline * 100).toFixed(1)
                return `${context.dataset.label}: ${drop.toFixed(1)}% (↓${dropPercent}%)`
              }
            }
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Number of Attempts (k)',
              font: {
                size: 14,
                weight: 'bold'
              }
            },
            grid: {
              color: '#e2e8f0'
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Success Rate (%)',
              font: {
                size: 14,
                weight: 'bold'
              }
            },
            min: 0,
            max: dynamicMaxValue,
            grid: {
              color: '#e2e8f0'
            },
            ticks: {
              callback: function(value) {
                return value + '%'
              }
            }
          }
        },
        elements: {
          line: {
            tension: 0.1
          },
          point: {
            hoverRadius: 8
          }
        }
      }
    })

    setChartInstance(chart)
  }

  // Handle column sorting
  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'desc' ? 'asc' : 'desc')
    } else {
      setSortColumn(column)
      setSortDirection('desc')
    }
  }

  // Clean up chart instance on unmount
  useEffect(() => {
    return () => {
      if (chartInstance) {
        chartInstance.destroy()
      }
    }
  }, [chartInstance])

  // Loading and error states
  if (isLoading) {
    return (
      <div className="leaderboard-container">
        <h2 className="leaderboard-title">τ²-bench Leaderboard</h2>
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <p>Loading leaderboard data...</p>
        </div>
      </div>
    )
  }

  if (loadError) {
    return (
      <div className="leaderboard-container">
        <h2 className="leaderboard-title">τ²-bench Leaderboard</h2>
        <div className="error-state">
          <p>Error loading leaderboard data: {loadError}</p>
          <button onClick={loadSubmissionData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (Object.keys(passKData).length === 0) {
    return (
      <div className="leaderboard-container">
        <h2 className="leaderboard-title">τ²-bench Leaderboard</h2>
        <div className="empty-state">
          <p>No leaderboard data available.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="leaderboard-container">
      <h2 className="leaderboard-title">τ²-bench Leaderboard</h2>

      {/* Combined Controls Row */}
      <div className="leaderboard-controls">
        {/* Modern View Toggle Switch */}
        <div className="view-toggle-switch">
          <div className="toggle-container">
            <button 
              className={`toggle-option ${leaderboardView === 'table' ? 'active' : ''}`}
              onClick={() => setLeaderboardView('table')}
            >
              📋 Table
            </button>
            <button 
              className={`toggle-option ${leaderboardView === 'chart' ? 'active' : ''}`}
              onClick={() => setLeaderboardView('chart')}
            >
              📊 Chart
            </button>
            <div 
              className="toggle-slider"
              style={{
                transform: leaderboardView === 'chart' ? 'translateX(100%)' : 'translateX(0%)'
              }}
            />
          </div>
        </div>

        {/* Domain Toggle Switch */}
        <div className="domain-toggle-switch">
          <div className="toggle-container domain-toggle-container">
            <button 
              className={`toggle-option domain-toggle-option ${domain === 'overall' ? 'active' : ''}`}
              onClick={() => setDomain('overall')}
            >
              📊 Overall
            </button>
            <button 
              className={`toggle-option domain-toggle-option ${domain === 'retail' ? 'active' : ''}`}
              onClick={() => setDomain('retail')}
            >
              🛍️ Retail
            </button>
            <button 
              className={`toggle-option domain-toggle-option ${domain === 'airline' ? 'active' : ''}`}
              onClick={() => setDomain('airline')}
            >
              ✈️ Airline
            </button>
            <button 
              className={`toggle-option domain-toggle-option ${domain === 'telecom' ? 'active' : ''}`}
              onClick={() => setDomain('telecom')}
            >
              📱 Telecom
            </button>
            <div 
              className="toggle-slider domain-toggle-slider"
              style={{
                transform: (() => {
                  if (domain === 'overall') return 'translateX(0%)';
                  if (domain === 'retail') return 'translateX(100%)';
                  if (domain === 'airline') return 'translateX(200%)';
                  if (domain === 'telecom') return 'translateX(300%)';
                  return 'translateX(0%)';
                })()
              }}
            />
          </div>
        </div>
      </div>

      {/* Chart View */}
      {leaderboardView === 'chart' && (
        <div className="reliability-visualization">
          <div className="pass-k-chart-container">
            <canvas id="passKChart" width="800" height="400"></canvas>
          </div>
        </div>
      )}

      {/* Table View */}
      {leaderboardView === 'table' && (
        <div className="reliability-metrics">
        <div className="metrics-table-container">
          <table className="reliability-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                 <th>Submitting Org</th>
                <th>User Sim</th>
                <th 
                  className={`sortable ${sortColumn === 'pass1' ? 'active' : ''}`}
                  onClick={() => handleSort('pass1')}
                >
                  Pass^1 {sortColumn === 'pass1' && (sortDirection === 'desc' ? '↓' : '↑')}
                </th>
                <th 
                  className={`sortable ${sortColumn === 'pass2' ? 'active' : ''}`}
                  onClick={() => handleSort('pass2')}
                >
                  Pass^2 {sortColumn === 'pass2' && (sortDirection === 'desc' ? '↓' : '↑')}
                </th>
                <th 
                  className={`sortable ${sortColumn === 'pass3' ? 'active' : ''}`}
                  onClick={() => handleSort('pass3')}
                >
                  Pass^3 {sortColumn === 'pass3' && (sortDirection === 'desc' ? '↓' : '↑')}
                </th>
                <th 
                  className={`sortable ${sortColumn === 'pass4' ? 'active' : ''}`}
                  onClick={() => handleSort('pass4')}
                >
                  Pass^4 {sortColumn === 'pass4' && (sortDirection === 'desc' ? '↓' : '↑')}
                </th>
                <th>Avg Cost</th>
              </tr>
            </thead>
            <tbody>
              {(() => {
                // Calculate domain-specific scores for ranking
                const modelStats = Object.entries(passKData)
                  .filter(([modelName, data]) => {
                    // For overall domain, only include models that have data for all 3 domains
                    if (domain === 'overall') {
                      return data.overall.some(val => val !== null)
                    }
                    // For individual domains, only include models that have data for that domain
                    return data[domain].some(val => val !== null)
                  })
                  .map(([modelName, data]) => {
                  const domainData = data[domain]
                  const pass1Score = domainData[0]
                  const hasCompleteData = domainData.every(val => val !== null)
                  const hasAnyData = domainData.some(val => val !== null)
                  const consistencyScore = hasCompleteData 
                    ? domainData[3] / domainData[0]
                    : null
                  
                  return {
                    name: modelName,
                    data: data,
                    domainData: domainData,
                    pass1Score,
                    hasCompleteData,
                    hasAnyData,
                    consistencyScore,
                    organization: data.organization
                  }
                })
                
                // Sort by selected column and direction
                modelStats.sort((a, b) => {
                  // First priority: models with any data for this domain
                  if (a.hasAnyData && !b.hasAnyData) return -1
                  if (!a.hasAnyData && b.hasAnyData) return 1
                  if (!a.hasAnyData && !b.hasAnyData) return 0
                  
                  // Second priority: complete data first for pass2-4
                  if (sortColumn !== 'pass1') {
                    if (a.hasCompleteData && !b.hasCompleteData) return -1
                    if (!a.hasCompleteData && b.hasCompleteData) return 1
                  }
                  
                  let aValue, bValue
                  if (sortColumn === 'pass1') {
                    aValue = a.pass1Score
                    bValue = b.pass1Score
                  } else {
                    const passIndex = parseInt(sortColumn.replace('pass', '')) - 1
                    aValue = a.domainData[passIndex]
                    bValue = b.domainData[passIndex]
                    
                    // Handle null values (missing data)
                    if (aValue === null && bValue === null) return 0
                    if (aValue === null) return 1
                    if (bValue === null) return -1
                  }
                  
                  const multiplier = sortDirection === 'desc' ? 1 : -1
                  return (bValue - aValue) * multiplier
                })
                
                return modelStats.map((model, index) => (
                   <tr key={model.name} className={`model-row ${index === 0 && model.hasCompleteData && sortColumn === 'pass1' && sortDirection === 'desc' ? 'top-performer' : ''} ${model.data.isNew ? 'new-model' : ''}`}>
                     {/* Rank */}
                     <td className={`rank-cell ${index === 0 ? 'gold-medal' : index === 1 ? 'silver-medal' : index === 2 ? 'bronze-medal' : ''}`}>
                       {index === 0 && model.hasAnyData ? (
                         <span className="medal-icon">🥇</span>
                       ) : index === 1 && model.hasAnyData ? (
                         <span className="medal-icon">🥈</span>
                       ) : index === 2 && model.hasAnyData ? (
                         <span className="medal-icon">🥉</span>
                       ) : (
                         <span className="rank-number">#{index + 1}</span>
                       )}
                     </td>
                     {/* Model Name */}
                     <td className="model-info">
                       <div 
                         className="model-name clickable-model" 
                         onClick={() => handleModelClick(model.name)}
                         title="Click to view submission details"
                       >
                         {model.name}
                         {model.data.isNew && <span className="new-badge">NEW</span>}
                         {!model.data.isVerified && (
                           <span className="unverified-badge" title="Unverified submission - see details for more information">
                             ⚠️
                           </span>
                         )}
                       </div>
                     </td>
                     
                     {/* Organization */}
                     <td className="organization-info">
                       <div className="org-container">
                         <div className="company-logo">
                          {model.organization === 'Anthropic' && (
                            <img src={`${import.meta.env.BASE_URL}claude.png`} alt="Anthropic" className="logo-img" />
                          )}
                          {model.organization === 'OpenAI' && (
                            <img src={`${import.meta.env.BASE_URL}openai.svg`} alt="OpenAI" className="logo-img" />
                          )}
                          {model.organization === 'Sierra' && (
                            <img src={`${import.meta.env.BASE_URL}sierra-logo.png`} alt="Sierra" className="logo-img" />
                          )}
                          {model.organization === 'Moonshot AI' && (
                            <span className="emoji-logo">🚀</span>
                          )}
                          {model.organization === 'DeepSeek' && (
                            <img src={`${import.meta.env.BASE_URL}DeepSeek_logo_icon.png`} alt="DeepSeek" className="logo-img" />
                          )}
                          {model.organization === 'Alibaba' && (
                            <img src={`${import.meta.env.BASE_URL}qwen-color.png`} alt="Alibaba" className="logo-img" />
                          )}
                          {model.organization === 'Google' && (
                            <img src={`${import.meta.env.BASE_URL}Google__G__logo.svg.png`} alt="Google" className="logo-img" />
                          )}
                         </div>
                         <span className="org-name">{model.organization}</span>
                       </div>
                     </td>
                     
                     {/* User Simulator */}
                     <td className="user-sim-info">
                       {model.data.userSimulator ? (
                         <span className="user-sim-name">{model.data.userSimulator}</span>
                       ) : (
                         <span className="no-data">—</span>
                       )}
                     </td>
                     {/* Pass^1 */}
                     <td className="metric-cell">
                       {model.pass1Score !== null ? (
                         <span className="metric-value">{model.pass1Score.toFixed(1)}%</span>
                       ) : (
                         <span className="no-data">No Data</span>
                       )}
                     </td>
                     {/* Pass^2-4 */}
                     {[1, 2, 3].map(passIndex => {
                       const value = model.domainData[passIndex]
                       
                       return (
                         <td key={passIndex} className="metric-cell">
                           {value !== null ? (
                             <span className="metric-value">{value.toFixed(1)}%</span>
                           ) : model.hasAnyData ? (
                             <span className="missing-data">—</span>
                           ) : (
                             <span className="no-data">No Data</span>
                           )}
                         </td>
                       )
                     })}
                     
                     {/* Average Cost */}
                     <td className="cost-cell">
                       {(() => {
                         if (domain === 'overall') {
                           // Calculate average cost across all three domains
                           const domains = ['retail', 'airline', 'telecom']
                           const costs = domains.map(d => model.data.costs[d]).filter(cost => cost !== null && cost !== undefined)
                           if (costs.length > 0) {
                             const avgCost = costs.reduce((sum, cost) => sum + cost, 0) / costs.length
                             return <span className="cost-value">${avgCost.toFixed(3)}</span>
                           } else {
                             return <span className="no-data">—</span>
                           }
                         } else {
                           const domainCost = model.data.costs[domain]
                           if (domainCost !== null && domainCost !== undefined) {
                             return <span className="cost-value">${domainCost.toFixed(3)}</span>
                           } else {
                             return <span className="no-data">—</span>
                           }
                         }
                       })()}
                     </td>
                  </tr>
                ))
              })()}
            </tbody>
          </table>
        </div>
        <div className="verification-note">
          <span className="note-icon">⚠️</span>
          <span className="note-text">
            The warning icon indicates unverified submissions. Click on any model name to view full verification details.
          </span>
        </div>
        </div>
      )}

      {/* Submissions Notice */}
      <div className="submissions-notice">
        <div className="submissions-content">
          <h3>Submit Your Results</h3>
          <p>
            Have new results to share? Submit your model evaluation results by creating a pull request to add your JSON submission file. 
            See our submission guidelines for the required format and process.
          </p>
          <div className="submission-links">
            <a 
              href="https://github.com/sierra-research/tau2-bench/blob/main/web/leaderboard/SUBMISSION_GUIDE.md" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="submissions-link primary"
            >
              View Submission Guidelines →
            </a>
            <a 
              href="https://github.com/sierra-research/tau2-bench/compare" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="submissions-link secondary"
            >
              Submit via Pull Request →
            </a>
          </div>
        </div>
      </div>

      {/* Submission Details Modal */}
      {showModal && selectedSubmission && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className={`modal-content ${modalClosing ? 'closing' : ''}`} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Submission Details</h2>
              <button className="modal-close" onClick={closeModal}>×</button>
            </div>
            
            <div className="modal-body">
              <div className="submission-details">
                {/* Basic Information */}
                <div className="detail-section">
                  <h3>Basic Information</h3>
                  <div className="detail-grid">
                    <div className="detail-item">
                      <label>Model Name:</label>
                      <span>{selectedSubmission.model_name}</span>
                    </div>
                    <div className="detail-item">
                      <label>Model Organization:</label>
                      <span>{selectedSubmission.model_organization}</span>
                    </div>
                    <div className="detail-item">
                      <label>Submitting Organization:</label>
                      <span>{selectedSubmission.submitting_organization}</span>
                    </div>
                    <div className="detail-item">
                      <label>Submission Date:</label>
                      <span>{selectedSubmission.submission_date}</span>
                    </div>
                    <div className="detail-item">
                      <label>Is New:</label>
                      <span>{selectedSubmission.is_new ? 'Yes' : 'No'}</span>
                    </div>
                  </div>
                </div>

                {/* Contact Information */}
                <div className="detail-section">
                  <h3>Contact Information</h3>
                  <div className="detail-grid">
                    <div className="detail-item">
                      <label>Email:</label>
                      <span>{selectedSubmission.contact_info?.email || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <label>Name:</label>
                      <span>{selectedSubmission.contact_info?.name || 'N/A'}</span>
                    </div>
                    {selectedSubmission.contact_info?.github && (
                      <div className="detail-item">
                        <label>GitHub:</label>
                        <span>{selectedSubmission.contact_info.github}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* References */}
                {selectedSubmission.references && selectedSubmission.references.length > 0 && (
                  <div className="detail-section">
                    <h3>References & Documentation</h3>
                    <div className="references-list">
                      {selectedSubmission.references.map((ref, index) => (
                        <div key={index} className="reference-item">
                          <div className="reference-header">
                            <span className={`reference-type ${ref.type || 'other'}`}>
                              {ref.type === 'paper' && '📄'}
                              {ref.type === 'blog_post' && '📝'}
                              {ref.type === 'documentation' && '📚'}
                              {ref.type === 'model_card' && '🗂️'}
                              {ref.type === 'github' && '🔗'}
                              {ref.type === 'huggingface' && '🤗'}
                              {(!ref.type || ref.type === 'other') && '🌐'}
                              <span className="reference-type-text">
                                {ref.type?.replace('_', ' ').toUpperCase() || 'OTHER'}
                              </span>
                            </span>
                          </div>
                          <a 
                            href={ref.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="reference-link"
                          >
                            {ref.title}
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Performance Results */}
                <div className="detail-section">
                  <h3>Performance Results</h3>
                  {selectedSubmission.results && (
                    <div className="results-tables">
                      {Object.entries(selectedSubmission.results).map(([domain, results]) => (
                        <div key={domain} className="domain-results">
                          <h4>{domain.charAt(0).toUpperCase() + domain.slice(1)} Domain</h4>
                          <div className="results-grid">
                            {[1, 2, 3, 4].map(pass => (
                              <div key={pass} className="result-item">
                                <label>Pass^{pass}:</label>
                                <span>
                                  {results[`pass_${pass}`] !== null && results[`pass_${pass}`] !== undefined 
                                    ? `${results[`pass_${pass}`].toFixed(1)}%` 
                                    : 'N/A'}
                                </span>
                              </div>
                            ))}
                            {results.cost && (
                              <div className="result-item">
                                <label>Cost:</label>
                                <span>${results.cost.toFixed(3)}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Methodology */}
                {selectedSubmission.methodology && (
                  <div className="detail-section">
                    <h3>Methodology</h3>
                    <div className="detail-grid">
                      {selectedSubmission.methodology.evaluation_date && (
                        <div className="detail-item">
                          <label>Evaluation Date:</label>
                          <span>{selectedSubmission.methodology.evaluation_date}</span>
                        </div>
                      )}
                      {selectedSubmission.methodology.tau2_bench_version && (
                        <div className="detail-item">
                          <label>Tau2-Bench Version:</label>
                          <span>{selectedSubmission.methodology.tau2_bench_version}</span>
                        </div>
                      )}
                      {selectedSubmission.methodology.user_simulator && (
                        <div className="detail-item">
                          <label>User Simulator:</label>
                          <span>{selectedSubmission.methodology.user_simulator}</span>
                        </div>
                      )}
                      {selectedSubmission.methodology.notes && (
                        <div className="detail-item full-width">
                          <label>Notes:</label>
                          <p className="notes-text">{selectedSubmission.methodology.notes}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Verification Status */}
                {selectedSubmission.methodology?.verification && (
                  <div className="detail-section">
                    <h3>Verification Status</h3>
                    <div className="verification-status">
                      <div className="verification-indicator">
                        {selectedSubmission.trajectories_available && 
                         selectedSubmission.methodology.verification.modified_prompts === false && 
                         selectedSubmission.methodology.verification.omitted_questions === false ? (
                          <span className="verified">✅ Verified</span>
                        ) : (
                          <span className="unverified">⚠️ Unverified</span>
                        )}
                      </div>
                      <div className="detail-grid">
                        <div className="detail-item">
                          <label>Trajectories Available:</label>
                          <span>{selectedSubmission.trajectories_available ? 'Yes' : 'No'}</span>
                        </div>
                        <div className="detail-item">
                          <label>Modified Prompts:</label>
                          <span>
                            {selectedSubmission.methodology.verification.modified_prompts === true ? 'Yes' : 
                             selectedSubmission.methodology.verification.modified_prompts === false ? 'No' : 'Unknown'}
                          </span>
                        </div>
                        <div className="detail-item">
                          <label>Omitted Questions:</label>
                          <span>
                            {selectedSubmission.methodology.verification.omitted_questions === true ? 'Yes' : 
                             selectedSubmission.methodology.verification.omitted_questions === false ? 'No' : 'Unknown'}
                          </span>
                        </div>
                        {selectedSubmission.methodology.verification.details && (
                          <div className="detail-item full-width">
                            <label>Verification Details:</label>
                            <p className="notes-text">{selectedSubmission.methodology.verification.details}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Leaderboard 