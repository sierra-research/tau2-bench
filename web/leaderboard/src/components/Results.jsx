import React, { useState, useEffect } from 'react'
import './Results.css'

// Data loading functions
const parseCSV = (text) => {
  const lines = text.trim().split('\n')
  const headers = lines[0].split(',')
  const data = []
  
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim()) {
      const values = lines[i].split(',')
      const row = {}
      headers.forEach((header, index) => {
        row[header.trim()] = values[index]?.trim() || ''
      })
      data.push(row)
    }
  }
  
  return { headers, data }
}

const parseCostInfo = (text) => {
  const lines = text.split('\n')
  const meanCostStart = lines.findIndex(line => line.includes('Mean cost per LLM:'))
  const sumCostStart = lines.findIndex(line => line.includes('Sum cost per LLM:'))
  
  const meanCosts = []
  const sumCosts = []
  
  // Parse mean costs (lines after meanCostStart + 2)
  for (let i = meanCostStart + 3; i < sumCostStart - 1; i++) {
    if (lines[i].trim()) {
      const parts = lines[i].split(/\s+/)
      if (parts.length >= 3) {
        meanCosts.push({
          model: parts[0],
          agentCost: parseFloat(parts[1]),
          userCost: parseFloat(parts[2])
        })
      }
    }
  }
  
  // Parse sum costs (lines after sumCostStart + 2)
  for (let i = sumCostStart + 3; i < lines.length; i++) {
    if (lines[i].trim()) {
      const parts = lines[i].split(/\s+/)
      if (parts.length >= 3) {
        sumCosts.push({
          model: parts[0],
          agentCost: parseFloat(parts[1]),
          userCost: parseFloat(parts[2])
        })
      }
    }
  }
  
  return { meanCosts, sumCosts }
}

const Results = () => {
  const [actionSuccessData, setActionSuccessData] = useState(null)
  const [workflowSuccessData, setWorkflowSuccessData] = useState(null)
  const [costData, setCostData] = useState(null)
  const [loading, setLoading] = useState(true)

  // Load data from files
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        
        // Load action success rates data
        const [telecomResponse, workflowResponse, costResponse] = await Promise.all([
          fetch(`${import.meta.env.BASE_URL}data/action_success_rates_telecom.csv`),
          fetch(`${import.meta.env.BASE_URL}data/action_success_rates_telecom-workflow.csv`),
          fetch(`${import.meta.env.BASE_URL}data/cost_info.txt`)
        ])
        
        const telecomText = await telecomResponse.text()
        const workflowText = await workflowResponse.text()
        const costText = await costResponse.text()
        
        setActionSuccessData(parseCSV(telecomText))
        setWorkflowSuccessData(parseCSV(workflowText))
        setCostData(parseCostInfo(costText))
        
      } catch (error) {
        console.error('Error loading data:', error)
      } finally {
        setLoading(false)
      }
    }
    
    loadData()
  }, [])

  // Key Contributions Component
  const KeyContributionsSection = () => (
    <div className="contributions-section">
      <h2>Key Contributions</h2>
      <p className="results-subtitle" style={{textAlign: 'center', marginBottom: '32px'}}>
        τ²-bench introduces four fundamental advances in agent evaluation methodology
      </p>
      <div className="contributions-grid">
        <div className="contribution-card">
          <div className="contribution-number">1</div>
          <h3>Telecom Dual-Control Domain</h3>
          <p>
            Novel domain modeled as a Dec-POMDP where both agent and user possess distinct tools to observe, 
            act upon, and verify the state of a shared, dynamic environment. This exposes crucial coordination 
            and communication challenges absent from single-control evaluations.
          </p>
        </div>
        <div className="contribution-card">
          <div className="contribution-number">2</div>
          <h3>Compositional Task Generator</h3>
          <p>
            Programmatically creates diverse, verifiable tasks from atomic components, ensuring domain coverage 
            and controlled complexity. Enables systematic scaling from simple to complex multi-stage problems.
          </p>
        </div>
        <div className="contribution-card">
          <div className="contribution-number">3</div>
          <h3>Reliable User Simulator</h3>
          <p>
            Tightly coupled with the environment, with behavior constrained by tools and observable states. 
            Improves simulation fidelity and reduces error rates compared to unconstrained natural language approaches.
          </p>
        </div>
        <div className="contribution-card">
          <div className="contribution-number">4</div>
          <h3>Fine-Grained Performance Analysis</h3>
          <p>
            Multiple ablations separating errors arising from reasoning vs communication/coordination. 
            Enables precise identification of agent failure modes in collaborative environments.
          </p>
        </div>
      </div>
    </div>
  )

  // Cross-Domain Performance Component
  const CrossDomainPerformanceSection = () => (
    <div className="cross-domain-section">
      <h2>Cross-Domain Performance Analysis</h2>
      <p className="results-subtitle" style={{textAlign: 'center', marginBottom: '32px'}}>
        Telecom domain presents significantly greater challenges than existing benchmarks, 
        revealing the complexity of dual-control collaborative environments
      </p>
      <div className="domain-performance-container">
        <div className="performance-comparison-table">
          <table className="performance-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Retail</th>
                <th>Airline</th>
                <th>Telecom</th>
                <th>Drop from Best</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="model-name">GPT-4.1</td>
                <td className="metric-cell">
                  <span className="metric-value">74.0%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '74%', backgroundColor: '#10b981'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">56.0%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '56%', backgroundColor: '#f59e0b'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">34.0%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '34%', backgroundColor: '#ef4444'}}></div>
                  </div>
                </td>
                <td className="drop-metric">-40.0%</td>
              </tr>
              <tr>
                <td className="model-name">Claude-3.7-Sonnet</td>
                <td className="metric-cell">
                  <span className="metric-value">72.1%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '72.1%', backgroundColor: '#10b981'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">64.2%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '64.2%', backgroundColor: '#f59e0b'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">49.0%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '49%', backgroundColor: '#ef4444'}}></div>
                  </div>
                </td>
                <td className="drop-metric">-23.1%</td>
              </tr>
              <tr>
                <td className="model-name">o4-mini</td>
                <td className="metric-cell">
                  <span className="metric-value">68.3%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '68.3%', backgroundColor: '#10b981'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">52.1%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '52.1%', backgroundColor: '#f59e0b'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">50.2%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '50.2%', backgroundColor: '#ef4444'}}></div>
                  </div>
                </td>
                <td className="drop-metric">-18.1%</td>
              </tr>
              <tr>
                <td className="model-name">GPT-4.1-mini</td>
                <td className="metric-cell">
                  <span className="metric-value">61.4%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '61.4%', backgroundColor: '#10b981'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">48.7%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '48.7%', backgroundColor: '#f59e0b'}}></div>
                  </div>
                </td>
                <td className="metric-cell">
                  <span className="metric-value">48.9%</span>
                  <div className="metric-bar">
                    <div className="metric-progress" style={{width: '48.9%', backgroundColor: '#ef4444'}}></div>
                  </div>
                </td>
                <td className="drop-metric">-12.5%</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="performance-insights">
          <div className="insight-card">
            <h3>Dual-Control Challenge</h3>
            <p>
              The telecom domain's dual-control setup where both agent and user have tools creates 
              unprecedented coordination challenges. Even state-of-the-art models like GPT-4.1 
              show a dramatic 40% performance drop compared to single-control retail tasks.
            </p>
          </div>
          <div className="insight-card">
            <h3>Consistency Patterns</h3>
            <p>
              While Claude-3.7-Sonnet performs comparably to airline domain initially, its pass@k 
              scores decline more rapidly with increased attempts, suggesting less consistent 
              performance under collaborative pressure.
            </p>
          </div>
        </div>
      </div>
    </div>
  )

  // Issue Type Performance Component
  const IssueTypePerformanceSection = () => (
    <div className="issue-performance-section">
      <h2>Performance by Issue Type</h2>
      <p className="results-subtitle" style={{textAlign: 'center', marginBottom: '32px'}}>
        Multi-stage reasoning and conditional logic in complex issue types pose substantial challenges to all models
      </p>
      <div className="issue-performance-grid">
        <div className="issue-type-card easy">
          <h3>Service Issues</h3>
          <div className="issue-stats">
            <div className="stat-item">
              <span className="stat-label">Mean Actions</span>
              <span className="stat-value">2.31</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Difficulty</span>
              <span className="difficulty-badge easy">Easy</span>
            </div>
          </div>
          <div className="model-performance">
            <div className="model-perf-item">
              <span className="model-name">GPT-4.1</span>
              <span className="perf-value">~65%</span>
            </div>
            <div className="model-perf-item">
              <span className="model-name">Claude-3.7</span>
              <span className="perf-value">~70%</span>
            </div>
            <div className="model-perf-item">
              <span className="model-name">o4-mini</span>
              <span className="perf-value">~60%</span>
            </div>
          </div>
          <p className="issue-description">
            Straightforward sequence of actions that can typically be resolved independently 
            without complex dependencies or multi-stage reasoning.
          </p>
        </div>
        
        <div className="issue-type-card medium">
          <h3>Mobile Data Issues</h3>
          <div className="issue-stats">
            <div className="stat-item">
              <span className="stat-label">Mean Actions</span>
              <span className="stat-value">4.31</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Difficulty</span>
              <span className="difficulty-badge medium">Medium</span>
            </div>
          </div>
          <div className="model-performance">
            <div className="model-perf-item">
              <span className="model-name">GPT-4.1</span>
              <span className="perf-value">~35%</span>
            </div>
            <div className="model-perf-item">
              <span className="model-name">Claude-3.7</span>
              <span className="perf-value">~45%</span>
            </div>
            <div className="model-perf-item">
              <span className="model-name">o4-mini</span>
              <span className="perf-value">~40%</span>
            </div>
          </div>
          <p className="issue-description">
            Often requires first checking for and potentially resolving underlying service issues, 
            creating dependencies and multi-stage problem-solving requirements.
          </p>
        </div>
        
        <div className="issue-type-card hard">
          <h3>MMS Issues</h3>
          <div className="issue-stats">
            <div className="stat-item">
              <span className="stat-label">Mean Actions</span>
              <span className="stat-value">6.00</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Difficulty</span>
              <span className="difficulty-badge hard">Hard</span>
            </div>
          </div>
          <div className="model-performance">
            <div className="model-perf-item">
              <span className="model-name">GPT-4.1</span>
              <span className="perf-value">~25%</span>
            </div>
            <div className="model-perf-item">
              <span className="model-name">Claude-3.7</span>
              <span className="perf-value">~35%</span>
            </div>
            <div className="model-perf-item">
              <span className="model-name">o4-mini</span>
              <span className="perf-value">~30%</span>
            </div>
          </div>
          <p className="issue-description">
            Most complex multi-stage problems requiring comprehensive diagnosis of underlying 
            service and data issues before addressing MMS-specific configuration problems.
          </p>
        </div>
      </div>
    </div>
  )

  // User Persona Analysis Component
  const UserPersonaAnalysisSection = () => (
    <div className="persona-analysis-section">
      <h2>User Persona Impact Analysis</h2>
      <p className="results-subtitle" style={{textAlign: 'center', marginBottom: '32px'}}>
        User characteristics significantly affect agent performance, highlighting the importance of persona-aware evaluation
      </p>
      <div className="persona-performance-container">
        <div className="persona-cards">
          <div className="persona-card easy-persona">
            <h3>Easy Persona</h3>
            <div className="persona-description">
              <p>Cooperative, follows instructions well, provides clear information</p>
            </div>
            <div className="persona-metrics">
              <div className="persona-metric">
                <span className="model-name">GPT-4.1</span>
                <span className="metric-value">42%</span>
                <span className="performance-indicator good">Best</span>
              </div>
              <div className="persona-metric">
                <span className="model-name">Claude-3.7</span>
                <span className="metric-value">55%</span>
                <span className="performance-indicator good">Best</span>
              </div>
              <div className="persona-metric">
                <span className="model-name">o4-mini</span>
                <span className="metric-value">58%</span>
                <span className="performance-indicator good">Best</span>
              </div>
            </div>
          </div>
          
          <div className="persona-card none-persona">
            <h3>No Persona</h3>
            <div className="persona-description">
              <p>Default user behavior without specific personality traits</p>
            </div>
            <div className="persona-metrics">
              <div className="persona-metric">
                <span className="model-name">GPT-4.1</span>
                <span className="metric-value">32%</span>
                <span className="performance-indicator moderate">Moderate</span>
              </div>
              <div className="persona-metric">
                <span className="model-name">Claude-3.7</span>
                <span className="metric-value">47%</span>
                <span className="performance-indicator moderate">Moderate</span>
              </div>
              <div className="persona-metric">
                <span className="model-name">o4-mini</span>
                <span className="metric-value">45%</span>
                <span className="performance-indicator moderate">Moderate</span>
              </div>
            </div>
          </div>
          
          <div className="persona-card hard-persona">
            <h3>Hard Persona</h3>
            <div className="persona-description">
              <p>Less cooperative, may provide unclear information, more challenging to guide</p>
            </div>
            <div className="persona-metrics">
              <div className="persona-metric">
                <span className="model-name">GPT-4.1</span>
                <span className="metric-value">28%</span>
                <span className="performance-indicator poor">Worst</span>
              </div>
              <div className="persona-metric">
                <span className="model-name">Claude-3.7</span>
                <span className="metric-value">43%</span>
                <span className="performance-indicator poor">Worst</span>
              </div>
              <div className="persona-metric">
                <span className="model-name">o4-mini</span>
                <span className="metric-value">42%</span>
                <span className="performance-indicator poor">Worst</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="persona-insights">
          <div className="insight-card">
            <h3>Persona Impact</h3>
            <p>
              All models perform significantly better with Easy personas compared to Hard personas, 
              with performance gaps of 10-15%. Interestingly, performance with no persona information 
              often matches or falls below Hard persona performance.
            </p>
          </div>
          <div className="insight-card">
            <h3>Deployment Implications</h3>
            <p>
              These results highlight the critical importance of testing AI systems with well-defined 
              user personas before real-world deployment, as user characteristics substantially impact success rates.
            </p>
          </div>
        </div>
      </div>
    </div>
  )

  // Action Success Rate Visualization Component
  const ActionSuccessRatesSection = ({ data, title }) => {
    if (!data || !data.data.length) return null

    // Extract model columns (exclude action_name and frequency)
    const modelColumns = data.headers.filter(header => 
      header !== 'action_name' && header !== 'frequency' && header !== ''
    )

    // Get top actions by frequency for better visualization
    const topActions = data.data
      .filter(row => row.frequency && parseFloat(row.frequency) > 0.03) // Show actions with >3% frequency
      .sort((a, b) => parseFloat(b.frequency) - parseFloat(a.frequency))
      .slice(0, 10)

    return (
      <div className="action-success-section">
        <h3>{title}</h3>
        <div className="action-success-table-container">
          <table className="action-success-table">
            <thead>
              <tr>
                <th>Action</th>
                <th>Frequency</th>
                {modelColumns.map(model => (
                  <th key={model} className="model-column">
                    {model.replace(/-2025-04-14|-20250219|-2025-04-16/g, '').replace(/-/g, ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {topActions.map((row, index) => (
                <tr key={row.action_name || index}>
                  <td className="action-name">{row.action_name?.replace(/_/g, ' ')}</td>
                  <td className="frequency-cell">
                    {row.frequency ? `${(parseFloat(row.frequency) * 100).toFixed(1)}%` : '-'}
                  </td>
                  {modelColumns.map(model => {
                    const value = parseFloat(row[model])
                    return (
                      <td key={model} className="success-rate-cell">
                        <div className="success-rate-container">
                          <span className="success-rate-value">
                            {!isNaN(value) ? `${(value * 100).toFixed(1)}%` : '-'}
                          </span>
                          {!isNaN(value) && (
                            <div className="success-rate-bar">
                              <div 
                                className="success-rate-progress" 
                                style={{
                                  width: `${value * 100}%`,
                                  backgroundColor: value > 0.8 ? '#10b981' : value > 0.6 ? '#f59e0b' : '#ef4444'
                                }}
                              ></div>
                            </div>
                          )}
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  // Cost Analysis Component
  const CostAnalysisSection = ({ data }) => {
    if (!data || !data.meanCosts.length) return null

    const modelDisplayNames = {
      'claude-3-7-sonnet-20250219': 'Claude-3.7 Sonnet',
      'gpt-4.1-2025-04-14': 'GPT-4.1',
      'gpt-4.1-mini-2025-04-14': 'GPT-4.1 Mini',
      'o4-mini-2025-04-16': 'O4 Mini'
    }

    return (
      <div className="cost-analysis-section">
        <h3>Cost Analysis</h3>
        <div className="cost-analysis-grid">
          <div className="cost-table-container">
            <h4>Mean Cost per Task</h4>
            <table className="cost-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Agent Cost</th>
                  <th>User Cost</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {data.meanCosts.map((cost) => (
                  <tr key={cost.model}>
                    <td>{modelDisplayNames[cost.model] || cost.model}</td>
                    <td>${cost.agentCost.toFixed(4)}</td>
                    <td>${cost.userCost.toFixed(4)}</td>
                    <td className="total-cost">${(cost.agentCost + cost.userCost).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="cost-table-container">
            <h4>Total Cost (All Tasks)</h4>
            <table className="cost-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Agent Cost</th>
                  <th>User Cost</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {data.sumCosts.map((cost) => (
                  <tr key={cost.model}>
                    <td>{modelDisplayNames[cost.model] || cost.model}</td>
                    <td>${cost.agentCost.toFixed(2)}</td>
                    <td>${cost.userCost.toFixed(2)}</td>
                    <td className="total-cost">${(cost.agentCost + cost.userCost).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="results-page">
      <div className="container">
        {/* Header */}
        <div className="results-header">
          <h1>τ²-Bench Research Analysis</h1>
          <p className="results-subtitle">
            Comprehensive evaluation of conversational agents in dual-control collaborative environments.
            Detailed analysis of coordination challenges, reasoning bottlenecks, and simulation quality.
          </p>
        </div>

        {loading && (
          <div className="loading-section">
            <p>Loading data...</p>
          </div>
        )}

        {/* Key Contributions */}
        <KeyContributionsSection />

        {/* Cross-Domain Performance */}
        <CrossDomainPerformanceSection />

        {/* Control Mode Comparison */}
        <div className="performance-section">
          <h2>Ablation Analysis: Reasoning vs Coordination</h2>
          <p className="results-subtitle" style={{textAlign: 'center', marginBottom: '32px'}}>
            Separating the impact of reasoning load from communication and coordination challenges
          </p>
          <div className="control-modes-container">
            <div className="control-mode-cards">
              <div className="control-mode-card">
                <h3>Default Mode</h3>
                <p>Agent and user collaborate in dual-control setup with shared environment access</p>
                <div className="mode-metrics">
                  <div className="mode-metric">
                    <span className="model-name">GPT-4.1</span>
                    <span className="metric-value">34.2%</span>
                  </div>
                  <div className="mode-metric">
                    <span className="model-name">O4-Mini</span>
                    <span className="metric-value">39.8%</span>
                  </div>
                </div>
              </div>
              <div className="control-mode-card">
                <h3>No-User Mode</h3>
                <p>Agent controls all tools independently, eliminating coordination overhead</p>
                <div className="mode-metrics">
                  <div className="mode-metric">
                    <span className="model-name">GPT-4.1</span>
                    <span className="metric-value">52.3%</span>
                    <span className="improvement">+18.1%</span>
                  </div>
                  <div className="mode-metric">
                    <span className="model-name">O4-Mini</span>
                    <span className="metric-value">64.8%</span>
                    <span className="improvement">+25.0%</span>
                  </div>
                </div>
              </div>
              <div className="control-mode-card">
                <h3>Oracle Mode</h3>
                <p>Agent provided with ground-truth action sequence, testing collaboration skills</p>
                <div className="mode-metrics">
                  <div className="mode-metric">
                    <span className="model-name">GPT-4.1</span>
                    <span className="metric-value">56.8%</span>
                    <span className="improvement">+22.6%</span>
                  </div>
                  <div className="mode-metric">
                    <span className="model-name">O4-Mini</span>
                    <span className="metric-value">72.7%</span>
                    <span className="improvement">+32.9%</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="ablation-insights">
              <div className="insight-card">
                <h3>Coordination Overhead</h3>
                <p>
                  The substantial performance drop (18-25%) from No-User to Default mode reveals that 
                  coordination and communication challenges are major bottlenecks beyond pure reasoning capabilities.
                </p>
              </div>
              <div className="insight-card">
                <h3>Reasoning vs Execution</h3>
                <p>
                  Oracle mode performance shows that o4-mini better utilizes ground-truth information than GPT-4.1, 
                  suggesting different strengths in plan execution vs generation.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Issue Type Performance */}
        <IssueTypePerformanceSection />

        {/* User Persona Analysis */}
        <UserPersonaAnalysisSection />

        {/* Task Complexity Analysis */}
        <div className="performance-section">
          <h2>Task Complexity Scaling</h2>
          <div className="complexity-analysis">
            <div className="complexity-section">
              <h3>Performance vs Task Length</h3>
              <div className="performance-insight">
                <p>
                  Agent performance drops dramatically as task complexity increases, regardless of coordination mode:
                </p>
                <ul>
                  <li><strong>1-3 actions:</strong> ~60-80% success rate</li>
                  <li><strong>4-6 actions:</strong> ~40-60% success rate</li>
                  <li><strong>7+ actions:</strong> Near-zero success rate</li>
                </ul>
                <p>
                  This pattern holds across both default and no-user modes, suggesting that maintaining 
                  reliability over longer-horizon tasks remains challenging regardless of user coordination overhead.
                  The gap between modes reduces as task length increases, indicating that reasoning complexity 
                  becomes the dominant bottleneck for very long tasks.
                </p>
              </div>
            </div>
            
            <div className="complexity-section">
              <h3>Multi-Task Scaling</h3>
              <div className="performance-insight">
                <p>
                  Performance also trends downward as the number of distinct sub-tasks increases within a single task:
                </p>
                <ul>
                  <li><strong>Single issue:</strong> Highest success rates</li>
                  <li><strong>Multiple issues:</strong> Requires complex conditional reasoning</li>
                  <li><strong>Transfer cases:</strong> Special handling for unsolvable problems</li>
                </ul>
                <p>
                  This validates that our domain design provides natural paths to scaling complexity through 
                  both increasing action length and combining different sub-tasks into unified problems.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Action Success Rates Section */}
        {!loading && (actionSuccessData || workflowSuccessData) && (
          <div className="detailed-analysis-section">
            <h2>Detailed Action-Level Analysis</h2>
                         {actionSuccessData && (
               <ActionSuccessRatesSection 
                 data={actionSuccessData} 
                 title="Telecom Action Success Rates"
               />
             )}
             {workflowSuccessData && (
               <ActionSuccessRatesSection 
                 data={workflowSuccessData} 
                 title="Telecom Workflow Action Success Rates"
               />
             )}
          </div>
        )}

        {/* User Simulator Quality */}
        <div className="performance-section">
          <h2>User Simulator Quality Assessment</h2>
          <p className="results-subtitle" style={{textAlign: 'center', marginBottom: '32px'}}>
            Dual-control design significantly improves simulation reliability through environmental constraints
          </p>
          <div className="simulator-quality-container">
            <table className="performance-table">
              <thead>
                <tr>
                  <th>Domain</th>
                  <th>Conversations</th>
                  <th>Critical Errors</th>
                  <th>Benign Errors</th>
                  <th>Total Error Rate</th>
                  <th>Quality Assessment</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><span className="domain-name">Airline</span></td>
                  <td>100</td>
                  <td>13 (13%)</td>
                  <td>34 (34%)</td>
                  <td>47%</td>
                  <td><span className="quality-badge moderate">Moderate</span></td>
                </tr>
                <tr>
                  <td><span className="domain-name">Retail</span></td>
                  <td>50</td>
                  <td>6 (12%)</td>
                  <td>14 (28%)</td>
                  <td>40%</td>
                  <td><span className="quality-badge good">Good</span></td>
                </tr>
                <tr className="top-performer">
                  <td><span className="domain-name">Telecom</span></td>
                  <td>50</td>
                  <td>3 (6%)</td>
                  <td>5 (10%)</td>
                  <td>16%</td>
                  <td><span className="quality-badge excellent">Excellent</span></td>
                </tr>
              </tbody>
            </table>
            <div className="simulator-insights">
              <div className="insight-card">
                <h3>Environmental Constraints</h3>
                <p>
                  The telecom domain's structured interface and clear action space naturally guide the user simulator 
                  toward correct interactions, resulting in a 67% reduction in error rates compared to unconstrained approaches.
                </p>
              </div>
              <div className="insight-card">
                <h3>Tool-Mediated Behavior</h3>
                <p>
                  Rather than relying heavily on natural language specifications, the dual-control design shapes and 
                  tightly constrains user behavior through environmental affordances, leading to more predictable and reliable simulations.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Domain Statistics */}
        <div className="performance-section">
          <h2>Benchmark Statistics</h2>
          <div className="domain-stats-grid">
            <div className="domain-stat-card">
              <h3>Retail Domain</h3>
              <div className="stat-details">
                <div className="stat-row">
                  <span>Database:</span>
                  <span>500 users, 50 products, 1,000 orders</span>
                </div>
                <div className="stat-row">
                  <span>Agent Tools:</span>
                  <span>7 write, 6 read</span>
                </div>
                <div className="stat-row">
                  <span>User Tools:</span>
                  <span>None (single-control)</span>
                </div>
                <div className="stat-row">
                  <span>Tasks:</span>
                  <span>115</span>
                </div>
              </div>
            </div>
            
            <div className="domain-stat-card">
              <h3>Airline Domain</h3>
              <div className="stat-details">
                <div className="stat-row">
                  <span>Database:</span>
                  <span>500 users, 300 flights, 2,000 reservations</span>
                </div>
                <div className="stat-row">
                  <span>Agent Tools:</span>
                  <span>6 write, 6 read</span>
                </div>
                <div className="stat-row">
                  <span>User Tools:</span>
                  <span>None (single-control)</span>
                </div>
                <div className="stat-row">
                  <span>Tasks:</span>
                  <span>50</span>
                </div>
              </div>
            </div>
            
            <div className="domain-stat-card telecom-highlight">
              <h3>Telecom Domain</h3>
              <div className="stat-details">
                <div className="stat-row">
                  <span>Database:</span>
                  <span>5 plans, 9 lines, 4 customers</span>
                </div>
                <div className="stat-row">
                  <span>Agent Tools:</span>
                  <span>6 write, 7 read</span>
                </div>
                <div className="stat-row">
                  <span>User Tools:</span>
                  <span>15 write, 15 read (dual-control)</span>
                </div>
                <div className="stat-row">
                  <span>Tasks:</span>
                  <span>114 (full: 2,285)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Cost Analysis Section */}
        {!loading && costData && (
          <div className="cost-section">
            <h2>Cost Analysis</h2>
            <CostAnalysisSection data={costData} />
          </div>
        )}

      </div>
    </div>
  )
}

export default Results 