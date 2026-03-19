import React, { useState, useEffect } from 'react'
import './Leaderboard.css'

const BENCHMARK_VALUES = new Set(['text', 'voice'])

const getBenchmarkFromHash = () => {
  const hash = window.location.hash.slice(1)
  const [route, queryString = ''] = hash.split('?')
  if (route !== 'leaderboard') return null

  const value = new URLSearchParams(queryString).get('benchmark')
  return BENCHMARK_VALUES.has(value) ? value : null
}

const Leaderboard = () => {
  // Benchmark selector: 'text' (τ-bench) or 'voice' (τ-voice)
  const [benchmark, setBenchmark] = useState(() => {
    const fromHash = getBenchmarkFromHash()
    if (fromHash) return fromHash

    const fromStorage = localStorage.getItem('benchmark')
    return BENCHMARK_VALUES.has(fromStorage) ? fromStorage : 'text'
  })
  // Add unified domain selection state with localStorage persistence
  const [domain, setDomain] = useState(() => {
    return localStorage.getItem('domain') || 'overall'
  })
  // Selected pass^k metric (1-4) with localStorage persistence
  const [selectedPassK, setSelectedPassK] = useState(() => {
    const stored = localStorage.getItem('selectedPassK')
    return stored ? parseInt(stored) : 1
  })
  const [sortDirection, setSortDirection] = useState(() => {
    return localStorage.getItem('sortDirection') || 'desc'
  })
  // Add submission type filter state (standard vs custom)
  const [showStandard, setShowStandard] = useState(() => {
    const stored = localStorage.getItem('showStandard')
    return stored === null ? true : stored === 'true'
  })
  const [showCustom, setShowCustom] = useState(() => {
    const stored = localStorage.getItem('showCustom')
    return stored === null ? false : stored === 'true'
  })
  // Legacy submissions toggle
  const [showLegacy, setShowLegacy] = useState(() => {
    return localStorage.getItem('showLegacy') === 'true'
  })
  // Info tooltip state
  const [showFilterInfo, setShowFilterInfo] = useState(false)
  // Expanded rows state (set of model names)
  const [expandedRows, setExpandedRows] = useState(new Set())
  
  // Add state for dynamically loaded data
  const [passKData, setPassKData] = useState({})
  const [fullSubmissionData, setFullSubmissionData] = useState({}) // Store full submission.json data
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState(null)
  
  // Modal state for submission details
  const [showModal, setShowModal] = useState(false)
  const [selectedSubmission, setSelectedSubmission] = useState(null)
  const [modalClosing, setModalClosing] = useState(false)

  // Function to handle model click and show details (keyed by submissionDir)
  const handleModelClick = (submissionKey) => {
    const submissionData = fullSubmissionData[submissionKey]
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
    }, 200) // Match the CSS animation duration (0.2s)
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
      const currentDirs = manifest.submissions || []
      const legacyDirs = manifest.legacy_submissions || []
      const voiceDirs = manifest.voice_submissions || []
      
      const loadedData = {}
      const fullSubmissions = {}
      
      // Helper to load a submission directory
      const loadSubmission = async (submissionDir, isLegacy, modality = 'text') => {
        try {
          const response = await fetch(`${import.meta.env.BASE_URL}submissions/${submissionDir}/submission.json`)
          if (!response.ok) {
            console.warn(`Failed to load ${submissionDir}: ${response.status}`)
            return
          }
          
          const submission = await response.json()
          
          // Store full submission data for modal display (keyed by submissionDir to avoid collisions)
          fullSubmissions[submissionDir] = {
            ...submission,
            submissionDir,
            isLegacy,
            modality
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
          const bankingData = [
            submission.results.banking_knowledge?.pass_1 || null,
            submission.results.banking_knowledge?.pass_2 || null,
            submission.results.banking_knowledge?.pass_3 || null,
            submission.results.banking_knowledge?.pass_4 || null
          ]
          
          // Calculate overall averages (only if all 4 domains have data)
          const hasRetailData = submission.results.retail?.pass_1 !== null && submission.results.retail?.pass_1 !== undefined
          const hasAirlineData = submission.results.airline?.pass_1 !== null && submission.results.airline?.pass_1 !== undefined
          const hasTelecomData = submission.results.telecom?.pass_1 !== null && submission.results.telecom?.pass_1 !== undefined
          const hasBankingData = submission.results.banking_knowledge?.pass_1 !== null && submission.results.banking_knowledge?.pass_1 !== undefined
          
          const overallData = (hasRetailData && hasAirlineData && hasTelecomData && hasBankingData) 
            ? [0, 1, 2, 3].map(passIndex => {
                const values = [retailData[passIndex], airlineData[passIndex], telecomData[passIndex], bankingData[passIndex]].filter(val => val !== null)
                return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : null
              })
            : [null, null, null, null] // No overall score if missing any domain
          
          const modelData = {
            modelName: submission.model_name,
            submissionDir,
            modality,
            retail: retailData,
            airline: airlineData,
            telecom: telecomData,
            banking_knowledge: bankingData,
            overall: overallData,
            // Cost information for each domain
            costs: {
              retail: submission.results.retail?.cost || null,
              airline: submission.results.airline?.cost || null,
              telecom: submission.results.telecom?.cost || null,
              banking_knowledge: submission.results.banking_knowledge?.cost || null
            },
            isLegacy,
            organization: submission.submitting_organization,
            reasoningEffort: submission.reasoning_effort || null,
            userSimulator: submission.methodology?.user_simulator || null,
            bankingRetrievalConfig: submission.results.banking_knowledge?.retrieval_config || null,
            // Voice-specific fields
            voiceConfig: submission.voice_config || null,
            // Add verification status
            // For 'custom' submissions, we relax the modified_prompts constraint
            // Custom submissions are allowed to modify prompts as long as they have trajectories and don't omit questions
            // For voice submissions, trajectories are never available so skip that check
            isVerified: modality === 'voice'
              ? (submission.methodology?.verification?.omitted_questions === false &&
                 (submission.submission_type === 'custom' || submission.methodology?.verification?.modified_prompts === false))
              : (submission.trajectories_available && 
                 submission.methodology?.verification?.omitted_questions === false &&
                 (submission.submission_type === 'custom' || submission.methodology?.verification?.modified_prompts === false)),
            verificationDetails: submission.methodology?.verification || null,
            // Submission type: 'standard' (default) or 'custom'
            submissionType: submission.submission_type || 'standard'
          }
          
          loadedData[submissionDir] = modelData
        } catch (error) {
          console.warn(`Error loading ${submissionDir}:`, error)
        }
      }
      
      // Load current text submissions
      for (const dir of currentDirs) {
        await loadSubmission(dir, false, 'text')
      }
      
      // Load legacy text submissions
      for (const dir of legacyDirs) {
        await loadSubmission(dir, true, 'text')
      }
      
      // Load voice submissions
      for (const dir of voiceDirs) {
        await loadSubmission(dir, false, 'voice')
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

  // Save leaderboard state to localStorage
  useEffect(() => {
    localStorage.setItem('benchmark', benchmark)
  }, [benchmark])

  // Keep benchmark in URL for shareable deep links, e.g. #leaderboard?benchmark=voice
  useEffect(() => {
    if (!window.location.hash.startsWith('#leaderboard')) return

    const hash = window.location.hash.slice(1)
    const [route, queryString = ''] = hash.split('?')
    const params = new URLSearchParams(queryString)
    params.set('benchmark', benchmark)

    const nextHash = `${route}?${params.toString()}`
    if (hash !== nextHash) {
      window.history.replaceState(null, '', `#${nextHash}`)
    }
  }, [benchmark])

  // React to manual hash edits or browser navigation events.
  useEffect(() => {
    const syncBenchmarkFromHash = () => {
      const fromHash = getBenchmarkFromHash()
      if (fromHash) {
        setBenchmark(prev => (prev === fromHash ? prev : fromHash))
      }
    }

    window.addEventListener('hashchange', syncBenchmarkFromHash)
    window.addEventListener('popstate', syncBenchmarkFromHash)
    return () => {
      window.removeEventListener('hashchange', syncBenchmarkFromHash)
      window.removeEventListener('popstate', syncBenchmarkFromHash)
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('domain', domain)
  }, [domain])

  useEffect(() => {
    localStorage.setItem('selectedPassK', selectedPassK)
  }, [selectedPassK])

  useEffect(() => {
    localStorage.setItem('sortDirection', sortDirection)
  }, [sortDirection])

  useEffect(() => {
    localStorage.setItem('showStandard', showStandard)
  }, [showStandard])

  useEffect(() => {
    localStorage.setItem('showCustom', showCustom)
  }, [showCustom])

  useEffect(() => {
    localStorage.setItem('showLegacy', showLegacy)
  }, [showLegacy])

  // Close filter info popup when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showFilterInfo && !event.target.closest('.filter-info-container')) {
        setShowFilterInfo(false)
      }
    }
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [showFilterInfo])

  // Handle benchmark toggle with domain reset
  const handleBenchmarkChange = (newBenchmark) => {
    setBenchmark(newBenchmark)
    setExpandedRows(new Set())
    if (newBenchmark === 'voice') {
      // Voice doesn't have banking_knowledge or a meaningful overall (no banking)
      // Reset to 'overall' which will show avg of available domains
      if (domain === 'banking_knowledge') {
        setDomain('overall')
      }
      // Voice only has pass^1
      setSelectedPassK(1)
    }
  }

  // Handle sort direction toggle on the score column
  const handleSort = () => {
    setSortDirection(sortDirection === 'desc' ? 'asc' : 'desc')
  }

  // Toggle row expansion
  const toggleExpand = (modelName) => {
    setExpandedRows(prev => {
      const next = new Set(prev)
      if (next.has(modelName)) {
        next.delete(modelName)
      } else {
        next.add(modelName)
      }
      return next
    })
  }

  // Loading and error states
  if (isLoading) {
    return (
      <div className="leaderboard-wrapper">
      <div className="leaderboard-container">
        <h2 className="leaderboard-title">Leaderboard</h2>
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <p>Loading leaderboard data...</p>
        </div>
      </div>
      </div>
    )
  }

  if (loadError) {
    return (
      <div className="leaderboard-wrapper">
      <div className="leaderboard-container">
        <h2 className="leaderboard-title">Leaderboard</h2>
        <div className="error-state">
          <p>Error loading leaderboard data: {loadError}</p>
          <button onClick={loadSubmissionData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
      </div>
    )
  }

  if (Object.keys(passKData).length === 0) {
    return (
      <div className="leaderboard-wrapper">
      <div className="leaderboard-container">
        <h2 className="leaderboard-title">Leaderboard</h2>
        <div className="empty-state">
          <p>No leaderboard data available.</p>
        </div>
      </div>
      </div>
    )
  }

  const hasUnverifiedSubmission = Object.values(passKData).some(data => {
    // Filter by benchmark modality
    if (data.modality !== benchmark) return false
    if (data.isLegacy && !showLegacy) return false
    const isStandard = data.submissionType === 'standard' || !data.submissionType
    const isCustom = data.submissionType === 'custom'
    if ((isStandard && !showStandard) || (isCustom && !showCustom)) return false
    if (domain === 'overall') {
      if (!data.overall.some(val => val !== null)) return false
    } else {
      if (!data[domain].some(val => val !== null)) return false
    }
    return !data.isVerified
  })

  // Determine domains available for current benchmark
  const isVoice = benchmark === 'voice'
  const availableDomains = isVoice
    ? [
        { key: 'overall', label: '📊 Overall' },
        { key: 'retail', label: '🛍️ Retail' },
        { key: 'airline', label: '✈️ Airline' },
        { key: 'telecom', label: '📱 Telecom' },
      ]
    : [
        { key: 'overall', label: '📊 Overall' },
        { key: 'banking_knowledge', label: '🏦 Banking' },
        { key: 'retail', label: '🛍️ Retail' },
        { key: 'airline', label: '✈️ Airline' },
        { key: 'telecom', label: '📱 Telecom' },
      ]

  // For voice overall, only average the 3 non-banking domains
  const voiceDomains = ['retail', 'airline', 'telecom']

  return (
    <div className="leaderboard-wrapper">
    <div className="leaderboard-container">
      {/* Benchmark Selector */}
      <div className="benchmark-selector">
        <div className="benchmark-toggle-container">
          <button
            className={`benchmark-toggle-option ${benchmark === 'text' ? 'active' : ''}`}
            onClick={() => handleBenchmarkChange('text')}
          >
            <span className="benchmark-icon">📝</span> τ-bench
          </button>
          <button
            className={`benchmark-toggle-option ${benchmark === 'voice' ? 'active' : ''}`}
            onClick={() => handleBenchmarkChange('voice')}
          >
            <span className="benchmark-icon">🎙️</span> τ-voice
          </button>
          <div
            className="benchmark-toggle-slider"
            style={{
              transform: benchmark === 'text' ? 'translateX(0%)' : 'translateX(100%)'
            }}
          />
        </div>
      </div>

      <h2 className="leaderboard-title">{isVoice ? 'τ-voice Leaderboard' : 'τ-bench Leaderboard'}</h2>

      {/* Combined Controls Row */}
      <div className="leaderboard-controls">
        {/* Domain Toggle Switch */}
        <div className="domain-toggle-switch">
          <div className="toggle-container domain-toggle-container" style={{ '--domain-count': availableDomains.length }}>
            {availableDomains.map(d => (
              <button
                key={d.key}
                className={`toggle-option domain-toggle-option ${domain === d.key ? 'active' : ''}`}
                onClick={() => setDomain(d.key)}
              >
                {d.label}
              </button>
            ))}
            <div 
              className="toggle-slider domain-toggle-slider"
              style={{
                width: `calc((100% - 8px) / ${availableDomains.length})`,
                transform: `translateX(${availableDomains.findIndex(d => d.key === domain) * 100}%)`
              }}
            />
          </div>
        </div>

        {/* Submission Type Filter */}
        <div className="submission-type-filter">
          <label className="checkbox-container">
            <input 
              type="checkbox" 
              checked={showStandard}
              onChange={(e) => setShowStandard(e.target.checked)}
            />
            <span className="checkbox-checkmark"></span>
            <span className="checkbox-label">Standard</span>
          </label>
          <label className="checkbox-container">
            <input 
              type="checkbox" 
              checked={showCustom}
              onChange={(e) => setShowCustom(e.target.checked)}
            />
            <span className="checkbox-checkmark"></span>
            <span className="checkbox-label">Custom</span>
          </label>
          {!isVoice && (
            <label className="checkbox-container">
              <input 
                type="checkbox" 
                checked={showLegacy}
                onChange={(e) => setShowLegacy(e.target.checked)}
              />
              <span className="checkbox-checkmark"></span>
              <span className="checkbox-label">Legacy (v1)</span>
            </label>
          )}
          <div className="filter-info-container">
            <button 
              className="filter-info-button"
              onClick={() => setShowFilterInfo(!showFilterInfo)}
              aria-label="What do Standard, Custom, and Legacy mean?"
            >
              <span className="info-icon">ⓘ</span>
            </button>
            {showFilterInfo && (
              <div className="filter-info-popup">
                <div className="filter-info-content">
                  <button className="filter-info-close" onClick={() => setShowFilterInfo(false)}>×</button>
                  <h4>Submission Types</h4>
                  <div className="filter-info-item">
                    <strong>Standard</strong>
                    <p>Results using the default τ-bench scaffold: a base LLM with the standard tool set and prompts.</p>
                  </div>
                  <div className="filter-info-item">
                    <strong>Custom</strong>
                    <p>Results using modified scaffolds, such as multi-model routers, additional tools, custom prompting strategies, or other orchestration approaches.</p>
                  </div>
                  <div className="filter-info-item">
                    <strong>Legacy (v1)</strong>
                    <p>Submissions from the original τ-bench v1 task set. These results are not directly comparable to current submissions due to task fixes in airline and retail domains.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Table View */}
      {(!showStandard && !showCustom && (isVoice || !showLegacy)) ? (
          <div className="filter-empty-state">
            <div className="empty-icon">🔍</div>
            <h3>No Results</h3>
            <p>Please select at least one submission type filter (Standard, Custom, or Legacy) to view results.</p>
          </div>
        ) : (
        <div className="reliability-metrics">
        <div className="metrics-table-container">
          <table className="reliability-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>{domain === 'banking_knowledge' ? 'Retrieval' : 'Submitting Org'}</th>
                {isVoice ? (
                  <th>Provider</th>
                ) : (
                  <th>Reasoning</th>
                )}
                <th>User Sim</th>
                <th className="passk-header-cell">
                  <div className="passk-header-toggle">
                    {isVoice ? (
                      <button className="passk-header-btn active">Pass^1</button>
                    ) : (
                      [1, 2, 3, 4].map(k => (
                        <button
                          key={k}
                          className={`passk-header-btn ${selectedPassK === k ? 'active' : ''}`}
                          onClick={() => setSelectedPassK(k)}
                        >
                          Pass^{k}
                        </button>
                      ))
                    )}
                    <button 
                      className="passk-sort-btn"
                      onClick={handleSort}
                      title={sortDirection === 'desc' ? 'Sorted descending' : 'Sorted ascending'}
                    >
                      {sortDirection === 'desc' ? '↓' : '↑'}
                    </button>
                  </div>
                </th>
                <th className="expand-header"></th>
              </tr>
            </thead>
            <tbody>
              {(() => {
                // Calculate domain-specific scores for ranking
                const modelStats = Object.entries(passKData)
                  .filter(([modelName, data]) => {
                    // Filter by benchmark modality
                    if (data.modality !== benchmark) {
                      return false
                    }
                    // Filter out legacy submissions unless toggled on
                    if (data.isLegacy && !showLegacy) {
                      return false
                    }
                    
                    // Filter by submission type
                    const isStandard = data.submissionType === 'standard' || !data.submissionType
                    const isCustom = data.submissionType === 'custom'
                    if ((isStandard && !showStandard) || (isCustom && !showCustom)) {
                      return false
                    }
                    
                    // For voice overall, compute from 3 domains only
                    if (isVoice && domain === 'overall') {
                      return voiceDomains.some(d => data[d]?.some(val => val !== null))
                    }
                    
                    // For overall domain, only include models that have data for all 4 domains
                    if (domain === 'overall') {
                      return data.overall.some(val => val !== null)
                    }
                    // For individual domains, only include models that have data for that domain
                    return data[domain].some(val => val !== null)
                  })
                  .map(([submissionKey, data]) => {
                  // For voice overall, compute average across 3 domains
                  const domainData = (isVoice && domain === 'overall')
                    ? [0, 1, 2, 3].map(passIndex => {
                        const values = voiceDomains
                          .map(d => data[d]?.[passIndex])
                          .filter(v => v !== null && v !== undefined)
                        return values.length > 0 ? values.reduce((s, v) => s + v, 0) / values.length : null
                      })
                    : data[domain]
                  const pass1Score = domainData[0]
                  const hasCompleteData = domainData.every(val => val !== null)
                  const hasAnyData = domainData.some(val => val !== null)
                  const consistencyScore = hasCompleteData 
                    ? domainData[3] / domainData[0]
                    : null
                  
                  return {
                    key: submissionKey,
                    displayName: data.modelName,
                    data: data,
                    domainData: domainData,
                    pass1Score,
                    hasCompleteData,
                    hasAnyData,
                    consistencyScore,
                    organization: data.organization
                  }
                })
                
                // Sort by selected pass^k metric and direction
                const passIndex = selectedPassK - 1
                modelStats.sort((a, b) => {
                  // First priority: models with any data for this domain
                  if (a.hasAnyData && !b.hasAnyData) return -1
                  if (!a.hasAnyData && b.hasAnyData) return 1
                  if (!a.hasAnyData && !b.hasAnyData) return 0
                  
                  const aValue = a.domainData[passIndex]
                  const bValue = b.domainData[passIndex]
                  
                  // Handle null values (missing data)
                  if (aValue === null && bValue === null) return 0
                  if (aValue === null) return 1
                  if (bValue === null) return -1
                  
                  const multiplier = sortDirection === 'desc' ? 1 : -1
                  return (bValue - aValue) * multiplier
                })
                
                // Show empty state if no results after filtering
                if (modelStats.length === 0) {
                  return (
                    <tr className="empty-results-row">
                      <td colSpan="7" className="empty-results-cell">
                        <div className="empty-results-content">
                          <span className="empty-icon">🔧</span>
                          <span className="empty-text">
                            {showCustom && !showStandard 
                              ? "No custom submissions yet. Be the first to submit results with a custom scaffold!"
                              : "No results match the current filters."}
                          </span>
                        </div>
                      </td>
                    </tr>
                  )
                }
                
                return modelStats.map((model, index) => {
                  const isExpanded = expandedRows.has(model.key)
                  return (
                   <React.Fragment key={model.key}>
                   <tr className={`model-row ${model.data.isLegacy ? 'legacy-model' : ''} ${isExpanded ? 'expanded' : ''}`}>
                     {/* Rank */}
                     <td className="rank-cell">
                       <span className={`rank-number ${!model.data.isLegacy && index === 0 ? 'rank-gold' : !model.data.isLegacy && index === 1 ? 'rank-silver' : !model.data.isLegacy && index === 2 ? 'rank-bronze' : ''}`}>
                         #{index + 1}
                       </span>
                     </td>
                     {/* Model Name */}
                     <td className="model-info">
                       <div className="model-name">
                         {model.displayName}
                         {model.data.isLegacy && <span className="legacy-badge">v1</span>}
                         {!model.data.isVerified && (
                           <span className="unverified-badge" title="Unverified submission - see details for more information">
                             ⚠️
                           </span>
                         )}
                       </div>
                     </td>
                     
                     {/* Organization / Retrieval Config (banking) */}
                     <td className="organization-info">
                       {domain === 'banking_knowledge' ? (
                         model.data.bankingRetrievalConfig ? (
                           <span className={`retrieval-badge retrieval-${model.data.bankingRetrievalConfig}`}>
                             🔍 {model.data.bankingRetrievalConfig === 'terminal' ? 'Terminal'
                               : model.data.bankingRetrievalConfig === 'text-emb-3-large' ? 'text-emb-3-large'
                               : model.data.bankingRetrievalConfig === 'qwen3-emb' ? 'Qwen3-Emb'
                               : model.data.bankingRetrievalConfig === 'bm25' ? 'BM25'
                               : model.data.bankingRetrievalConfig}
                           </span>
                         ) : (
                           <span className="no-data">—</span>
                         )
                       ) : (
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
                          {(model.organization === 'Alibaba' || model.organization === 'Qwen') && (
                            <img src={`${import.meta.env.BASE_URL}qwen-color.png`} alt="Qwen" className="logo-img" />
                          )}
                         {model.organization === 'Google' && (
                           <img src={`${import.meta.env.BASE_URL}Google__G__logo.svg.png`} alt="Google" className="logo-img" />
                         )}
                         {model.organization === 'NVIDIA' && (
                           <img src={`${import.meta.env.BASE_URL}Logo-nvidia-transparent-PNG.png`} alt="NVIDIA" className="logo-img" />
                         )}
                        </div>
                         <span className="org-name">{model.organization}</span>
                       </div>
                       )}
                     </td>

                     {/* Reasoning Effort / Voice Provider */}
                     {isVoice ? (
                       <td className="reasoning-info">
                         {model.data.voiceConfig?.provider ? (
                           <span className="voice-provider-badge">
                             {model.data.voiceConfig.provider}
                           </span>
                         ) : (
                           <span className="no-data">—</span>
                         )}
                       </td>
                     ) : (
                       <td className="reasoning-info">
                         {model.data.reasoningEffort ? (
                           <span className={`reasoning-badge reasoning-${model.data.reasoningEffort}`}>
                             {model.data.reasoningEffort}
                           </span>
                         ) : (
                           <span className="no-data">—</span>
                         )}
                       </td>
                     )}
                     
                     {/* User Simulator */}
                     <td className="user-sim-info">
                       {model.data.userSimulator ? (
                         isVoice && model.data.userSimulator.startsWith('v') ? (
                           <a
                             href={`https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-${model.data.userSimulator}`}
                             target="_blank"
                             rel="noopener noreferrer"
                             className="user-sim-name user-sim-version-link"
                             title="View voice user simulator source at this version"
                           >{model.data.userSimulator}</a>
                         ) : (
                           <span className="user-sim-name">{model.data.userSimulator}</span>
                         )
                       ) : (
                         <span className="no-data">—</span>
                       )}
                     </td>
                     {/* Score (selected Pass^k) */}
                     <td className="metric-cell score-cell">
                       {(() => {
                         const value = model.domainData[selectedPassK - 1]
                         if (value !== null) {
                           return (
                             <div className="score-bar-container">
                               <div className="score-bar-track">
                                 <div 
                                   className="score-bar-fill"
                                   style={{ width: `${Math.min(value, 100)}%` }}
                                 />
                               </div>
                               <span className="score-bar-value">{value.toFixed(1)}%</span>
                             </div>
                           )
                         } else {
                           return <span className="no-data">—</span>
                         }
                       })()}
                     </td>
                     {/* Expand Toggle */}
                     <td className="expand-cell" onClick={() => toggleExpand(model.key)}>
                       <span className={`expand-caret ${isExpanded ? 'open' : ''}`}>▶</span>
                     </td>
                  </tr>
                  {/* Expandable Domain Breakdown Row */}
                  {isExpanded && (
                    <tr className="domain-detail-row">
                      <td colSpan="7" className="domain-detail-cell">
                        <div className="domain-breakdown">
                          {(isVoice
                            ? [
                                { key: 'retail', label: 'Retail', icon: '🛍️', desc: 'Order cancellations, returns, exchanges, address changes, and product inquiries.' },
                                { key: 'airline', label: 'Airline', icon: '✈️', desc: 'Flight bookings, modifications, cancellations, refunds, baggage, and compensation.' },
                                { key: 'telecom', label: 'Telecom', icon: '📱', desc: 'Technical support for connectivity issues, bill payments, and plan management.' },
                              ]
                            : [
                                { key: 'retail', label: 'Retail', icon: '🛍️', desc: 'Order cancellations, returns, exchanges, address changes, and product inquiries.' },
                                { key: 'airline', label: 'Airline', icon: '✈️', desc: 'Flight bookings, modifications, cancellations, refunds, baggage, and compensation.' },
                                { key: 'telecom', label: 'Telecom', icon: '📱', desc: 'Technical support for connectivity issues, bill payments, and plan management.' },
                                { key: 'banking_knowledge', label: 'Banking', icon: '🏦', desc: 'Banking customer service with knowledge retrieval over policy documents.' },
                              ]
                          ).map(({ key, label, icon, desc }) => {
                            const value = model.data[key]?.[selectedPassK - 1]
                            const submissionInfo = fullSubmissionData[model.key]
                            const hasTraj = submissionInfo?.trajectories_available && submissionInfo?.trajectory_files?.[key]
                            const retrievalConfig = key === 'banking_knowledge' ? model.data.bankingRetrievalConfig : null
                            const retrievalLabel = retrievalConfig === 'terminal' ? 'Terminal'
                              : retrievalConfig === 'text-emb-3-large' ? 'text-emb-3-large'
                              : retrievalConfig === 'qwen3-emb' ? 'Qwen3-Emb'
                              : retrievalConfig === 'bm25' ? 'BM25'
                              : retrievalConfig
                            return (
                              <div key={key} className="domain-breakdown-card">
                                <div className="domain-card-header">
                                  <span className="domain-breakdown-label">
                                    <span className="domain-breakdown-icon">{icon}</span>
                                    {label}
                                  </span>
                                  {retrievalConfig && (
                                    <span className={`retrieval-badge retrieval-${retrievalConfig}`} title={`Retrieval: ${retrievalLabel}`}>
                                      🔍 {retrievalLabel}
                                    </span>
                                  )}
                                  <span className="domain-info-icon" data-tooltip={desc}>ⓘ</span>
                                </div>
                                <div className="domain-card-body">
                                  {value !== null && value !== undefined ? (
                                    <div className="score-bar-container">
                                      <div className="score-bar-track">
                                        <div 
                                          className="score-bar-fill domain-bar-fill"
                                          style={{ width: `${Math.min(value, 100)}%` }}
                                        />
                                      </div>
                                      <span className="score-bar-value">{value.toFixed(1)}%</span>
                                    </div>
                                  ) : (
                                    <span className="no-data domain-no-data">—</span>
                                  )}
                                </div>
                                {hasTraj && (
                                  <a
                                    className="view-trajectories-link"
                                    href={`#trajectory-visualizer?model=${encodeURIComponent(submissionInfo.submissionDir)}&domain=${key}`}
                                  >
                                    View trajectories →
                                  </a>
                                )}
                              </div>
                            )
                          })}
                          <button
                            className="submission-details-btn"
                            onClick={() => handleModelClick(model.key)}
                          >
                            <span className="submission-details-btn-icon">📋</span>
                            <span className="submission-details-btn-label">Details</span>
                          </button>
                        </div>
                      </td>
                    </tr>
                  )}
                  </React.Fragment>
                  )
                })
              })()}
            </tbody>
          </table>
        </div>
        {hasUnverifiedSubmission && (
        <div className="verification-note">
          <span className="note-icon">⚠️</span>
          <span className="note-text">
            The warning icon indicates unverified submissions. Expand a row and click "Submission details" to view full verification information.
          </span>
        </div>
        )}
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
        <div className="sd-modal-overlay" onClick={closeModal}>
          <div className={`sd-modal ${modalClosing ? 'closing' : ''}`} onClick={(e) => e.stopPropagation()}>
            <div className="sd-modal-header">
              <h3>{selectedSubmission.model_name}</h3>
              <button className="sd-modal-close" onClick={closeModal}>✕</button>
            </div>
            <div className="sd-modal-body">
              <table className="sd-table">
                <tbody>
                  {/* Submission Info */}
                  <tr className="sd-section-header"><td colSpan="2">SUBMISSION</td></tr>
                  <tr><td>Model Organization</td><td>{selectedSubmission.model_organization}</td></tr>
                  <tr><td>Submitting Organization</td><td>{selectedSubmission.submitting_organization}</td></tr>
                  <tr><td>Submission Date</td><td>{selectedSubmission.submission_date}</td></tr>
                  <tr><td>Type</td><td>{selectedSubmission.submission_type || 'standard'}</td></tr>
                  <tr><td>Modality</td><td>{selectedSubmission.modality || 'text'}</td></tr>

                  {/* Contact */}
                  <tr className="sd-section-header"><td colSpan="2">CONTACT</td></tr>
                  {selectedSubmission.contact_info?.name && (
                    <tr><td>Name</td><td>{selectedSubmission.contact_info.name}</td></tr>
                  )}
                  <tr><td>Email</td><td>{selectedSubmission.contact_info?.email || '—'}</td></tr>
                  {selectedSubmission.contact_info?.github && (
                    <tr><td>GitHub</td><td>{selectedSubmission.contact_info.github}</td></tr>
                  )}

                  {/* Voice Config */}
                  {selectedSubmission.voice_config && (
                    <>
                      <tr className="sd-section-header"><td colSpan="2">VOICE CONFIGURATION</td></tr>
                      <tr><td>Provider</td><td>{selectedSubmission.voice_config.provider}</td></tr>
                      <tr><td>Model</td><td>{selectedSubmission.voice_config.model}</td></tr>
                      {selectedSubmission.voice_config.tick_duration_seconds != null && (
                        <tr><td>Tick Duration</td><td>{selectedSubmission.voice_config.tick_duration_seconds}s</td></tr>
                      )}
                      {selectedSubmission.voice_config.max_steps_seconds != null && (
                        <tr><td>Max Duration</td><td>{selectedSubmission.voice_config.max_steps_seconds}s</td></tr>
                      )}
                      {selectedSubmission.voice_config.user_tts_provider && (
                        <tr><td>User TTS</td><td>{selectedSubmission.voice_config.user_tts_provider}</td></tr>
                      )}
                    </>
                  )}

                  {/* Methodology */}
                  {selectedSubmission.methodology && (
                    <>
                      <tr className="sd-section-header"><td colSpan="2">METHODOLOGY</td></tr>
                      {selectedSubmission.methodology.user_simulator && (
                        <tr><td>User Simulator</td><td>
                          {selectedSubmission.modality === 'voice' && selectedSubmission.methodology.user_simulator.startsWith('v') ? (
                            <a
                              href={`https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-${selectedSubmission.methodology.user_simulator}`}
                              target="_blank"
                              rel="noopener noreferrer"
                            >{selectedSubmission.methodology.user_simulator}</a>
                          ) : (
                            selectedSubmission.methodology.user_simulator
                          )}
                        </td></tr>
                      )}
                      {selectedSubmission.methodology.evaluation_date && (
                        <tr><td>Evaluation Date</td><td>{selectedSubmission.methodology.evaluation_date}</td></tr>
                      )}
                      {selectedSubmission.methodology.tau2_bench_version && (
                        <tr><td>Bench Version</td><td>{selectedSubmission.methodology.tau2_bench_version}</td></tr>
                      )}
                      {selectedSubmission.methodology.notes && (
                        <tr><td>Notes</td><td className="sd-wrap">{selectedSubmission.methodology.notes}</td></tr>
                      )}
                    </>
                  )}

                  {/* Results */}
                  {selectedSubmission.results && Object.entries(selectedSubmission.results).map(([dmn, res]) => (
                    <React.Fragment key={dmn}>
                      <tr className="sd-section-header">
                        <td colSpan="2">{dmn.toUpperCase()} RESULTS</td>
                      </tr>
                      {[1, 2, 3, 4].map(k => (
                        <tr key={k}>
                          <td>Pass^{k}</td>
                          <td>{res[`pass_${k}`] != null ? `${res[`pass_${k}`].toFixed(1)}%` : '—'}</td>
                        </tr>
                      ))}
                      {res.cost != null && (
                        <tr><td>Avg Cost</td><td>${res.cost.toFixed(3)}</td></tr>
                      )}
                    </React.Fragment>
                  ))}

                  {/* Verification */}
                  {selectedSubmission.methodology?.verification && (
                    <>
                      <tr className="sd-section-header"><td colSpan="2">VERIFICATION</td></tr>
                      <tr>
                        <td>Status</td>
                        <td>
                          {(() => {
                            const isVoiceSub = selectedSubmission.modality === 'voice'
                            const verified = isVoiceSub
                              ? (selectedSubmission.methodology.verification.omitted_questions === false &&
                                 (selectedSubmission.submission_type === 'custom' || selectedSubmission.methodology.verification.modified_prompts === false))
                              : (selectedSubmission.trajectories_available && 
                                 selectedSubmission.methodology.verification.omitted_questions === false &&
                                 (selectedSubmission.submission_type === 'custom' || selectedSubmission.methodology.verification.modified_prompts === false))
                            return verified
                              ? <span className="sd-badge sd-verified">Verified</span>
                              : <span className="sd-badge sd-unverified">Unverified</span>
                          })()}
                        </td>
                      </tr>
                      <tr><td>Trajectories</td><td>{selectedSubmission.trajectories_available ? 'Yes' : 'No'}</td></tr>
                      <tr>
                        <td>Modified Prompts</td>
                        <td>{selectedSubmission.methodology.verification.modified_prompts === true ? 'Yes' : selectedSubmission.methodology.verification.modified_prompts === false ? 'No' : '—'}</td>
                      </tr>
                      <tr>
                        <td>Omitted Questions</td>
                        <td>{selectedSubmission.methodology.verification.omitted_questions === true ? 'Yes' : selectedSubmission.methodology.verification.omitted_questions === false ? 'No' : '—'}</td>
                      </tr>
                      {selectedSubmission.methodology.verification.details && (
                        <tr><td>Details</td><td className="sd-wrap">{selectedSubmission.methodology.verification.details}</td></tr>
                      )}
                    </>
                  )}

                  {/* References */}
                  {selectedSubmission.references && selectedSubmission.references.length > 0 && (
                    <>
                      <tr className="sd-section-header"><td colSpan="2">REFERENCES</td></tr>
                      {selectedSubmission.references.map((ref, i) => (
                        <tr key={i}>
                          <td>{ref.type?.replace('_', ' ') || 'link'}</td>
                          <td>
                            <a href={ref.url} target="_blank" rel="noopener noreferrer" className="sd-link">
                              {ref.title}
                            </a>
                          </td>
                        </tr>
                      ))}
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
    </div>
  )
}

export default Leaderboard 