import React, { useState, useEffect } from 'react'
import './Leaderboard.css'
import ProgressView from './ProgressView'

// The leaderboard is split into three buckets, one per benchmark track:
// τ³-Banking (published as τ-knowledge), τ³-Voice (published as τ-voice:
// retail/airline/telecom in real-time voice), and τ²-bench (core:
// retail/airline/telecom in text, near saturation).
const BENCHMARK_VALUES = new Set(['core', 'knowledge', 'voice'])

// Pre-bucket URLs and localStorage used benchmark=text for what is now 'core'.
const normalizeBenchmark = (value) => (value === 'text' ? 'core' : value)

const getBenchmarkFromHash = () => {
  const hash = window.location.hash.slice(1)
  const [route, queryString = ''] = hash.split('?')
  // Both #leaderboard?benchmark=… and #progress?benchmark=… select the
  // benchmark on the same view, so accept either route.
  if (route !== 'leaderboard' && route !== 'progress') return null

  const value = normalizeBenchmark(new URLSearchParams(queryString).get('benchmark'))
  return BENCHMARK_VALUES.has(value) ? value : null
}

const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`

const NO_CACHE = { cache: 'no-cache' }
const CORE_DOMAINS = [
  { key: 'overall', label: '📊 Overall' },
  { key: 'retail', label: '🛍️ Retail' },
  { key: 'airline', label: '✈️ Airline' },
  { key: 'telecom', label: '📱 Telecom' },
]
// TODO(voice-banking): when banking is supported in voice mode, add a
// Text | Voice modality toggle inside the Knowledge bucket rather than a
// banking domain tab under Voice. That keeps the Voice bucket's Overall
// (core 3 domains) stable and keeps all knowledge scores in one place.
const KNOWLEDGE_DOMAINS = [
  { key: 'banking_knowledge', label: '🏦 Banking' },
]
const VOICE_DOMAINS = [
  { key: 'overall', label: '📊 Overall' },
  { key: 'retail', label: '🛍️ Retail' },
  { key: 'airline', label: '✈️ Airline' },
  { key: 'telecom', label: '📱 Telecom' },
]

// Key order determines toggle order: newest tracks first.
const BENCHMARK_CONFIG = {
  knowledge: {
    label: 'τ³-Banking',
    icon: '🏦',
    title: 'τ³-Banking Leaderboard',
    description: 'Text agents resolving banking customer-service tasks over a ~700-document knowledge base. Published as τ-knowledge.',
    // Shown on hover wherever the track name appears without room for the
    // full description; maps the display name back to the paper name.
    hoverNote: 'τ³-Banking was published as τ-knowledge',
    modality: 'text',
    domains: KNOWLEDGE_DOMAINS,
    defaultDomain: 'banking_knowledge',
    breakdownDomains: ['banking_knowledge'],
  },
  voice: {
    label: 'τ³-Voice',
    icon: '🎙️',
    title: 'τ³-Voice Leaderboard',
    description: 'Real-time, full-duplex voice agents on retail, airline, and telecom customer-service tasks. Published as τ-voice.',
    hoverNote: 'τ³-Voice was published as τ-voice',
    modality: 'voice',
    domains: VOICE_DOMAINS,
    defaultDomain: 'overall',
    breakdownDomains: ['retail', 'airline', 'telecom'],
  },
  core: {
    label: 'τ²-bench',
    icon: '📝',
    title: 'τ²-bench Leaderboard',
    description: 'Text agents on retail, airline, and telecom customer-service tasks, where the agent and the user both act on the world.',
    modality: 'text',
    domains: CORE_DOMAINS,
    defaultDomain: 'overall',
    breakdownDomains: ['retail', 'airline', 'telecom'],
  },
}

const DOMAIN_CARDS = {
  retail: { key: 'retail', label: 'Retail', icon: '🛍️', desc: 'Order cancellations, returns, exchanges, address changes, and product inquiries.' },
  airline: { key: 'airline', label: 'Airline', icon: '✈️', desc: 'Flight bookings, modifications, cancellations, refunds, baggage, and compensation.' },
  telecom: { key: 'telecom', label: 'Telecom', icon: '📱', desc: 'Technical support for connectivity issues, bill payments, and plan management.' },
  banking_knowledge: { key: 'banking_knowledge', label: 'Banking', icon: '🏦', desc: 'Banking customer service with knowledge retrieval over policy documents.' },
}

const formatVoicePipeline = (pipeline) => {
  if (!pipeline) return ''
  return [
    pipeline.asr ? `ASR: ${pipeline.asr}` : null,
    pipeline.llm ? `LLM: ${pipeline.llm}` : null,
    pipeline.tts ? `TTS: ${pipeline.tts}` : null,
  ].filter(Boolean).join('\n')
}

// Voice interaction quality (τ-voice panel, condensed to four headline metrics).
// Rates backed by fewer than MIN_INTERACTION_N events are hidden as noise.
const MIN_INTERACTION_N = 10

const INTERACTION_METRICS = [
  {
    key: 'response_rate',
    label: 'Responsiveness',
    unit: '%',
    better: 'higher',
    desc: 'Fraction of user turns that received an agent response before the user had to speak again. Higher is better.',
  },
  {
    key: 'response_latency_mean',
    label: 'Latency',
    unit: 's',
    better: 'lower',
    desc: 'Mean seconds from the end of a user turn to the start of the agent response. Lower is better.',
  },
  {
    key: 'agent_interruption_rate',
    label: 'Interrupts',
    unit: '%',
    better: 'lower',
    desc: 'Agent interruption events per eligible user turn. An agent can interrupt the same turn more than once, so this can exceed 100%. Lower is better.',
  },
  {
    key: 'selectivity',
    label: 'Selectivity',
    unit: '%',
    better: 'higher',
    desc: 'How well the agent ignores audio not directed at it (backchannels, vocal tics, background speech). Higher is better.',
  },
]

const SELECTIVITY_PARTS = [
  { key: 'selectivity_backchannel', countKey: 'backchannel_total' },
  { key: 'selectivity_vocal_tic', countKey: 'vocal_tic_total' },
  { key: 'selectivity_non_directed', countKey: 'non_directed_total' },
]

const getInteractionPanel = (interactionMetrics, domainKey) => {
  if (!interactionMetrics) return null
  return domainKey === 'overall'
    ? interactionMetrics.overall || null
    : interactionMetrics.domains?.[domainKey] || null
}

// Support gate for a single rate: 'ok' when backed by >= MIN_INTERACTION_N
// events, 'low_n' when under-supported (including when counts are missing and
// support can't be verified), 'undefined' when there were no qualifying
// events so the rate isn't measurable at all.
const rateStatus = (value, n) => {
  if (value === null || value === undefined) return 'undefined'
  return n >= MIN_INTERACTION_N ? 'ok' : 'low_n'
}

// Latency is averaged over responded turns only, so its support count is
// response_rate * response_total rather than response_total itself.
const metricEventCount = (panel, metricKey) => {
  const total = panel.counts?.response_total ?? 0
  return metricKey === 'response_latency_mean'
    ? Math.round((panel.response_rate ?? 0) * total)
    : total
}

const selectivityPartStatus = (panel, part) =>
  rateStatus(panel[part.key], panel.counts?.[part.countKey] ?? 0)

// Composite selectivity is the mean of all three component rates. If any
// component is missing or under-supported the composite is hidden entirely:
// silently dropping a component would rank rows on differently-defined
// quantities (a 2-component mean vs everyone else's 3-component mean).
const panelSelectivity = (panel) => {
  const statuses = SELECTIVITY_PARTS.map((part) => selectivityPartStatus(panel, part))
  if (statuses.every((s) => s === 'undefined')) return { reason: 'undefined' }
  if (statuses.some((s) => s !== 'ok')) return { reason: 'low_n' }
  const sum = SELECTIVITY_PARTS.reduce((s, part) => s + panel[part.key], 0)
  return { reason: 'ok', value: sum / SELECTIVITY_PARTS.length }
}

const panelMetric = (panel, metricKey) => {
  if (metricKey === 'selectivity') return panelSelectivity(panel)
  const status = rateStatus(panel[metricKey], metricEventCount(panel, metricKey))
  return status === 'ok' ? { reason: 'ok', value: panel[metricKey] } : { reason: status }
}

// The overall panel is an unweighted mean of domain rates, so it is only as
// reliable as its weakest contributor: a domain rate that exists but is
// under-supported enters that mean with full weight. Hide the overall value
// whenever any contributing domain rate fails the support gate. (A domain
// with no qualifying events contributes nothing and doesn't count against it.)
const overallContaminated = (interactionMetrics, metricKey) => {
  const domains = Object.values(interactionMetrics.domains || {})
  if (metricKey === 'selectivity') {
    return domains.some((panel) =>
      SELECTIVITY_PARTS.some((part) => selectivityPartStatus(panel, part) === 'low_n'))
  }
  return domains.some((panel) =>
    rateStatus(panel[metricKey], metricEventCount(panel, metricKey)) === 'low_n')
}

// Returns { value, reason }: reason is 'ok', 'low_n', 'undefined', or
// 'unavailable' (submission has no interaction_metrics block).
const getInteractionCellInfo = (interactionMetrics, domainKey, metricKey) => {
  const panel = getInteractionPanel(interactionMetrics, domainKey)
  if (!panel) return { value: null, reason: 'unavailable' }
  const metric = panelMetric(panel, metricKey)
  if (metric.reason !== 'ok') return { value: null, reason: metric.reason }
  if (domainKey === 'overall' && overallContaminated(interactionMetrics, metricKey)) {
    return { value: null, reason: 'low_n' }
  }
  return { value: metric.value, reason: 'ok' }
}

const getInteractionValue = (interactionMetrics, domainKey, metricKey) =>
  getInteractionCellInfo(interactionMetrics, domainKey, metricKey).value

const INTERACTION_NO_DATA_TOOLTIP = {
  unavailable: 'Interaction metrics not available for this submission',
  low_n: `Not shown: backed by fewer than ${MIN_INTERACTION_N} events`,
  undefined: 'Not shown: no qualifying events in this run',
}

const formatInteractionValue = (value, unit) => {
  if (value === null || value === undefined) return '—'
  return unit === 's' ? `${value.toFixed(2)}s` : `${(value * 100).toFixed(1)}%`
}

const Leaderboard = () => {
  // Benchmark selector: 'core' (τ²-bench), 'knowledge' (τ-knowledge), or
  // 'voice' (τ-voice)
  const [benchmark, setBenchmark] = useState(() => {
    const fromHash = getBenchmarkFromHash()
    if (fromHash) return fromHash

    const fromStorage = normalizeBenchmark(localStorage.getItem('benchmark'))
    // Default to the first toggle position (newest track)
    return BENCHMARK_VALUES.has(fromStorage) ? fromStorage : 'knowledge'
  })
  // Add unified domain selection state with localStorage persistence
  const [domain, setDomain] = useState(() => {
    const storedBenchmark = normalizeBenchmark(localStorage.getItem('benchmark'))
    const storedDomain = localStorage.getItem('domain')
    const config = BENCHMARK_CONFIG[storedBenchmark] || BENCHMARK_CONFIG.knowledge
    return config.domains.some(({ key }) => key === storedDomain)
      ? storedDomain
      : config.defaultDomain
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
  // Voice ranking mode: 'passk' (default) or 'interaction' (τ-voice panel)
  const [rankBy, setRankBy] = useState('passk')
  // null = keep the pass^1 ordering; set by clicking a metric column header
  const [interactionMetric, setInteractionMetric] = useState(null)
  // Expanded rows state (set of model names)
  const [expandedRows, setExpandedRows] = useState(new Set())
  const [openPipelineKey, setOpenPipelineKey] = useState(null)
  
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
      const manifestResponse = await fetch(`${SUBMISSIONS_BASE}/manifest.json`, NO_CACHE)
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
          const response = await fetch(`${SUBMISSIONS_BASE}/${submissionDir}/submission.json`, NO_CACHE)
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
          
          // Overall = average of the three core domains (retail, airline,
          // telecom), only when all three have data. Banking is scored
          // separately in the Knowledge bucket.
          const hasRetailData = submission.results.retail?.pass_1 !== null && submission.results.retail?.pass_1 !== undefined
          const hasAirlineData = submission.results.airline?.pass_1 !== null && submission.results.airline?.pass_1 !== undefined
          const hasTelecomData = submission.results.telecom?.pass_1 !== null && submission.results.telecom?.pass_1 !== undefined
          
          const overallData = (hasRetailData && hasAirlineData && hasTelecomData) 
            ? [0, 1, 2, 3].map(passIndex => {
                const values = [retailData[passIndex], airlineData[passIndex], telecomData[passIndex]].filter(val => val !== null)
                return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : null
              })
            : [null, null, null, null] // No overall score if missing any core domain
          
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
            modelOrganization: submission.model_organization,
            reasoningEffort: submission.reasoning_effort || null,
            userSimulator: submission.methodology?.user_simulator || null,
            bankingRetrievalConfig: submission.results.banking_knowledge?.retrieval_config || null,
            // Voice-specific fields
            voiceConfig: submission.voice_config || null,
            interactionMetrics: submission.interaction_metrics || null,
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

  // Keep benchmark in URL for shareable deep links, e.g.
  // #leaderboard?benchmark=voice or #progress?benchmark=voice
  useEffect(() => {
    const currentHash = window.location.hash
    if (!currentHash.startsWith('#leaderboard') && !currentHash.startsWith('#progress')) {
      return
    }

    const hash = currentHash.slice(1)
    const [route, queryString = ''] = hash.split('?')
    const params = new URLSearchParams(queryString)
    params.set('benchmark', benchmark)
    params.delete('view')

    const nextHash = `${route}?${params.toString()}`
    if (hash !== nextHash) {
      window.history.replaceState(null, '', `#${nextHash}`)
    }
  }, [benchmark])

  // React to manual hash edits or browser navigation events.
  useEffect(() => {
    const syncFromHash = () => {
      const benchmarkFromHash = getBenchmarkFromHash()
      if (benchmarkFromHash) {
        setBenchmark(prev => (prev === benchmarkFromHash ? prev : benchmarkFromHash))
      }
    }

    window.addEventListener('hashchange', syncFromHash)
    window.addEventListener('popstate', syncFromHash)
    return () => {
      window.removeEventListener('hashchange', syncFromHash)
      window.removeEventListener('popstate', syncFromHash)
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('domain', domain)
  }, [domain])

  useEffect(() => {
    const config = BENCHMARK_CONFIG[benchmark]
    if (!config.domains.some(({ key }) => key === domain)) {
      setDomain(config.defaultDomain)
    }
  }, [benchmark, domain])

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
    const config = BENCHMARK_CONFIG[newBenchmark]
    if (!config.domains.some(({ key }) => key === domain)) {
      setDomain(config.defaultDomain)
    }
    if (newBenchmark === 'voice') {
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

  const benchConfig = BENCHMARK_CONFIG[benchmark]

  const hasUnverifiedSubmission = Object.values(passKData).some(data => {
    // Filter by benchmark modality
    if (data.modality !== benchConfig.modality) return false
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
  const availableDomains = benchConfig.domains
  const benchmarkKeys = Object.keys(BENCHMARK_CONFIG)

  // For voice overall, only average the 3 non-banking domains
  const voiceDomains = ['retail', 'airline', 'telecom']

  return (
    <div className="leaderboard-wrapper">
    <div className="leaderboard-container">
      {/* Benchmark Selector */}
      <div className="benchmark-selector">
        <div className="benchmark-toggle-container" style={{ '--benchmark-count': benchmarkKeys.length }}>
          {benchmarkKeys.map((key) => (
            <button
              key={key}
              className={`benchmark-toggle-option ${benchmark === key ? 'active' : ''}`}
              onClick={() => handleBenchmarkChange(key)}
              title={BENCHMARK_CONFIG[key].hoverNote}
            >
              <span className="benchmark-icon">{BENCHMARK_CONFIG[key].icon}</span> {BENCHMARK_CONFIG[key].label}
            </button>
          ))}
          <div
            className="benchmark-toggle-slider"
            style={{
              width: `calc((100% - 8px) / ${benchmarkKeys.length})`,
              transform: `translateX(${benchmarkKeys.indexOf(benchmark) * 100}%)`
            }}
          />
        </div>
      </div>

      <div className="leaderboard-title-row">
        <h2 className="leaderboard-title">{benchConfig.title}</h2>
      </div>
      <p className="leaderboard-subtitle">{benchConfig.description}</p>

      {/* Combined Controls Row — applies to both ranking and progress views */}
      <div className="leaderboard-controls">
        {/* Domain Toggle Switch (hidden when the bucket has a single domain) */}
        {availableDomains.length > 1 && (
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
        )}

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
          {benchmark === 'core' && (
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
      {(!showStandard && !showCustom && (benchmark !== 'core' || !showLegacy)) ? (
          <div className="filter-empty-state">
            <div className="empty-icon">🔍</div>
            <h3>No Results</h3>
            <p>Please select at least one submission type filter (Standard, Custom, or Legacy) to view results.</p>
          </div>
        ) : (
        <div className="reliability-metrics">
        <div className="metrics-table-container">
          <table className={`reliability-table ${isVoice ? 'voice-table' : ''} ${isVoice && rankBy === 'interaction' ? 'interaction-mode' : ''}`}>
            <thead>
              {(() => {
                // Voice keeps every column mounted in both ranking modes and
                // collapses the inactive ones with CSS transitions, so
                // switching modes animates instead of re-laying-out the table.
                const interactionMode = isVoice && rankBy === 'interaction'
                const headerSpan = isVoice ? 2 : 1
                return (
                  <>
                    <tr>
                      <th rowSpan={headerSpan}>Rank</th>
                      <th rowSpan={headerSpan}>Model</th>
                      <th className="release-header" rowSpan={headerSpan}>
                        <span className="col-anim">Released</span>
                      </th>
                      <th rowSpan={headerSpan}>{domain === 'banking_knowledge' ? 'Retrieval' : isVoice ? 'Provider' : 'Submitting Org'}</th>
                      <th rowSpan={headerSpan}>Reasoning</th>
                      <th rowSpan={headerSpan}>User Sim</th>
                      <th className="passk-header-cell" rowSpan={headerSpan}>
                        <div className="passk-header-toggle">
                          {isVoice ? (
                            <>
                              <button
                                className={`passk-header-btn ${rankBy === 'passk' ? 'active' : ''}`}
                                onClick={() => setRankBy('passk')}
                                title="Rank by task success"
                              >
                                Pass^1
                              </button>
                              <span className="col-anim interaction-toggle-wrap">
                                <button
                                  className="passk-header-btn"
                                  onClick={() => {
                                    setRankBy('interaction')
                                    setInteractionMetric(null)
                                  }}
                                  title="Show interaction quality, measured from the same full-duplex trajectories"
                                >
                                  Interaction Metrics
                                </button>
                              </span>
                            </>
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
                          {(!isVoice || rankBy === 'passk') && (
                            <button
                              className="passk-sort-btn"
                              onClick={handleSort}
                              title={sortDirection === 'desc' ? 'Sorted descending' : 'Sorted ascending'}
                            >
                              {sortDirection === 'desc' ? '↓' : '↑'}
                            </button>
                          )}
                        </div>
                      </th>
                      {isVoice && (
                        <th className="interaction-group-header" colSpan={4}>
                          <div className="passk-header-toggle col-anim">
                            <button
                              className="passk-header-btn active"
                              title="Showing interaction quality (still ordered by pass^1) — click a metric column to sort by it. Click Pass^1 to collapse."
                            >
                              Interaction Metrics
                            </button>
                          </div>
                        </th>
                      )}
                      <th className="expand-header" rowSpan={headerSpan}></th>
                    </tr>
                    {isVoice && (
                      <tr className="interaction-subheader-row">
                        {INTERACTION_METRICS.map((m) => (
                          <th
                            key={m.key}
                            className={`interaction-col-header ${interactionMode && interactionMetric === m.key ? 'active' : ''}`}
                            onClick={() => interactionMode && setInteractionMetric(interactionMetric === m.key ? null : m.key)}
                            title={interactionMode && interactionMetric === m.key
                              ? 'Back to pass^1 order'
                              : `Sort by ${m.label.toLowerCase()}`}
                          >
                            <div className="col-anim">
                              {m.label} {m.better === 'lower' ? '↓' : '↑'}
                              <span
                                className="interaction-info-icon"
                                data-tooltip={m.desc}
                                onClick={(event) => event.stopPropagation()}
                              >
                                ⓘ
                              </span>
                            </div>
                          </th>
                        ))}
                      </tr>
                    )}
                  </>
                )
              })()}
            </thead>
            <tbody>
              {(() => {
                // Calculate domain-specific scores for ranking
                const modelStats = Object.entries(passKData)
                  .filter(([, data]) => {
                    // Filter by benchmark modality
                    if (data.modality !== benchConfig.modality) {
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
                    
                    // For overall domain, only include models that have data for all core domains
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
                    organization: data.organization,
                    interactionValue: isVoice && interactionMetric
                      ? getInteractionValue(data.interactionMetrics, domain, interactionMetric)
                      : null,
                  }
                })

                // Entering interaction mode keeps the pass^1 ordering until the
                // user explicitly picks a metric column to sort by.
                const rankByInteraction = isVoice && rankBy === 'interaction' && interactionMetric !== null
                if (rankByInteraction) {
                  // Rank by the selected interaction metric in its natural
                  // direction; models without metrics sort last.
                  const better = INTERACTION_METRICS.find(m => m.key === interactionMetric)?.better
                  modelStats.sort((a, b) => {
                    if (a.interactionValue === null && b.interactionValue === null) return 0
                    if (a.interactionValue === null) return 1
                    if (b.interactionValue === null) return -1
                    return better === 'lower'
                      ? a.interactionValue - b.interactionValue
                      : b.interactionValue - a.interactionValue
                  })
                } else {
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
                }
                
                // Show empty state if no results after filtering
                if (modelStats.length === 0) {
                  return (
                    <tr className="empty-results-row">
                      <td colSpan="8" className="empty-results-cell">
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
                  const displayOrg = isVoice ? (model.data.voiceConfig?.provider || model.organization) : model.organization
                  const pipelineSummary = isVoice ? formatVoicePipeline(model.data.voiceConfig?.pipeline) : ''
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

                     {/* Release Date (from model_release.release_date); collapses
                         in interaction mode to make room for the metric columns */}
                     <td className="release-date-cell">
                       <span className="col-anim">
                       {(() => {
                         const releaseInfo = fullSubmissionData[model.key]?.model_release
                         const releaseDate = releaseInfo?.release_date
                         if (!releaseDate) return <span className="no-data">—</span>
                         const label = new Date(releaseDate + 'T00:00:00Z').toLocaleDateString('en-US', {
                           year: 'numeric',
                           month: 'short',
                           day: 'numeric',
                           timeZone: 'UTC',
                         })
                         const inner = (
                           <span className="release-date" title={releaseDate}>{label}</span>
                         )
                         return releaseInfo?.announcement_url ? (
                           <a
                             className="release-date-link"
                             href={releaseInfo.announcement_url}
                             target="_blank"
                             rel="noopener noreferrer"
                             title={releaseInfo.announcement_title || releaseInfo.announcement_url}
                           >
                             {label}
                           </a>
                         ) : inner
                       })()}
                       </span>
                     </td>

                     {/* Organization / Retrieval Config (banking) */}
                     <td className={`organization-info${domain === 'banking_knowledge' ? ' organization-info-retrieval' : ''}`}>
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
                         {displayOrg === 'Anthropic' && (
                           <img src={`${import.meta.env.BASE_URL}claude.png`} alt="Anthropic" className="logo-img" />
                         )}
                         {displayOrg === 'OpenAI' && (
                           <img src={`${import.meta.env.BASE_URL}openai.svg`} alt="OpenAI" className="logo-img" />
                         )}
                         {displayOrg === 'Sierra' && (
                           <img src={`${import.meta.env.BASE_URL}sierra-logo.png`} alt="Sierra" className="logo-img" />
                         )}
                         {displayOrg === 'Moonshot AI' && (
                           <span className="emoji-logo">🚀</span>
                         )}
                         {displayOrg === 'DeepSeek' && (
                           <img src={`${import.meta.env.BASE_URL}DeepSeek_logo_icon.png`} alt="DeepSeek" className="logo-img" />
                         )}
                         {(displayOrg === 'Alibaba' || displayOrg === 'Qwen') && (
                           <img src={`${import.meta.env.BASE_URL}qwen-color.png`} alt="Qwen" className="logo-img" />
                         )}
                        {displayOrg === 'Google' && (
                          <img src={`${import.meta.env.BASE_URL}Google__G__logo.svg.png`} alt="Google" className="logo-img" />
                        )}
                        {displayOrg === 'NVIDIA' && (
                          <img src={`${import.meta.env.BASE_URL}Logo-nvidia-transparent-PNG.png`} alt="NVIDIA" className="logo-img" />
                        )}
                        {displayOrg === 'xAI' && (
                          <img src={`${import.meta.env.BASE_URL}xai-logo.svg`} alt="xAI" className="logo-img" />
                        )}
                       </div>
                        <div className="org-text">
                          <span className="org-name" title={pipelineSummary || displayOrg}>{displayOrg}</span>
                          {pipelineSummary && (
                            <span className="voice-pipeline-summary">
                              <span>ASR + LLM + TTS</span>
                              <button
                                type="button"
                                className={`voice-pipeline-info ${openPipelineKey === model.key ? 'open' : ''}`}
                                data-tooltip={pipelineSummary}
                                aria-label={pipelineSummary}
                                aria-expanded={openPipelineKey === model.key}
                                onClick={(event) => {
                                  event.stopPropagation()
                                  setOpenPipelineKey(openPipelineKey === model.key ? null : model.key)
                                }}
                              >
                                ⓘ
                              </button>
                            </span>
                          )}
                        </div>
                      </div>
                       )}
                     </td>

                     {/* Reasoning Effort */}
                     <td className="reasoning-info">
                       {model.data.reasoningEffort ? (
                         <span style={{textTransform: 'lowercase'}}>{model.data.reasoningEffort}</span>
                       ) : (
                         <span className="no-data">—</span>
                       )}
                     </td>
                     
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
                     {/* Score (selected Pass^k); in interaction mode the bar
                         collapses away, leaving the plain number */}
                     <td className="metric-cell score-cell">
                       {(() => {
                         const value = model.domainData[selectedPassK - 1]
                         if (value === null) {
                           return <span className="no-data">—</span>
                         }
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
                       })()}
                     </td>
                     {/* Interaction metrics fan-out (voice); collapsed outside
                         interaction mode */}
                     {isVoice && INTERACTION_METRICS.map((m) => {
                       const { value, reason } = getInteractionCellInfo(model.data.interactionMetrics, domain, m.key)
                       return (
                         <td
                           key={m.key}
                           className={`metric-cell interaction-cell ${rankBy === 'interaction' && interactionMetric === m.key ? 'interaction-cell-sorted' : ''}`}
                         >
                           <span className="col-anim">
                             {value !== null ? (
                               formatInteractionValue(value, m.unit)
                             ) : (
                               <span className="no-data" title={INTERACTION_NO_DATA_TOOLTIP[reason]}>
                                 —
                               </span>
                             )}
                           </span>
                         </td>
                       )
                     })}
                     {/* Expand Toggle */}
                     <td className="expand-cell" onClick={() => toggleExpand(model.key)}>
                       <span className={`expand-caret ${isExpanded ? 'open' : ''}`}>▶</span>
                     </td>
                  </tr>
                  {/* Expandable Domain Breakdown Row */}
                  {isExpanded && (
                    <tr className="domain-detail-row">
                      <td colSpan={isVoice ? 12 : 8} className="domain-detail-cell">
                        <div className="domain-breakdown">
                          {benchConfig.breakdownDomains.map((k) => DOMAIN_CARDS[k]).map(({ key, label, icon, desc }) => {
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
                                {isVoice && model.data.interactionMetrics && (
                                  <div className="domain-interaction-metrics">
                                    {INTERACTION_METRICS.map((metric) => {
                                      const cell = getInteractionCellInfo(model.data.interactionMetrics, key, metric.key)
                                      return (
                                        <div key={metric.key} className="domain-interaction-row" title={metric.desc}>
                                          <span className="domain-interaction-label">
                                            {metric.label} {metric.better === 'lower' ? '↓' : '↑'}
                                          </span>
                                          <span className="domain-interaction-value">
                                            {cell.value !== null
                                              ? formatInteractionValue(cell.value, metric.unit)
                                              : <span className="no-data" title={INTERACTION_NO_DATA_TOOLTIP[cell.reason]}>—</span>}
                                          </span>
                                        </div>
                                      )
                                    })}
                                  </div>
                                )}
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
                        {isVoice && model.data.interactionMetrics && (
                          <div className="interaction-definitions">
                            <a
                              className="interaction-breakdown-link"
                              href="https://github.com/sierra-research/tau2-bench/blob/main/docs/interaction-metrics.md"
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              Metric definitions →
                            </a>
                          </div>
                        )}
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


      {/* Progress Over Time (always below the ranking table) */}
      <div id="progress" style={{ scrollMarginTop: '80px' }}>
      <ProgressView
        passKData={passKData}
        fullSubmissionData={fullSubmissionData}
        benchmark={benchmark}
        domain={domain}
        showStandard={showStandard}
        showCustom={showCustom}
        showLegacy={showLegacy}
        baseUrl={import.meta.env.BASE_URL}
      />
      </div>

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
              href="https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md" 
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

                  {/* Model Release */}
                  {selectedSubmission.model_release && (
                    <>
                      <tr className="sd-section-header"><td colSpan="2">MODEL RELEASE</td></tr>
                      {selectedSubmission.model_release.release_date && (
                        <tr><td>Release Date</td><td>{selectedSubmission.model_release.release_date}</td></tr>
                      )}
                      {selectedSubmission.model_release.announcement_url && (
                        <tr>
                          <td>Announcement</td>
                          <td>
                            <a
                              href={selectedSubmission.model_release.announcement_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="sd-link"
                            >
                              {selectedSubmission.model_release.announcement_title || selectedSubmission.model_release.announcement_url}
                            </a>
                          </td>
                        </tr>
                      )}
                    </>
                  )}

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
                      {selectedSubmission.voice_config.pipeline && (
                        <>
                          <tr className="sd-section-header"><td colSpan="2">CASCADE COMPONENTS</td></tr>
                          {selectedSubmission.voice_config.pipeline.asr && (
                            <tr><td>ASR</td><td>{selectedSubmission.voice_config.pipeline.asr}</td></tr>
                          )}
                          {selectedSubmission.voice_config.pipeline.llm && (
                            <tr><td>LLM</td><td>{selectedSubmission.voice_config.pipeline.llm}</td></tr>
                          )}
                          {selectedSubmission.voice_config.pipeline.tts && (
                            <tr><td>TTS</td><td>{selectedSubmission.voice_config.pipeline.tts}</td></tr>
                          )}
                        </>
                      )}
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
