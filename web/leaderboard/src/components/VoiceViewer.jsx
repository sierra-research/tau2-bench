import { useRef, useEffect, useState } from 'react'
import { buildSimData } from '../utils/voiceDataTransform'
import { renderVoiceViewer } from '../utils/voiceViewerRenderer'
import './VoiceViewer.css'

const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`

/**
 * Voice trajectory viewer component.
 *
 * Loads a full simulation JSON from the submissions directory,
 * transforms it into viewer data, and renders the overview view
 * (speech activity timeline + conversation table) with audio playback.
 */
const VoiceViewer = ({
  submissionDir,
  trajectoryDir,
  simulationId,
  taskId,
  voiceConfig,
  taskInfo,
}) => {
  const containerRef = useRef(null)
  const cleanupRef = useRef(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [simData, setSimData] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)

  useEffect(() => {
    if (!submissionDir || !trajectoryDir || !simulationId) return

    let cancelled = false
    const loadSim = async () => {
      try {
        setLoading(true)
        setError(null)

        const url = `${SUBMISSIONS_BASE}/${submissionDir}/trajectories/${trajectoryDir}/simulations/${simulationId}.json`
        const res = await fetch(url)
        if (!res.ok) {
          throw new Error(
            res.status === 404
              ? 'Simulation data not available for this task. The full tick data may not be included in this submission.'
              : `Failed to load simulation: ${res.statusText}`
          )
        }
        const simJson = await res.json()
        if (cancelled) return

        const tickDur = voiceConfig?.tick_duration_seconds || 0.2
        const data = buildSimData(simJson, {
          tickDur,
          domain: voiceConfig?.domain || '',
          agentModel: voiceConfig?.model || '',
          agentProvider: voiceConfig?.provider || '',
          taskInfo: taskInfo || {},
        })

        setSimData(data)

        // Construct the audio URL — the stereo mix lives alongside simulation data
        const audioSrc = `${SUBMISSIONS_BASE}/${submissionDir}/trajectories/${trajectoryDir}/artifacts/task_${taskId}/sim_${simulationId}/audio/both.wav`
        setAudioUrl(audioSrc)
      } catch (err) {
        if (!cancelled) setError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadSim()
    return () => { cancelled = true }
  }, [submissionDir, trajectoryDir, simulationId, taskId, voiceConfig, taskInfo])

  useEffect(() => {
    if (!simData || !containerRef.current) return

    if (cleanupRef.current) {
      cleanupRef.current()
      cleanupRef.current = null
    }

    cleanupRef.current = renderVoiceViewer(containerRef.current, simData, audioUrl)

    return () => {
      if (cleanupRef.current) {
        cleanupRef.current()
        cleanupRef.current = null
      }
    }
  }, [simData, audioUrl])

  if (loading) {
    return (
      <div className="voice-viewer-loading">
        <div className="loading-spinner"></div>
        <p>Loading voice simulation data...</p>
        <p className="loading-note">This may take a moment for large simulations</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="voice-viewer-error">
        <p>⚠️ {error}</p>
        <p className="voice-viewer-error-hint">
          Voice trajectory data may need to be downloaded via CLI for detailed inspection.
        </p>
      </div>
    )
  }

  return <div className="voice-viewer" ref={containerRef} />
}

export default VoiceViewer
