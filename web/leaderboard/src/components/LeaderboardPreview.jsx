import { useState, useEffect } from 'react'
import './LeaderboardPreview.css'

const MEDAL_EMOJI = ['🥇', '🥈', '🥉']

const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`

function LeaderboardPreview({ onViewFullLeaderboard }) {
  const [textTop3, setTextTop3] = useState([])
  const [voiceTop3, setVoiceTop3] = useState([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadPreviewData()
  }, [])

  const loadPreviewData = async () => {
    try {
      const manifestResponse = await fetch(`${SUBMISSIONS_BASE}/manifest.json`)
      if (!manifestResponse.ok) return
      const manifest = await manifestResponse.json()

      const textDirs = [...(manifest.submissions || []), ...(manifest.legacy_submissions || [])]
      const voiceDirs = manifest.voice_submissions || []

      // Load text submissions and compute overall pass^1
      const textModels = []
      for (const dir of textDirs) {
        try {
          const res = await fetch(`${SUBMISSIONS_BASE}/${dir}/submission.json`)
          if (!res.ok) continue
          const sub = await res.json()

          // Only include standard submissions
          if (sub.submission_type && sub.submission_type !== 'standard') continue

          const r = sub.results
          const airline = r.airline?.pass_1
          const retail = r.retail?.pass_1
          const telecom = r.telecom?.pass_1
          const banking = r.banking_knowledge?.pass_1

          if (airline != null && retail != null && telecom != null && banking != null) {
            const overall = (airline + retail + telecom + banking) / 4
            textModels.push({
              name: sub.model_name,
              org: sub.model_organization,
              overall: overall,
            })
          }
        } catch { /* skip */ }
      }

      textModels.sort((a, b) => b.overall - a.overall)
      setTextTop3(textModels.slice(0, 3))

      // Load voice submissions and compute overall pass^1
      const voiceModels = []
      const voiceDomains = ['airline', 'retail', 'telecom']
      for (const dir of voiceDirs) {
        try {
          const res = await fetch(`${SUBMISSIONS_BASE}/${dir}/submission.json`)
          if (!res.ok) continue
          const sub = await res.json()

          if (sub.submission_type && sub.submission_type !== 'standard') continue

          const r = sub.results
          const values = voiceDomains
            .map(d => r[d]?.pass_1)
            .filter(v => v != null)

          if (values.length > 0) {
            const overall = values.reduce((s, v) => s + v, 0) / values.length
            voiceModels.push({
              name: sub.model_name,
              org: sub.model_organization,
              overall: overall,
            })
          }
        } catch { /* skip */ }
      }

      voiceModels.sort((a, b) => b.overall - a.overall)
      setVoiceTop3(voiceModels.slice(0, 3))
    } catch (err) {
      console.warn('Failed to load leaderboard preview:', err)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="leaderboard-preview">
        <div className="preview-loading">Loading leaderboard...</div>
      </div>
    )
  }

  return (
    <div className="leaderboard-preview">
      <div className="preview-tables">
        <div className="preview-table-wrapper">
          <h3 className="preview-table-title">
            <span className="preview-mode-badge text">Text</span>
            Overall
          </h3>
          <table className="preview-table">
            <thead>
              <tr>
                <th className="preview-rank-col">#</th>
                <th className="preview-model-col">Model</th>
                <th className="preview-score-col">Pass^1</th>
              </tr>
            </thead>
            <tbody>
              {textTop3.map((model, i) => (
                <tr key={i} className="preview-row">
                  <td className="preview-rank">{MEDAL_EMOJI[i]}</td>
                  <td className="preview-model">
                    <span className="preview-model-name">{model.name}</span>
                    <span className="preview-model-org">{model.org}</span>
                  </td>
                  <td className="preview-score">{model.overall.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="preview-more">⋯</div>
        </div>
        <div className="preview-table-wrapper">
          <h3 className="preview-table-title">
            <span className="preview-mode-badge voice">Voice</span>
            Overall
          </h3>
          <table className="preview-table">
            <thead>
              <tr>
                <th className="preview-rank-col">#</th>
                <th className="preview-model-col">Model</th>
                <th className="preview-score-col">Pass^1</th>
              </tr>
            </thead>
            <tbody>
              {voiceTop3.map((model, i) => (
                <tr key={i} className="preview-row">
                  <td className="preview-rank">{MEDAL_EMOJI[i]}</td>
                  <td className="preview-model">
                    <span className="preview-model-name">{model.name}</span>
                    <span className="preview-model-org">{model.org}</span>
                  </td>
                  <td className="preview-score">{model.overall.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="preview-more">⋯</div>
        </div>
      </div>
      <button className="preview-cta" onClick={onViewFullLeaderboard}>
        View Full Leaderboard →
      </button>
    </div>
  )
}

export default LeaderboardPreview
