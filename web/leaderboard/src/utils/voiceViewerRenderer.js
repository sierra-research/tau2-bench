/**
 * Vanilla JS renderer for the voice simulation overview view.
 * Adapted from src/experiments/tau_voice/visualization/templates/viewer.js
 *
 * Renders the "overview" view (speech activity timeline + conversation table)
 * into a container element. Includes audio playback with synchronized playhead,
 * click-to-seek, speed control, and keyboard shortcuts when audio is available.
 *
 * @param {HTMLElement} container - DOM element to render into
 * @param {Object} D - SIM_DATA object (from voiceDataTransform.buildSimData)
 * @param {string|null} audioSrc - URL to the audio file (WAV), or null
 * @returns {Function} cleanup function to call on unmount
 */
export function renderVoiceViewer(container, D, audioSrc) {
  container.innerHTML = ''
  const totalDur = D.totalDuration
  if (!totalDur || totalDur <= 0) {
    container.innerHTML = '<div class="vv-empty">No tick data available for this simulation.</div>'
    return () => {}
  }

  const cleanupFns = []
  const TL_LABEL_W = 150

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag)
    if (attrs) Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'className') el.className = v
      else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v)
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v)
      else el.setAttribute(k, v)
    })
    children.flat(Infinity).forEach(c => {
      if (c == null) return
      el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c)
    })
    return el
  }

  function pct(t) { return (t / totalDur * 100) + '%' }

  function fmt(s) {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return m + ':' + String(sec).padStart(2, '0')
  }

  function fmtMs(ms) {
    const s = ms / 1000
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    const milli = Math.floor(ms % 1000)
    return `${m}:${String(sec).padStart(2, '0')}.${String(milli).padStart(3, '0')}`
  }

  function escapeHtml(text) {
    if (!text) return ''
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }

  function seededRand(seed) {
    let s = seed
    return function () {
      s = (s * 16807 + 0) % 2147483647
      return s / 2147483647
    }
  }

  // ── Info Card ──
  const infoItems = [
    ['Duration', `${D.duration}s (${fmt(D.duration)})`],
    ['Termination', D.terminationReason],
    ['Mode', D.mode],
    ['Ticks', `${D.numTicks} × ${D.tickDuration}s`],
    ['Reward', D.reward !== null ? D.reward : 'N/A'],
  ]

  if (D.speechEnvironment.complexity)
    infoItems.push(['Complexity', D.speechEnvironment.complexity])
  if (D.speechEnvironment.backgroundNoise)
    infoItems.push(['Background Noise', D.speechEnvironment.backgroundNoise])
  if (D.speechEnvironment.personaName)
    infoItems.push(['Persona', D.speechEnvironment.personaName])

  const infoGrid = h('div', { className: 'vv-info-grid' },
    ...infoItems.map(([label, value]) =>
      h('div', { className: 'vv-info-cell' },
        h('div', { className: 'vv-info-label' }, label),
        h('div', { className: 'vv-info-value' }, String(value)),
      )
    ),
  )

  if (D.taskInfo && D.taskInfo.reason) {
    const taskItems = [
      ['Reason for Call', D.taskInfo.reason],
      ['Task Instructions', D.taskInfo.taskInstructions],
      ['Known Info', D.taskInfo.knownInfo],
      ['Unknown Info', D.taskInfo.unknownInfo],
    ].filter(([_, v]) => v)

    if (taskItems.length) {
      container.appendChild(
        h('div', { className: 'vv-card vv-task-info-card' },
          h('div', { className: 'vv-card-header' },
            h('span', { className: 'vv-card-header-icon' }, '📋'),
            'User Task',
          ),
          h('div', { className: 'vv-info-grid' },
            ...taskItems.map(([label, value]) =>
              h('div', { className: 'vv-info-cell' },
                h('div', { className: 'vv-info-label' }, label),
                h('div', { className: 'vv-info-value' }, h('pre', null, value)),
              )
            ),
          ),
        )
      )
    }
  }

  container.appendChild(
    h('div', { className: 'vv-card' },
      h('div', { className: 'vv-card-header' },
        h('span', { className: 'vv-card-header-icon' }, '⚙️'),
        'Simulation Info',
      ),
      infoGrid,
    )
  )

  // ── Timeline Controls ──
  const playIcon = h('span', { className: 'vv-transport-icon' }, '▶')
  const pauseIcon = h('span', { className: 'vv-transport-icon', style: { display: 'none' } }, '⏸')
  const playBtn = h('button', { className: 'vv-transport-btn vv-transport-btn-play', title: 'Play / Pause' }, playIcon, pauseIcon)
  const timeDisplay = h('span', { className: 'vv-time-display' }, `0:00 / ${fmt(totalDur)}`)

  const speeds = [1, 1.5, 2, 0.5]
  let speedIdx = 0
  const speedBtn = h('button', { className: 'vv-speed-btn', title: 'Playback speed' }, '1×')

  // ── Timeline Card ──
  const trackUser = h('div', { className: 'vv-tl-track-wrap vv-tl-track-user' })
  const trackAgent = h('div', { className: 'vv-tl-track-wrap vv-tl-track-agent' })
  const effectsLane = h('div', { className: 'vv-annotation-lane vv-tl-effects-lane' })
  const playhead = h('div', { className: 'vv-playhead-line' })
  const clickOverlay = h('div', { className: 'vv-tl-click-overlay' })
  const tlRuler = h('div', { className: 'vv-ruler' })
  const legendEl = h('div', { className: 'vv-tl-legend' })

  const envDesc = D.speechEnvironment.backgroundNoise
    ? (D.speechEnvironment.environment || 'with noise')
    : null

  const tlTracksArea = h('div', { className: 'vv-tl-tracks-area' },
    h('div', { className: 'vv-track-row' },
      h('div', { className: 'vv-track-label' },
        h('div', { className: 'vv-track-accent', style: { background: 'var(--vv-user-color)' } }),
        h('div', { className: 'vv-track-info' },
          h('div', { className: 'vv-track-name' }, 'User'),
          envDesc ? h('div', { className: 'vv-track-desc' }, envDesc) : null,
        ),
      ),
      trackUser,
    ),
    h('div', { className: 'vv-track-row vv-annotation-row' },
      h('div', { className: 'vv-track-label' },
        h('div', { className: 'vv-track-accent', style: { background: 'var(--vv-gray-dark)' } }),
        h('div', { className: 'vv-track-info' },
          h('div', { className: 'vv-track-name' }, 'Effects'),
        ),
      ),
      effectsLane,
    ),
    h('div', { className: 'vv-track-row' },
      h('div', { className: 'vv-track-label' },
        h('div', { className: 'vv-track-accent', style: { background: 'var(--vv-agent-color)' } }),
        h('div', { className: 'vv-track-info' },
          h('div', { className: 'vv-track-name' }, 'Agent'),
        ),
      ),
      trackAgent,
    ),
    playhead,
    clickOverlay,
  )

  const timelineCard = h('div', { className: 'vv-card vv-timeline-card' },
    h('div', { className: 'vv-card-header' },
      h('span', { className: 'vv-card-header-icon' }, '📊'),
      'Speech Activity Timeline',
    ),
    h('div', { className: 'vv-timeline-controls' }, playBtn, timeDisplay, speedBtn),
    tlTracksArea,
    h('div', { className: 'vv-ruler-row' },
      h('div', { className: 'vv-track-label vv-ruler-spacer' }),
      tlRuler,
    ),
    legendEl,
  )

  container.appendChild(timelineCard)

  // Tooltip
  const tooltip = h('div', { className: 'vv-tooltip' })
  container.appendChild(tooltip)

  // ── Audio Player ──
  let audioEl = null
  if (audioSrc) {
    audioEl = h('audio', { controls: '', preload: 'metadata' },
      h('source', { src: audioSrc, type: 'audio/wav' }),
    )
    const stickyPlayer = h('div', { className: 'vv-sticky-player' },
      audioEl,
      h('div', { className: 'vv-shortcut-hint' }, 'Space = play/pause · ←→ = ±5s'),
    )
    container.appendChild(stickyPlayer)

    audioEl.addEventListener('error', () => {
      stickyPlayer.innerHTML = '<div class="vv-audio-unavailable">Audio file not available for this simulation</div>'
      audioEl = null
    })
  }

  // ── Render Speech Blocks ──
  let userSpeechEls = []
  let agentSpeechEls = []

  function renderSpeechBlocks(segments, track, type) {
    const rand = seededRand(type === 'user' ? 42 : 137)
    const els = []
    segments.forEach(seg => {
      const el = h('div', { className: `vv-tl-speech vv-tl-speech-${type}` })
      el.style.left = pct(seg.start)
      el.style.width = ((seg.end - seg.start) / totalDur * 100) + '%'
      el.dataset.start = seg.start
      el.dataset.end = seg.end
      el.dataset.text = seg.text
      el.title = seg.text.slice(0, 120) + (seg.text.length > 120 ? '…' : '')
      track.appendChild(el)

      const segDur = seg.end - seg.start
      const barCount = Math.max(4, Math.round(segDur * 2.5))
      for (let i = 0; i < barCount; i++) {
        const bar = document.createElement('div')
        bar.className = `vv-tl-bar vv-tl-bar-${type}`
        const height = 20 + rand() * 80
        bar.style.height = height + '%'
        bar.style.left = (i / barCount * 100) + '%'
        bar.style.bottom = ((100 - height) / 2) + '%'
        el.appendChild(bar)
      }
      els.push(el)
    })
    return els
  }

  function renderNoise() {
    const rand = seededRand(999)
    const w = trackUser.offsetWidth || 800
    const spacing = 4
    const n = Math.floor(w / spacing)
    for (let i = 0; i < n; i++) {
      const bar = document.createElement('div')
      bar.className = 'vv-tl-noise-bar'
      bar.style.height = (3 + rand() * 12) + 'px'
      bar.style.left = (i * spacing) + 'px'
      trackUser.appendChild(bar)
    }
  }

  const effectIcons = {
    frame_drop: '⏸',
    burst_noise: '⚡',
    out_of_turn_speech: '🗣',
    background_noise: '🔊',
    telephony: '📞',
  }

  function renderEffects() {
    if (!D.effects?.length) return
    D.effects.forEach(e => {
      const tSec = e.start_ms / 1000
      const endMs = e.end_ms != null ? e.end_ms : e.start_ms + 200
      const startPct = (tSec / totalDur * 100)
      const widthPct = Math.max(0.4, (endMs - e.start_ms) / 1000 / totalDur * 100)
      const cls = 'vv-effect vv-effect-' + e.type.replace(/_/g, '-')
      let title = e.type.replace(/_/g, ' ') + ` @ ${fmtMs(e.start_ms)}`
      if (e.params?.duration_ms) title += ` (${e.params.duration_ms}ms)`
      if (e.params?.text) title += ` "${e.params.text}"`
      const icon = effectIcons[e.type] || ''
      const el = h('div', { className: cls, title },
        h('span', { className: 'vv-effect-icon' }, icon),
      )
      el.style.left = startPct + '%'
      el.style.width = widthPct + '%'
      effectsLane.appendChild(el)
    })
  }

  function renderXAxis() {
    const step = totalDur <= 60 ? 5 : totalDur <= 180 ? 10 : 20
    for (let t = 0; t <= totalDur; t += step) {
      const mark = h('div', { className: 'vv-ruler-mark' })
      mark.style.left = (t / totalDur * 100) + '%'
      mark.dataset.time = t + 's'
      tlRuler.appendChild(mark)
    }
  }

  const effectLegendMeta = {
    frame_drop: { label: 'Frame Drop', cls: 'vv-effect-frame-drop', icon: '⏸' },
    burst_noise: { label: 'Burst Noise', cls: 'vv-effect-burst-noise', icon: '⚡' },
    out_of_turn_speech: { label: 'Out-of-Turn Speech', cls: 'vv-effect-out-of-turn-speech', icon: '🗣' },
    background_noise: { label: 'Background Noise', cls: 'vv-effect-background-noise', icon: '🔊' },
    telephony: { label: 'Telephony', cls: 'vv-effect-telephony', icon: '📞' },
  }

  function renderLegend() {
    const items = [
      { label: 'User Speech', html: '<span class="vv-legend-swatch"><span class="vv-legend-block vv-legend-block-user"></span></span>' },
      { label: 'Agent Speech', html: '<span class="vv-legend-swatch"><span class="vv-legend-block vv-legend-block-agent"></span></span>' },
    ]
    if (D.effects?.length) {
      const activeTypes = new Set(D.effects.map(e => e.type))
      ;['frame_drop', 'burst_noise', 'out_of_turn_speech', 'background_noise', 'telephony'].forEach(type => {
        if (!activeTypes.has(type)) return
        const m = effectLegendMeta[type]
        items.push({
          label: m.label,
          html: `<span class="vv-legend-swatch"><span class="vv-legend-effect-swatch ${m.cls}">${m.icon}</span></span>`,
        })
      })
    }
    items.forEach(item => {
      const el = document.createElement('div')
      el.className = 'vv-legend-item'
      el.innerHTML = item.html + '<span>' + item.label + '</span>'
      legendEl.appendChild(el)
    })
  }

  function initTimeline() {
    trackUser.innerHTML = ''
    trackAgent.innerHTML = ''
    effectsLane.innerHTML = ''
    tlRuler.innerHTML = ''
    legendEl.innerHTML = ''

    if (D.speechEnvironment.backgroundNoise) renderNoise()
    userSpeechEls = renderSpeechBlocks(D.speech.user, trackUser, 'user')
    agentSpeechEls = renderSpeechBlocks(D.speech.agent, trackAgent, 'agent')
    renderEffects()
    renderXAxis()
    renderLegend()
    timeDisplay.textContent = `0:00 / ${fmt(totalDur)}`
  }

  // ── Conversation Table ──
  const hasEffects = D.tickRows.some(r => r.effects?.length > 0)

  function seekAudio(timeSec) {
    if (!audioEl) return
    audioEl.currentTime = Math.max(0, Math.min(timeSec, audioEl.duration || totalDur))
    if (audioEl.paused) audioEl.play()
  }

  function buildTickTable() {
    const headerCells = [
      h('th', { className: 'vv-tick-col' }, 'Tick'),
      h('th', { className: 'vv-time-col' }, 'Time'),
      h('th', null, 'Agent Speech'),
      h('th', null, 'Agent Tools'),
      h('th', null, 'User Speech'),
      h('th', null, 'User Tools'),
    ]
    if (hasEffects) headerCells.push(h('th', { className: 'vv-effects-col-header' }, 'Effects'))
    const thead = h('thead', null, h('tr', null, ...headerCells))

    const rows = D.tickRows.map(row => {
      const tickRange = row.tickStart === row.tickEnd
        ? String(row.tickStart)
        : `${row.tickStart}–${row.tickEnd}`
      const timeStr = fmtMs(row.timeS * 1000)

      function buildToolsCell(calls, results) {
        if (!calls.length && !results.length) return h('td', { className: 'vv-tool-col' }, h('span', { className: 'vv-empty-cell' }, '—'))
        const items = []
        calls.forEach(c => items.push(h('div', { className: 'vv-tool-call-item' }, c)))
        results.forEach((r, i) => {
          const details = h('details', { className: 'vv-tool-result-item' },
            h('summary', null, `Result ${i + 1}`),
            h('pre', null, r),
          )
          items.push(details)
        })
        return h('td', { className: 'vv-tool-col' }, ...items)
      }

      const agentTd = h('td', { className: 'vv-agent-col' })
      if (row.agentText) agentTd.textContent = row.agentText
      else agentTd.appendChild(h('span', { className: 'vv-empty-cell' }, '—'))

      const userTd = h('td', { className: 'vv-user-col' })
      if (row.userText) userTd.textContent = row.userText
      else userTd.appendChild(h('span', { className: 'vv-empty-cell' }, '—'))

      const timeTd = h('td', {
        className: `vv-time-col ${audioEl ? 'vv-time-col-clickable' : ''}`,
        title: audioEl ? 'Click to play from here' : '',
        onClick: audioEl ? () => seekAudio(row.timeS) : undefined,
      }, timeStr)

      const rowChildren = [
        h('td', { className: 'vv-tick-col' }, tickRange),
        timeTd,
        agentTd,
        buildToolsCell(row.agentCalls, row.agentResults),
        userTd,
        buildToolsCell(row.userCalls, row.userResults),
      ]

      if (hasEffects) {
        const effectItems = (row.effects || []).map(e => {
          const s = (e.startMs / 1000).toFixed(1) + 's'
          const en = e.endMs != null ? (e.endMs / 1000).toFixed(1) + 's' : '...'
          const label = e.type.replace(/_/g, ' ')
          let detail = ''
          if (e.params) {
            if (e.params.duration_ms) detail = ` ${e.params.duration_ms}ms`
            if (e.params.file) detail = ` ${e.params.file}`
          }
          return h('div', { className: 'vv-effect-item vv-effect-' + e.type.replace(/_/g, '-') },
            h('span', { className: 'vv-effect-label' }, label),
            h('span', { className: 'vv-effect-time' }, ` (${s}–${en})${detail}`),
          )
        })
        rowChildren.push(
          h('td', { className: 'vv-effects-col' },
            effectItems.length > 0 ? effectItems : [h('span', { className: 'vv-empty-cell' }, '—')],
          )
        )
      }

      return h('tr', { 'data-start-time': String(row.timeS) }, ...rowChildren)
    })

    const tbody = h('tbody', null, ...rows)
    const table = h('table', { className: 'vv-tick-table' }, thead, tbody)
    return { table, tbody, rows, card: h('div', { className: 'vv-card' },
      h('div', { className: 'vv-card-header' },
        h('span', { className: 'vv-card-header-icon' }, '📝'),
        `Conversation · ${D.tickRows.length} groups`,
      ),
      h('div', { className: 'vv-tick-table-wrapper' }, table),
    )}
  }

  const tickView = buildTickTable()
  container.appendChild(tickView.card)

  // ── Audio Interaction ──
  if (audioEl) {
    playBtn.addEventListener('click', () => {
      if (audioEl.paused) audioEl.play(); else audioEl.pause()
    })

    audioEl.addEventListener('play', () => {
      playIcon.style.display = 'none'
      pauseIcon.style.display = 'block'
      playhead.classList.add('active')
    })
    audioEl.addEventListener('pause', () => {
      playIcon.style.display = 'block'
      pauseIcon.style.display = 'none'
    })
    audioEl.addEventListener('ended', () => {
      playIcon.style.display = 'block'
      pauseIcon.style.display = 'none'
      playhead.classList.remove('active')
    })

    speedBtn.addEventListener('click', () => {
      speedIdx = (speedIdx + 1) % speeds.length
      audioEl.playbackRate = speeds[speedIdx]
      speedBtn.textContent = speeds[speedIdx] + '×'
    })

    clickOverlay.addEventListener('click', e => {
      const rect = tlTracksArea.getBoundingClientRect()
      const waveW = rect.width - TL_LABEL_W
      const x = e.clientX - rect.left - TL_LABEL_W
      const ratio = Math.max(0, Math.min(1, x / waveW))
      seekAudio(ratio * totalDur)
    })

    let rafId = null
    function updatePlayhead() {
      const t = audioEl.currentTime
      const areaRect = tlTracksArea.getBoundingClientRect()
      const waveW = areaRect.width - TL_LABEL_W
      const px = TL_LABEL_W + (t / totalDur) * waveW
      playhead.style.left = px + 'px'
      timeDisplay.textContent = `${fmt(t)} / ${fmt(totalDur)}`

      userSpeechEls.forEach(el => {
        const inRange = t >= parseFloat(el.dataset.start) && t <= parseFloat(el.dataset.end)
        el.classList.toggle('active', inRange)
      })
      agentSpeechEls.forEach(el => {
        const inRange = t >= parseFloat(el.dataset.start) && t <= parseFloat(el.dataset.end)
        el.classList.toggle('active', inRange)
      })

      let activeRow = null
      tickView.rows.forEach(row => {
        const rowTime = parseFloat(row.dataset.startTime)
        if (rowTime <= t) activeRow = row
      })
      const prev = tickView.tbody.querySelector('tr.vv-playing')
      if (prev && prev !== activeRow) prev.classList.remove('vv-playing')
      if (activeRow && activeRow !== prev) activeRow.classList.add('vv-playing')
    }

    function tick() {
      updatePlayhead()
      rafId = requestAnimationFrame(tick)
    }

    audioEl.addEventListener('play', () => { rafId = requestAnimationFrame(tick) })
    audioEl.addEventListener('pause', () => { cancelAnimationFrame(rafId); updatePlayhead() })
    audioEl.addEventListener('ended', () => { cancelAnimationFrame(rafId); updatePlayhead() })
    audioEl.addEventListener('seeked', updatePlayhead)

    const onKeyDown = e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.code === 'Space') { e.preventDefault(); audioEl.paused ? audioEl.play() : audioEl.pause() }
      if (e.code === 'ArrowLeft') { e.preventDefault(); audioEl.currentTime = Math.max(0, audioEl.currentTime - 5) }
      if (e.code === 'ArrowRight') { e.preventDefault(); audioEl.currentTime = Math.min(audioEl.duration || totalDur, audioEl.currentTime + 5) }
    }
    document.addEventListener('keydown', onKeyDown)
    cleanupFns.push(() => document.removeEventListener('keydown', onKeyDown))
    cleanupFns.push(() => cancelAnimationFrame(rafId))
  } else {
    playBtn.disabled = true
    playBtn.title = 'No audio available'
    speedBtn.disabled = true
  }

  // Tooltip interaction on timeline
  tlTracksArea.addEventListener('mousemove', e => {
    const target = e.target.closest('.vv-tl-speech, .vv-effect')
    if (target && target.title) {
      tooltip.innerHTML = escapeHtml(target.title).replace(/\n/g, '<br>')
      tooltip.classList.add('show')
      tooltip.style.left = (e.clientX + 12) + 'px'
      tooltip.style.top = (e.clientY - 35) + 'px'
    } else {
      tooltip.classList.remove('show')
    }
  })
  tlTracksArea.addEventListener('mouseleave', () => tooltip.classList.remove('show'))

  // Click speech block to seek
  tlTracksArea.addEventListener('click', e => {
    const speech = e.target.closest('.vv-tl-speech')
    if (speech && audioEl) {
      seekAudio(parseFloat(speech.dataset.start))
    }
  })

  // Init
  requestAnimationFrame(initTimeline)

  let resizeTimer
  const onResize = () => {
    clearTimeout(resizeTimer)
    resizeTimer = setTimeout(initTimeline, 200)
  }
  window.addEventListener('resize', onResize)

  return () => {
    window.removeEventListener('resize', onResize)
    clearTimeout(resizeTimer)
    cleanupFns.forEach(fn => fn())
    if (audioEl) { audioEl.pause(); audioEl.src = '' }
    container.innerHTML = ''
  }
}
