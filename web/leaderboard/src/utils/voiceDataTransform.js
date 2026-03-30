/**
 * Transform raw SimulationRun JSON into the SIM_DATA format expected by the voice viewer.
 *
 * Ports the Python functions from export_viewer.py:
 *   _extract_speech_segments, _extract_effect_events, _extract_tick_groups, _build_sim_data
 */

const DEFAULT_TICK_DURATION_S = 0.2

function extractSpeechSegments(ticks, tickDur) {
  const userSegments = []
  const agentSegments = []
  let userStart = null
  let userTextParts = []
  let agentStart = null
  let agentTextParts = []

  if (!ticks || !ticks.length) return { user: [], agent: [] }

  for (const t of ticks) {
    const tickTime = t.tick_id * tickDur
    const userSpeaking = t.user_chunk?.contains_speech && t.user_chunk?.content
    const agentSpeaking = t.agent_chunk?.contains_speech && t.agent_chunk?.content

    if (userSpeaking) {
      if (userStart === null) userStart = tickTime
      userTextParts.push(t.user_chunk.content)
    } else if (userStart !== null) {
      userSegments.push({
        start: Math.round(userStart * 1000) / 1000,
        end: Math.round(tickTime * 1000) / 1000,
        text: userTextParts.join('').trim(),
      })
      userStart = null
      userTextParts = []
    }

    if (agentSpeaking) {
      if (agentStart === null) agentStart = tickTime
      agentTextParts.push(t.agent_chunk.content)
    } else if (agentStart !== null) {
      agentSegments.push({
        start: Math.round(agentStart * 1000) / 1000,
        end: Math.round(tickTime * 1000) / 1000,
        text: agentTextParts.join('').trim(),
      })
      agentStart = null
      agentTextParts = []
    }
  }

  const totalTime = ticks.length * tickDur
  if (userStart !== null) {
    userSegments.push({
      start: Math.round(userStart * 1000) / 1000,
      end: Math.round(totalTime * 1000) / 1000,
      text: userTextParts.join('').trim(),
    })
  }
  if (agentStart !== null) {
    agentSegments.push({
      start: Math.round(agentStart * 1000) / 1000,
      end: Math.round(totalTime * 1000) / 1000,
      text: agentTextParts.join('').trim(),
    })
  }

  return { user: userSegments, agent: agentSegments }
}

function extractEffectEvents(sim) {
  if (!sim.effect_timeline?.events) return []
  return sim.effect_timeline.events.map(e => ({
    type: e.effect_type,
    start_ms: e.start_ms,
    end_ms: e.end_ms,
    participant: e.participant,
    params: e.params,
  }))
}

function getOverlappingEffects(effectTimeline, startMs, endMs) {
  if (!effectTimeline?.events) return []
  return effectTimeline.events
    .filter(e => e.start_ms < endMs && (e.end_ms ?? Infinity) > startMs)
    .map(e => ({
      type: e.effect_type,
      startMs: e.start_ms,
      endMs: e.end_ms,
      participant: e.participant,
      params: e.params,
    }))
}

function getTickPattern(tick) {
  if (tick.agent_tool_calls?.length || tick.agent_tool_results?.length) return '__tool__'
  if (tick.user_tool_calls?.length || tick.user_tool_results?.length) return '__tool__'

  let tta = null
  if (tick.user_chunk?.turn_taking_action) tta = tick.user_chunk.turn_taking_action.action
  else if (tick.agent_chunk?.turn_taking_action) tta = tick.agent_chunk.turn_taking_action.action

  if (tta) {
    const norm = tta.split(':')[0].trim().toLowerCase()
    if (norm === 'generate_message' || norm === 'keep_talking') return 'active_speech'
    return norm
  }

  const hasAgent = !!(tick.agent_chunk?.content && tick.agent_chunk?.contains_speech)
  const hasUser = !!(tick.user_chunk?.content && tick.user_chunk?.contains_speech)
  if (!hasAgent && !hasUser) return null
  return 'active_speech'
}

function formatToolCall(tc) {
  const name = tc.function?.name || tc.name || 'unknown'
  let args = tc.function?.arguments || tc.arguments || '{}'
  if (typeof args === 'string') {
    try { args = JSON.parse(args) } catch { /* keep as string */ }
  }
  if (typeof args === 'object') {
    const parts = Object.entries(args).map(([k, v]) => {
      const val = typeof v === 'string' ? `"${v}"` : JSON.stringify(v)
      return `${k}=${val}`
    })
    return `${name}(${parts.join(', ')})`
  }
  return `${name}(${args})`
}

function formatToolResult(tr) {
  const content = tr.content || ''
  try {
    const parsed = JSON.parse(content)
    return JSON.stringify(parsed, null, 2).slice(0, 500)
  } catch {
    return String(content).slice(0, 500)
  }
}

function extractTickGroups(sim, tickDur) {
  if (!sim.ticks?.length) return []

  const ticks = sim.ticks
  const groups = []
  let i = 0

  while (i < ticks.length) {
    const tick = ticks[i]
    const pattern = getTickPattern(tick)
    const startTick = tick.tick_id
    const groupTicks = [tick]

    if (pattern === '__tool__') {
      groups.push({ start: startTick, end: startTick, ticks: groupTicks })
      i++
      continue
    }

    let lastPattern = pattern
    let j = i + 1
    while (j < ticks.length) {
      const nextTick = ticks[j]
      const np = getTickPattern(nextTick)
      if (np === '__tool__') break
      if (np === null) { groupTicks.push(nextTick); j++; continue }
      if (lastPattern === null) { lastPattern = np; groupTicks.push(nextTick); j++; continue }
      if (np !== lastPattern) break
      groupTicks.push(nextTick)
      j++
    }

    const endTick = ticks[j - 1].tick_id
    groups.push({ start: startTick, end: endTick, ticks: groupTicks })
    i = j
  }

  const rows = []
  for (const g of groups) {
    let agentText = ''
    let userText = ''
    const agentCalls = []
    const agentResults = []
    const userCalls = []
    const userResults = []

    for (const tick of g.ticks) {
      if (tick.agent_chunk?.content) agentText += tick.agent_chunk.content
      if (tick.user_chunk?.content) userText += tick.user_chunk.content
      if (tick.agent_tool_calls) agentCalls.push(...tick.agent_tool_calls)
      if (tick.agent_tool_results) agentResults.push(...tick.agent_tool_results)
      if (tick.user_tool_calls) userCalls.push(...tick.user_tool_calls)
      if (tick.user_tool_results) userResults.push(...tick.user_tool_results)
    }

    if (!agentText && !userText && !agentCalls.length && !agentResults.length &&
        !userCalls.length && !userResults.length) continue

    const timeS = Math.round(g.start * tickDur * 1000) / 1000
    const tickDurMs = Math.round(tickDur * 1000)
    const rowStartMs = g.start * tickDurMs
    const rowEndMs = (g.end + 1) * tickDurMs
    const effects = getOverlappingEffects(sim.effect_timeline, rowStartMs, rowEndMs)

    rows.push({
      tickStart: g.start,
      tickEnd: g.end,
      timeS,
      agentText: agentText.trim(),
      userText: userText.trim(),
      agentCalls: agentCalls.map(formatToolCall),
      agentResults: agentResults.map(formatToolResult),
      userCalls: userCalls.map(formatToolCall),
      userResults: userResults.map(formatToolResult),
      effects,
    })
  }

  return rows
}

/**
 * Build SIM_DATA from a raw SimulationRun JSON + optional Results metadata.
 *
 * @param {Object} sim - Raw simulation JSON (from sim_<id>.json)
 * @param {Object} opts - Additional metadata
 * @param {number} opts.tickDur - Tick duration in seconds
 * @param {string} opts.domain - Domain name
 * @param {string} opts.agentModel - Agent model name
 * @param {string} opts.agentProvider - Agent provider
 * @param {Object} opts.taskInfo - Task information (reason, knownInfo, etc.)
 * @returns {Object} SIM_DATA for the viewer
 */
export function buildSimData(sim, opts = {}) {
  const tickDur = opts.tickDur || DEFAULT_TICK_DURATION_S
  const totalDuration = sim.ticks?.length
    ? Math.round(sim.ticks.length * tickDur * 1000) / 1000
    : sim.duration || 0

  const speech = extractSpeechSegments(sim.ticks, tickDur)
  const effects = extractEffectEvents(sim)
  const tickRows = extractTickGroups(sim, tickDur)

  const envInfo = {}
  if (sim.speech_environment) {
    const se = sim.speech_environment
    envInfo.complexity = se.complexity || 'unknown'
    envInfo.backgroundNoise = se.background_noise_file || ''
    envInfo.environment = se.environment || ''
    envInfo.personaName = se.persona_name || ''
    envInfo.telephonyEnabled = se.telephony_enabled || false
  }

  let reward = null
  let rewardClass = 'unknown'
  if (sim.reward_info) {
    reward = sim.reward_info.reward
    rewardClass = reward > 0 ? 'success' : 'failure'
  }

  return {
    simId: sim.id,
    taskId: String(sim.task_id),
    trial: sim.trial || 0,
    mode: sim.mode || 'full_duplex',
    duration: Math.round((sim.duration || 0) * 100) / 100,
    totalDuration,
    terminationReason: sim.termination_reason || 'unknown',
    reward,
    rewardClass,
    tickDuration: tickDur,
    numTicks: sim.ticks?.length || 0,
    domain: opts.domain || '',
    agentModel: opts.agentModel || '',
    agentProvider: opts.agentProvider || '',
    userModel: opts.userModel || '',
    speechEnvironment: envInfo,
    taskInfo: opts.taskInfo || {},
    speech,
    effects,
    tickRows,
    audioTaps: [],
  }
}
