(function () {
  'use strict';

  const D = SIM_DATA;
  const totalDur = D.totalDuration;
  const hasTaps = !!(D.audioTaps && D.audioTaps.length);

  // ── Helpers ──────────────────────────────
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'className') el.className = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'style' && typeof v === 'object')
        Object.assign(el.style, v);
      else el.setAttribute(k, v);
    });
    children.flat(Infinity).forEach(c => {
      if (c == null) return;
      el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return el;
  }

  function pct(t) { return (t / totalDur * 100) + '%'; }

  function fmt(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m + ':' + String(sec).padStart(2, '0');
  }

  function fmtMs(ms) {
    const s = ms / 1000;
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    const milli = Math.floor(ms % 1000);
    return `${m}:${String(sec).padStart(2, '0')}.${String(milli).padStart(3, '0')}`;
  }

  function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function seededRand(seed) {
    let s = seed;
    return function () {
      s = (s * 16807 + 0) % 2147483647;
      return s / 2147483647;
    };
  }

  // ── State ────────────────────────────────
  let currentView = 'overview';
  const speeds = [1, 1.5, 2, 0.5];
  let speedIdx = 0;
  let tapsExpanded = false;

  // ── Build DOM ────────────────────────────
  const app = document.getElementById('app');

  // Header
  const rewardBadge = D.reward !== null
    ? h('span', { className: `badge badge-${D.rewardClass}` },
        h('span', { className: `dot dot-${D.rewardClass}` }),
        D.rewardClass === 'success' ? 'Pass' : 'Fail')
    : null;

  const viewBtns = {};
  function makeViewBtn(id, label) {
    const btn = h('button', {
      className: 'view-btn' + (id === currentView ? ' active' : ''),
      onClick: () => switchView(id),
    }, label);
    viewBtns[id] = btn;
    return btn;
  }

  const header = h('div', { className: 'header' },
    h('div', { className: 'header-inner' },
      h('div', { className: 'header-title' },
        h('h1', null, h('span', { className: 'tau' }, 'τ'), `-bench · Task ${D.taskId}`),
        rewardBadge,
      ),
      h('div', { className: 'header-meta' },
        h('span', { className: 'badge badge-domain' }, D.domain),
        D.agentProvider ? h('span', { className: 'badge badge-provider' }, D.agentProvider) : null,
        D.agentModel ? h('span', { className: 'badge badge-model' }, D.agentModel.split('/').pop()) : null,
        h('div', { className: 'view-switcher' },
          makeViewBtn('overview', '🔍 Overview'),
          makeViewBtn('timeline', '📊 Timeline'),
          makeViewBtn('ticks', '📝 Conversation'),
          ...(hasTaps ? [makeViewBtn('taps', '🎛️ Audio Tracks')] : []),
        ),
      ),
    ),
  );
  app.appendChild(header);

  // Main
  const main = h('div', { className: 'main' });
  app.appendChild(main);

  // ── Info Card ────────────────────────────
  const infoItems = [
    ['Task ID', D.taskId],
    ['Duration', `${D.duration}s (${fmt(D.duration)})`],
    ['Termination', D.terminationReason],
    ['Mode', D.mode],
    ['Ticks', `${D.numTicks} × ${D.tickDuration}s`],
    ['Reward', D.reward !== null ? D.reward : 'N/A'],
  ];

  if (D.speechEnvironment.complexity)
    infoItems.push(['Complexity', D.speechEnvironment.complexity]);
  if (D.speechEnvironment.backgroundNoise)
    infoItems.push(['Background Noise', D.speechEnvironment.backgroundNoise]);
  if (D.speechEnvironment.personaName)
    infoItems.push(['Persona', D.speechEnvironment.personaName]);

  const infoGrid = h('div', { className: 'info-grid' },
    ...infoItems.map(([label, value]) =>
      h('div', { className: 'info-cell' },
        h('div', { className: 'info-label' }, label),
        h('div', { className: 'info-value' }, String(value)),
      )
    ),
  );

  // Task info
  const taskCards = [];
  if (D.taskInfo && D.taskInfo.reason) {
    const taskItems = [
      ['Reason for Call', D.taskInfo.reason],
      ['Task Instructions', D.taskInfo.taskInstructions],
      ['Known Info', D.taskInfo.knownInfo],
      ['Unknown Info', D.taskInfo.unknownInfo],
    ].filter(([_, v]) => v);

    taskCards.push(
      h('div', { className: 'card task-info-card' },
        h('div', { className: 'card-header' },
          h('span', { className: 'card-header-icon' }, '📋'),
          'User Task',
        ),
        h('div', { className: 'info-grid' },
          ...taskItems.map(([label, value]) =>
            h('div', { className: 'info-cell' },
              h('div', { className: 'info-label' }, label),
              h('div', { className: 'info-value' }, h('pre', null, value)),
            )
          ),
        ),
      )
    );
  }

  const simInfoCard = h('div', { className: 'card' },
    h('div', { className: 'card-header' },
      h('span', { className: 'card-header-icon' }, '⚙️'),
      'Simulation Info',
    ),
    infoGrid,
  );

  taskCards.forEach(c => main.appendChild(c));
  main.appendChild(simInfoCard);

  // ── View Panels ─────────────────────────
  const overviewPanel = h('div', { className: 'view-panel active', id: 'view-overview' });
  const timelinePanel = h('div', { className: 'view-panel', id: 'view-timeline' });
  const tickPanel = h('div', { className: 'view-panel', id: 'view-ticks' });
  const tapsPanel = h('div', { className: 'view-panel', id: 'view-taps' });
  main.appendChild(overviewPanel);
  main.appendChild(timelinePanel);
  main.appendChild(tickPanel);
  main.appendChild(tapsPanel);

  // ── Timeline Card Construction ──────────
  // Controls (Unicode icons, shared transport style with mixer)
  const playIcon = h('span', { className: 'transport-btn-icon' }, '▶');
  const pauseIcon = h('span', { className: 'transport-btn-icon', style: { display: 'none' } }, '⏸');

  const playBtn = h('button', { className: 'transport-btn transport-btn-play', title: 'Play / Pause' }, playIcon, pauseIcon);
  const timeDisplay = h('span', { className: 'time-display' }, `0:00 / ${fmt(totalDur)}`);
  const speedBtn = h('button', { className: 'speed-btn', title: 'Playback speed' }, '1×');

  const trackUser = h('div', { className: 'tl-track-wrap tl-track-user' });
  const trackAgent = h('div', { className: 'tl-track-wrap tl-track-agent' });
  const effectsLane = h('div', { className: 'mx-annotation-lane tl-effects-lane' });
  const playhead = h('div', { className: 'mx-playhead-line' });
  const clickOverlay = h('div', { className: 'tl-click-overlay' });
  const tlRuler = h('div', { className: 'mx-ruler' });
  const legendEl = h('div', { className: 'tl-legend' });

  const envDesc = D.speechEnvironment.backgroundNoise
    ? (D.speechEnvironment.environment || 'with noise')
    : null;

  const tlTracksArea = h('div', { className: 'tl-tracks-area', id: 'tlTracks' },
    h('div', { className: 'mx-track-row' },
      h('div', { className: 'mx-track-label' },
        h('div', { className: 'mx-track-accent', style: { background: 'var(--user-color)' } }),
        h('div', { className: 'mx-track-info' },
          h('div', { className: 'mx-track-name' }, 'User'),
          envDesc ? h('div', { className: 'mx-track-desc' }, envDesc) : null,
        ),
      ),
      trackUser,
    ),
    h('div', { className: 'mx-track-row mx-annotation-row' },
      h('div', { className: 'mx-track-label' },
        h('div', { className: 'mx-track-accent', style: { background: 'var(--sierra-gray-dark)' } }),
        h('div', { className: 'mx-track-info' },
          h('div', { className: 'mx-track-name' }, 'Effects'),
        ),
      ),
      effectsLane,
    ),
    h('div', { className: 'mx-track-row' },
      h('div', { className: 'mx-track-label' },
        h('div', { className: 'mx-track-accent', style: { background: 'var(--agent-color)' } }),
        h('div', { className: 'mx-track-info' },
          h('div', { className: 'mx-track-name' }, 'Agent'),
        ),
      ),
      trackAgent,
    ),
    playhead,
    clickOverlay,
  );

  // Expand button (for expanding timeline to show audio tracks)
  const expandBtnIcon = h('span', { className: 'tl-expand-icon' }, '▶');
  const expandBtn = h('button', {
    className: 'tl-expand-btn',
    title: 'Expand to show all audio track waveforms',
    onClick: () => toggleExpandTaps(),
  }, expandBtnIcon, h('span', null, ' Audio Tracks'));
  const expandSection = h('div', { className: 'tl-expand-section' });

  const timelineCardHeader = h('div', { className: 'card-header' },
    h('span', { className: 'card-header-icon' }, '📊'),
    'Speech Activity Timeline',
    ...(hasTaps ? [h('div', { style: { flex: '1' } }), expandBtn] : []),
  );

  const TL_LABEL_W = 150;

  const timelineCard = h('div', { className: 'card timeline-card' },
    timelineCardHeader,
    h('div', { className: 'timeline-controls' }, playBtn, timeDisplay, speedBtn),
    tlTracksArea,
    h('div', { className: 'mx-ruler-row' },
      h('div', { className: 'mx-track-label mx-ruler-spacer' }),
      tlRuler,
    ),
    legendEl,
    expandSection,
  );

  // Start in overview panel
  overviewPanel.appendChild(timelineCard);

  // Tooltip
  const tooltip = h('div', { className: 'tl-tooltip' });
  document.body.appendChild(tooltip);

  // ── Tick View ────────────────────────────
  const hasEffects = D.tickRows.some(r => r.effects && r.effects.length > 0);

  function buildTickTable() {
    const headerCells = [
      h('th', { className: 'tick-col' }, 'Tick'),
      h('th', { className: 'time-col' }, 'Time'),
      h('th', null, 'Agent Speech'),
      h('th', null, 'Agent Tools'),
      h('th', null, 'User Speech'),
      h('th', null, 'User Tools'),
    ];
    if (hasEffects) headerCells.push(h('th', { className: 'effects-col-header' }, 'Effects'));
    const thead = h('thead', null, h('tr', null, ...headerCells));

    const rows = D.tickRows.map(row => {
      const tickRange = row.tickStart === row.tickEnd
        ? String(row.tickStart)
        : `${row.tickStart}–${row.tickEnd}`;
      const timeStr = fmtMs(row.timeS * 1000);

      function buildToolsCell(calls, results) {
        if (!calls.length && !results.length) return h('td', { className: 'tool-col' }, h('span', { className: 'empty-cell' }, '—'));
        const items = [];
        calls.forEach(c => items.push(h('div', { className: 'tool-call-item' }, c)));
        results.forEach((r, i) => {
          const details = h('details', { className: 'tool-result-item' },
            h('summary', null, `Result ${i + 1}`),
            h('pre', null, r),
          );
          items.push(details);
        });
        return h('td', { className: 'tool-col' }, ...items);
      }

      const agentTd = h('td', { className: 'agent-col' });
      if (row.agentText) agentTd.textContent = row.agentText;
      else agentTd.appendChild(h('span', { className: 'empty-cell' }, '—'));

      const userTd = h('td', { className: 'user-col' });
      if (row.userText) userTd.textContent = row.userText;
      else userTd.appendChild(h('span', { className: 'empty-cell' }, '—'));

      const rowChildren = [
        h('td', { className: 'tick-col' }, tickRange),
        h('td', { className: 'time-col', title: 'Click to play from here', onClick: () => seekAudio(row.timeS) }, timeStr),
        agentTd,
        buildToolsCell(row.agentCalls, row.agentResults),
        userTd,
        buildToolsCell(row.userCalls, row.userResults),
      ];

      if (hasEffects) {
        const effectItems = (row.effects || []).map(e => {
          const s = (e.startMs / 1000).toFixed(1) + 's';
          const en = e.endMs != null ? (e.endMs / 1000).toFixed(1) + 's' : '...';
          let label = e.type.replace(/_/g, ' ');
          let detail = '';
          if (e.params) {
            if (e.params.duration_ms) detail = ` ${e.params.duration_ms}ms`;
            if (e.params.file) detail = ` ${e.params.file}`;
          }
          return h('div', { className: 'effect-item effect-' + e.type.replace(/_/g, '-') },
            h('span', { className: 'effect-label' }, label),
            h('span', { className: 'effect-time' }, ` (${s}–${en})${detail}`),
          );
        });
        const effectsTd = h('td', { className: 'effects-col' },
          effectItems.length > 0 ? effectItems : [h('span', { className: 'empty-cell' }, '—')],
        );
        rowChildren.push(effectsTd);
      }

      const tr = h('tr', { 'data-start-time': String(row.timeS), 'data-tick-start': String(row.tickStart) }, ...rowChildren);
      return tr;
    });

    const tbody = h('tbody', null, ...rows);
    const table = h('table', { className: 'tick-table' }, thead, tbody);
    const card = h('div', { className: 'card' },
      h('div', { className: 'card-header' },
        h('span', { className: 'card-header-icon' }, '📝'),
        `Conversation · ${D.tickRows.length} groups`,
      ),
      h('div', { className: 'tick-table-wrapper' }, table),
    );
    return { table, tbody, rows, card };
  }

  const tickView = buildTickTable();
  // Start in overview panel (below timeline)
  overviewPanel.appendChild(tickView.card);

  // ── Shared helpers for effect + tool annotation rows ──
  const effectMetaShared = {
    frame_drop:          { label: 'Frame Drops',        accent: 'var(--sierra-orange)' },
    burst_noise:         { label: 'Burst Noise',        accent: 'var(--sierra-purple)' },
    out_of_turn_speech:  { label: 'Out-of-Turn Speech', accent: 'var(--sierra-orange-mid)' },
    background_noise:    { label: 'Background Noise',   accent: 'var(--sierra-gray-dark)' },
    telephony:           { label: 'Telephony',          accent: 'var(--sierra-blue-mid)' },
  };

  function buildEffectRows(container, dur) {
    if (!D.effects || !D.effects.length) return;
    const byType = {};
    D.effects.forEach(e => { if (!byType[e.type]) byType[e.type] = []; byType[e.type].push(e); });
    const typeOrder = ['frame_drop', 'burst_noise', 'out_of_turn_speech', 'background_noise', 'telephony'];
    const activeTypes = typeOrder.filter(t => byType[t]);
    if (!activeTypes.length) return;

    container.appendChild(h('div', { className: 'mx-group-sep' },
      h('span', { className: 'mx-group-dot', style: { background: 'var(--sierra-gray-dark)' } }),
      h('span', null, 'Effects'),
    ));

    activeTypes.forEach(type => {
      const events = byType[type];
      const meta = effectMetaShared[type] || { label: type, accent: '#999' };
      const lane = h('div', { className: 'mx-annotation-lane' });
      events.forEach(e => {
        const startPct = (e.start_ms / 1000) / dur * 100;
        const endMs = e.end_ms != null ? e.end_ms : e.start_ms + 200;
        const widthPct = Math.max(0.15, (endMs - e.start_ms) / 1000 / dur * 100);
        const durMs = endMs - e.start_ms;
        const parts = [`${meta.label}`, `Time: ${(e.start_ms / 1000).toFixed(1)}s – ${(endMs / 1000).toFixed(1)}s (${durMs}ms)`];
        if (e.participant) parts.push(`Participant: ${e.participant}`);
        if (e.params) {
          if (e.params.duration_ms) parts.push(`Duration: ${e.params.duration_ms}ms`);
          if (e.params.file) parts.push(`File: ${e.params.file}`);
          if (e.params.text) parts.push(`Text: ${e.params.text}`);
          if (e.params.type) parts.push(`Type: ${e.params.type}`);
          if (e.params.snr_db != null) parts.push(`SNR: ${e.params.snr_db}dB`);
        }
        const cls = 'mx-effect mx-effect-' + type.replace(/_/g, '-');
        const marker = h('div', { className: cls, title: parts.join('\n') });
        marker.style.left = startPct + '%';
        marker.style.width = widthPct + '%';
        lane.appendChild(marker);
      });
      container.appendChild(h('div', { className: 'mx-track-row mx-annotation-row' },
        h('div', { className: 'mx-track-label' },
          h('div', { className: 'mx-track-accent', style: { background: meta.accent } }),
          h('div', { className: 'mx-track-info' },
            h('div', { className: 'mx-track-name' }, meta.label),
            h('div', { className: 'mx-track-desc' }, `${events.length} events`),
          ),
        ),
        lane,
      ));
    });
  }

  function buildToolCallRows(container, dur) {
    const agentToolEvents = [];
    const userToolEvents = [];
    D.tickRows.forEach(row => {
      if (row.agentCalls && row.agentCalls.length > 0)
        agentToolEvents.push({ timeS: row.timeS, tickStart: row.tickStart, tickEnd: row.tickEnd, names: row.agentCalls });
      if (row.userCalls && row.userCalls.length > 0)
        userToolEvents.push({ timeS: row.timeS, tickStart: row.tickStart, tickEnd: row.tickEnd, names: row.userCalls });
    });
    if (!agentToolEvents.length && !userToolEvents.length) return;

    container.appendChild(h('div', { className: 'mx-group-sep' },
      h('span', { className: 'mx-group-dot', style: { background: 'var(--sierra-orange-mid)' } }),
      h('span', null, 'Tool Calls'),
    ));

    [
      { label: 'Agent Tools', events: agentToolEvents, color: 'rgba(249, 98, 5, 0.7)' },
      { label: 'User Tools', events: userToolEvents, color: 'rgba(18, 108, 235, 0.7)' },
    ].forEach(({ label, events, color }) => {
      if (!events.length) return;
      const lane = h('div', { className: 'mx-annotation-lane' });
      events.forEach(c => {
        const startPct = (c.timeS / dur) * 100;
        const durS = (c.tickEnd - c.tickStart + 1) * D.tickDuration;
        const widthPct = Math.max(0.3, (durS / dur) * 100);
        c.names.forEach((name, i) => {
          const toolTitle = `${name}\nTime: ${c.timeS.toFixed(1)}s\nTicks: ${c.tickStart}–${c.tickEnd}`;
          const block = h('div', { className: 'mx-tool-block', title: toolTitle, style: { background: color } });
          block.style.left = startPct + '%';
          block.style.width = widthPct + '%';
          if (c.names.length > 1) block.style.top = (i * 14) + 'px';
          block.appendChild(h('span', { className: 'mx-tool-label' }, name.split('(')[0]));
          lane.appendChild(block);
        });
      });
      container.appendChild(h('div', { className: 'mx-track-row mx-annotation-row' },
        h('div', { className: 'mx-track-label' },
          h('div', { className: 'mx-track-accent', style: { background: color } }),
          h('div', { className: 'mx-track-info' },
            h('div', { className: 'mx-track-name' }, label),
            h('div', { className: 'mx-track-desc' }, `${events.length} calls`),
          ),
        ),
        lane,
      ));
    });
  }

  // ── Multitrack Mixer (Taps Tab) ─────────
  const mixerState = {
    tracks: [],
    audioEls: [],
    playing: false,
    duration: 0,
    rafId: null,
  };

  function buildMultitrackMixer() {
    const allTaps = D.audioTaps || [];
    if (!allTaps.length) return;

    const taps = allTaps;
    if (!taps.length) return;

    const maxDuration = Math.max(...taps.map(t => t.waveform?.duration || 0));
    mixerState.duration = maxDuration;

    // Transport controls (Unicode icons to match timeline play button)
    const mxPlayIcon = h('span', { className: 'transport-btn-icon' }, '▶');
    const mxPauseIcon = h('span', { className: 'transport-btn-icon', style: { display: 'none' } }, '⏸');

    const mxPlayBtn = h('button', { className: 'transport-btn transport-btn-play', title: 'Play / Pause all tracks' }, mxPlayIcon, mxPauseIcon);
    const mxStopBtn = h('button', { className: 'transport-btn transport-btn-stop', title: 'Stop and rewind' }, '⏹');
    const mxTimeDisplay = h('span', { className: 'mx-time' }, `0:00 / ${fmt(maxDuration)}`);

    const mxSoloAllBtn = h('button', { className: 'mx-util-btn', title: 'Unmute all', onClick: () => {
      mixerState.tracks.forEach(t => { t.muted = false; t.muteBtn.classList.remove('active'); t.audioEl.muted = false; });
    }}, 'Unmute All');

    const mxMuteAllBtn = h('button', { className: 'mx-util-btn', title: 'Mute all', onClick: () => {
      mixerState.tracks.forEach(t => { t.muted = true; t.muteBtn.classList.add('active'); t.audioEl.muted = true; });
    }}, 'Mute All');

    const transport = h('div', { className: 'mx-transport' },
      mxPlayBtn, mxStopBtn, mxTimeDisplay,
      h('div', { className: 'mx-transport-spacer' }),
      mxSoloAllBtn, mxMuteAllBtn,
    );

    // Track rows
    const tracksContainer = h('div', { className: 'mx-tracks' });
    const mxPlayheadLine = h('div', { className: 'mx-playhead-line' });

    // ── Build tracks: User audio → Effects → Agent audio → Tool calls ──
    const userTaps = taps.filter(t => t.participant === 'user');
    const agentTaps = taps.filter(t => t.participant === 'agent');

    function addAudioTapRows(tapList, groupLabel, groupColor) {
      if (!tapList.length) return;
      tracksContainer.appendChild(h('div', { className: 'mx-group-sep' },
        h('span', { className: 'mx-group-dot', style: { background: groupColor } }),
        h('span', null, groupLabel),
      ));

      tapList.forEach((tap, idx) => {
        const audioEl = h('audio', { preload: 'auto' },
          h('source', { src: tap.src, type: 'audio/wav' }),
        );
        document.body.appendChild(audioEl);

        const canvas = h('canvas', { className: 'mx-waveform-canvas', height: '60' });
        const waveformWrap = h('div', { className: 'mx-waveform-wrap' }, canvas);

        const muteBtn = h('button', { className: 'mx-mute-btn', title: 'Mute' }, 'M');
        const soloBtn = h('button', { className: 'mx-solo-btn', title: 'Solo' }, 'S');

        const trackInfo = {
          tap, idx, audioEl, canvas, muted: false, soloed: false, muteBtn, soloBtn,
        };
        mixerState.tracks.push(trackInfo);
        mixerState.audioEls.push(audioEl);

        muteBtn.addEventListener('click', () => {
          trackInfo.muted = !trackInfo.muted;
          muteBtn.classList.toggle('active', trackInfo.muted);
          updateSoloMuteState();
        });

        soloBtn.addEventListener('click', () => {
          trackInfo.soloed = !trackInfo.soloed;
          soloBtn.classList.toggle('active', trackInfo.soloed);
          updateSoloMuteState();
        });

        const accentColor = tap.participant === 'agent' ? 'var(--agent-color)' : 'var(--user-color)';
        tracksContainer.appendChild(h('div', { className: 'mx-track-row' },
          h('div', { className: 'mx-track-label' },
            h('div', { className: 'mx-track-accent', style: { background: accentColor } }),
            h('div', { className: 'mx-track-info' },
              h('div', { className: 'mx-track-name' }, tap.label),
              h('div', { className: 'mx-track-desc' }, tap.description),
            ),
            h('div', { className: 'mx-track-controls' }, muteBtn, soloBtn),
          ),
          waveformWrap,
        ));
      });
    }

    addAudioTapRows(userTaps, 'User', 'var(--user-color)');
    buildEffectRows(tracksContainer, maxDuration);
    addAudioTapRows(agentTaps, 'Agent', 'var(--agent-color)');
    buildToolCallRows(tracksContainer, maxDuration);

    const waveformArea = h('div', { className: 'mx-waveform-area' }, tracksContainer, mxPlayheadLine);

    // Time ruler
    const ruler = h('div', { className: 'mx-ruler' });
    const rulerStep = maxDuration <= 60 ? 5 : maxDuration <= 180 ? 10 : 20;
    for (let t = 0; t <= maxDuration; t += rulerStep) {
      const mark = h('div', { className: 'mx-ruler-mark' });
      mark.style.left = (t / maxDuration * 100) + '%';
      mark.dataset.time = t + 's';
      ruler.appendChild(mark);
    }

    const mixerCard = h('div', { className: 'card mx-card' },
      h('div', { className: 'card-header' },
        h('span', { className: 'card-header-icon' }, '🎛️'),
        `Multitrack Mixer · ${taps.length} tracks`,
      ),
      transport,
      h('div', { className: 'mx-ruler-row' },
        h('div', { className: 'mx-track-label mx-ruler-spacer' }),
        ruler,
      ),
      waveformArea,
    );
    tapsPanel.appendChild(mixerCard);

    // Draw waveforms
    mixerState.drawWaveform = drawWaveform;
    function drawWaveform(track) {
      const canvas = track.canvas;
      const wf = track.tap.waveform;
      if (!wf || !wf.mins || !wf.mins.length) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.parentElement.getBoundingClientRect();
      const w = rect.width || 600;
      const hh = 60;
      canvas.width = w * dpr;
      canvas.height = hh * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = hh + 'px';

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, hh);

      const mid = hh / 2;
      const color = track.tap.participant === 'agent' ? 'rgba(249, 98, 5, 0.6)' : 'rgba(18, 108, 235, 0.6)';
      const colorDim = track.tap.participant === 'agent' ? 'rgba(249, 98, 5, 0.15)' : 'rgba(18, 108, 235, 0.15)';
      ctx.fillStyle = track.muted ? colorDim : color;

      const trackDur = wf.duration || maxDuration;
      const scale = trackDur / maxDuration;
      const drawW = w * scale;
      const n = wf.mins.length;
      const barW = Math.max(1, drawW / n);
      for (let i = 0; i < n; i++) {
        const x = (i / n) * drawW;
        const minVal = wf.mins[i];
        const maxVal = wf.maxs[i];
        const top = mid - maxVal * mid;
        const bot = mid - minVal * mid;
        ctx.fillRect(x, top, barW + 0.5, Math.max(1, bot - top));
      }

      ctx.strokeStyle = track.muted ? 'rgba(0,0,0,0.05)' : 'rgba(0,0,0,0.1)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(w, mid);
      ctx.stroke();
    }

    mixerState.drawAllWaveforms = drawAllWaveforms;
    function drawAllWaveforms() {
      mixerState.tracks.forEach(drawWaveform);
    }

    // Solo/Mute logic
    function updateSoloMuteState() {
      const anySoloed = mixerState.tracks.some(t => t.soloed);
      mixerState.tracks.forEach(t => {
        if (anySoloed) {
          t.audioEl.muted = !t.soloed || t.muted;
        } else {
          t.audioEl.muted = t.muted;
        }
        const dimmed = (anySoloed && !t.soloed) || t.muted;
        t.canvas.parentElement.classList.toggle('mx-dimmed', dimmed);
      });
      drawAllWaveforms();
    }

    // Click waveform to seek
    tracksContainer.addEventListener('click', e => {
      const wrap = e.target.closest('.mx-waveform-wrap');
      if (!wrap) return;
      const rect = wrap.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      const seekTime = ratio * maxDuration;
      mixerState.audioEls.forEach(a => { a.currentTime = Math.max(0, Math.min(seekTime, a.duration || maxDuration)); });
      updateMixerPlayhead();
      if (!mixerState.playing) {
        mxPlayheadLine.classList.add('active');
      }
    });

    // Transport
    function playAll() {
      mixerState.playing = true;
      mxPlayIcon.style.display = 'none';
      mxPauseIcon.style.display = 'block';
      mxPlayheadLine.classList.add('active');
      mixerState.audioEls.forEach(a => a.play());
      mixerState.rafId = requestAnimationFrame(mixerTick);
    }

    function pauseAll() {
      mixerState.playing = false;
      mxPlayIcon.style.display = 'block';
      mxPauseIcon.style.display = 'none';
      mixerState.audioEls.forEach(a => a.pause());
      cancelAnimationFrame(mixerState.rafId);
    }

    function stopAll() {
      pauseAll();
      mixerState.audioEls.forEach(a => { a.currentTime = 0; });
      mxPlayheadLine.classList.remove('active');
      updateMixerPlayhead();
    }

    mxPlayBtn.addEventListener('click', () => {
      if (mixerState.playing) pauseAll(); else playAll();
    });
    mxStopBtn.addEventListener('click', stopAll);

    function updateMixerPlayhead() {
      const ref = mixerState.audioEls[0];
      if (!ref) return;
      const t = ref.currentTime;
      const areaEl = waveformArea;
      const areaRect = areaEl.getBoundingClientRect();
      const labelW = 220;
      const waveW = areaRect.width - labelW;
      const px = labelW + (t / maxDuration) * waveW;
      mxPlayheadLine.style.left = px + 'px';
      mxTimeDisplay.textContent = `${fmt(t)} / ${fmt(maxDuration)}`;
    }

    function mixerTick() {
      updateMixerPlayhead();
      const ref = mixerState.audioEls[0];
      if (ref && ref.ended) {
        pauseAll();
        return;
      }
      mixerState.rafId = requestAnimationFrame(mixerTick);
    }

    // Keyboard for mixer view
    document.addEventListener('keydown', e => {
      if (currentView !== 'taps') return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (mixerState.playing) pauseAll(); else playAll();
      }
      if (e.code === 'ArrowLeft') {
        e.preventDefault();
        const t = Math.max(0, (mixerState.audioEls[0]?.currentTime || 0) - 5);
        mixerState.audioEls.forEach(a => { a.currentTime = t; });
        updateMixerPlayhead();
      }
      if (e.code === 'ArrowRight') {
        e.preventDefault();
        const t = Math.min(maxDuration, (mixerState.audioEls[0]?.currentTime || 0) + 5);
        mixerState.audioEls.forEach(a => { a.currentTime = t; });
        updateMixerPlayhead();
      }
    });

    let mxResizeTimer;
    window.addEventListener('resize', () => {
      if (currentView !== 'taps') return;
      clearTimeout(mxResizeTimer);
      mxResizeTimer = setTimeout(drawAllWaveforms, 200);
    });

    // Tooltip for effects and tool blocks in the mixer
    waveformArea.addEventListener('mousemove', e => {
      const target = e.target.closest('.mx-effect, .mx-tool-block');
      if (target && target.title) {
        tooltip.innerHTML = escapeHtml(target.title).replace(/\n/g, '<br>');
        tooltip.classList.add('show');
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY - 35) + 'px';
      } else {
        tooltip.classList.remove('show');
      }
    });
    waveformArea.addEventListener('mouseleave', () => tooltip.classList.remove('show'));
  }

  buildMultitrackMixer();

  // ── Expanded Taps (inside timeline card) ──
  const expandState = {
    built: false,
    tracks: [],
    playheadLine: null,
    waveformArea: null,
    maxDuration: totalDur,
  };

  function toggleExpandTaps() {
    tapsExpanded = !tapsExpanded;
    expandSection.classList.toggle('active', tapsExpanded);
    expandBtnIcon.textContent = tapsExpanded ? '▼' : '▶';
    if (tapsExpanded && !expandState.built) {
      buildExpandedTaps();
      expandState.built = true;
    }
    if (tapsExpanded && expandState.tracks.length) {
      requestAnimationFrame(() => expandState.tracks.forEach(drawExpandedWaveform));
    }
  }

  function buildExpandedTaps() {
    const allTaps = D.audioTaps || [];
    if (!allTaps.length) return;

    const maxDur = Math.max(...allTaps.map(t => t.waveform?.duration || 0), totalDur);
    expandState.maxDuration = maxDur;

    // Time ruler
    const ruler = h('div', { className: 'mx-ruler' });
    const rulerStep = maxDur <= 60 ? 5 : maxDur <= 180 ? 10 : 20;
    for (let t = 0; t <= maxDur; t += rulerStep) {
      const mark = h('div', { className: 'mx-ruler-mark' });
      mark.style.left = (t / maxDur * 100) + '%';
      mark.dataset.time = t + 's';
      ruler.appendChild(mark);
    }

    const tracksContainer = h('div', { className: 'mx-tracks' });
    const exPlayheadLine = h('div', { className: 'mx-playhead-line' });
    expandState.playheadLine = exPlayheadLine;

    // Build expanded tracks: User audio → Effects → Agent audio → Tool calls
    const exUserTaps = allTaps.filter(t => t.participant === 'user');
    const exAgentTaps = allTaps.filter(t => t.participant === 'agent');

    function addExpandedTapRows(tapList, groupLabel, groupColor) {
      if (!tapList.length) return;
      tracksContainer.appendChild(h('div', { className: 'mx-group-sep' },
        h('span', { className: 'mx-group-dot', style: { background: groupColor } }),
        h('span', null, groupLabel),
      ));

      tapList.forEach(tap => {
        const canvas = h('canvas', { className: 'mx-waveform-canvas', height: '40' });
        const waveformWrap = h('div', { className: 'mx-waveform-wrap ex-waveform-wrap' }, canvas);
        const accentColor = tap.participant === 'agent' ? 'var(--agent-color)' : 'var(--user-color)';
        tracksContainer.appendChild(h('div', { className: 'mx-track-row' },
          h('div', { className: 'mx-track-label' },
            h('div', { className: 'mx-track-accent', style: { background: accentColor } }),
            h('div', { className: 'mx-track-info' },
              h('div', { className: 'mx-track-name' }, tap.label),
              h('div', { className: 'mx-track-desc' }, tap.description),
            ),
          ),
          waveformWrap,
        ));
        expandState.tracks.push({ tap, canvas, maxDuration: maxDur });
      });
    }

    addExpandedTapRows(exUserTaps, 'User', 'var(--user-color)');
    buildEffectRows(tracksContainer, maxDur);
    addExpandedTapRows(exAgentTaps, 'Agent', 'var(--agent-color)');
    buildToolCallRows(tracksContainer, maxDur);

    const waveformArea = h('div', { className: 'mx-waveform-area' }, tracksContainer, exPlayheadLine);
    expandState.waveformArea = waveformArea;

    expandSection.appendChild(
      h('div', { className: 'tl-expand-inner' },
        h('div', { className: 'mx-ruler-row' },
          h('div', { className: 'mx-track-label mx-ruler-spacer' }),
          ruler,
        ),
        waveformArea,
      )
    );

    // Click-to-seek in expanded waveforms (uses main audio)
    waveformArea.addEventListener('click', e => {
      const wrap = e.target.closest('.mx-waveform-wrap, .mx-annotation-lane');
      if (!wrap) return;
      const rect = wrap.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      const seekTime = ratio * maxDur;
      seekAudio(seekTime);
    });

    // Tooltip for expanded effects
    waveformArea.addEventListener('mousemove', e => {
      const target = e.target.closest('.mx-effect, .mx-tool-block');
      if (target && target.title) {
        tooltip.innerHTML = escapeHtml(target.title).replace(/\n/g, '<br>');
        tooltip.classList.add('show');
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY - 35) + 'px';
      } else {
        tooltip.classList.remove('show');
      }
    });
    waveformArea.addEventListener('mouseleave', () => tooltip.classList.remove('show'));
  }

  function drawExpandedWaveform(track) {
    const canvas = track.canvas;
    const wf = track.tap.waveform;
    if (!wf || !wf.mins || !wf.mins.length) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width || 600;
    const hh = 40;
    canvas.width = w * dpr;
    canvas.height = hh * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = hh + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, hh);

    const mid = hh / 2;
    const color = track.tap.participant === 'agent' ? 'rgba(249, 98, 5, 0.6)' : 'rgba(18, 108, 235, 0.6)';
    ctx.fillStyle = color;

    const trackDur = wf.duration || track.maxDuration;
    const scale = trackDur / track.maxDuration;
    const drawW = w * scale;
    const n = wf.mins.length;
    const barW = Math.max(1, drawW / n);
    for (let i = 0; i < n; i++) {
      const x = (i / n) * drawW;
      const minVal = wf.mins[i];
      const maxVal = wf.maxs[i];
      const top = mid - maxVal * mid;
      const bot = mid - minVal * mid;
      ctx.fillRect(x, top, barW + 0.5, Math.max(1, bot - top));
    }

    ctx.strokeStyle = 'rgba(0,0,0,0.1)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();
  }

  function updateExpandPlayhead() {
    if (!tapsExpanded || !expandState.playheadLine || !audioEl) return;
    const t = audioEl.currentTime;
    const maxDur = expandState.maxDuration;
    const areaEl = expandState.waveformArea;
    if (!areaEl) return;
    const areaRect = areaEl.getBoundingClientRect();
    const labelW = 150;
    const waveW = areaRect.width - labelW;
    const px = labelW + (t / maxDur) * waveW;
    expandState.playheadLine.style.left = px + 'px';
    expandState.playheadLine.classList.toggle('active', !audioEl.paused || t > 0);
  }

  // ── Audio Player ─────────────────────────
  let audioEl = null;
  if (AUDIO_SRC) {
    audioEl = h('audio', { id: 'mainAudio', controls: '', preload: 'metadata' },
      h('source', { src: AUDIO_SRC, type: 'audio/wav' }),
    );
    const player = h('div', { className: 'sticky-player' },
      audioEl,
      h('div', { className: 'shortcut-hint' }, 'Space = play/pause · ←→ = ±5s'),
    );
    document.body.appendChild(player);
  }

  // ── Render Timeline ──────────────────────
  function renderSpeechBlocks(segments, track, type) {
    const rand = seededRand(type === 'user' ? 42 : 137);
    const speechEls = [];
    segments.forEach(seg => {
      const el = h('div', { className: `tl-speech tl-speech-${type}` });
      el.style.left = pct(seg.start);
      el.style.width = ((seg.end - seg.start) / totalDur * 100) + '%';
      el.dataset.start = seg.start;
      el.dataset.end = seg.end;
      el.dataset.text = seg.text;
      el.title = seg.text.slice(0, 120) + (seg.text.length > 120 ? '…' : '');
      track.appendChild(el);

      const segDur = seg.end - seg.start;
      const barCount = Math.max(4, Math.round(segDur * 2.5));
      for (let i = 0; i < barCount; i++) {
        const bar = document.createElement('div');
        bar.className = `tl-bar tl-bar-${type}`;
        const height = 20 + rand() * 80;
        bar.style.height = height + '%';
        bar.style.left = (i / barCount * 100) + '%';
        bar.style.bottom = ((100 - height) / 2) + '%';
        el.appendChild(bar);
      }
      speechEls.push(el);
    });
    return speechEls;
  }

  function renderNoise() {
    const rand = seededRand(999);
    const w = trackUser.offsetWidth || 800;
    const spacing = 4;
    const n = Math.floor(w / spacing);
    for (let i = 0; i < n; i++) {
      const bar = document.createElement('div');
      bar.className = 'tl-noise-bar';
      bar.style.height = (3 + rand() * 12) + 'px';
      bar.style.left = (i * spacing) + 'px';
      trackUser.appendChild(bar);
    }
  }

  const effectIcons = {
    frame_drop: '',
    burst_noise: '⚡',
    out_of_turn_speech: '🗣',
    background_noise: '🔊',
    telephony: '📞',
  };

  function renderEffects() {
    D.effects.forEach(e => {
      const tSec = e.start_ms / 1000;
      const endMs = e.end_ms != null ? e.end_ms : e.start_ms + 200;
      const startPct = (tSec / totalDur * 100);
      const widthPct = Math.max(0.4, (endMs - e.start_ms) / 1000 / totalDur * 100);
      const cls = 'mx-effect mx-effect-' + e.type.replace(/_/g, '-');
      let title = e.type.replace(/_/g, ' ') + ` @ ${fmtMs(e.start_ms)}`;
      if (e.params?.duration_ms) title += ` (${e.params.duration_ms}ms)`;
      if (e.params?.text) title += ` "${e.params.text}"`;
      const icon = effectIcons[e.type] || '';
      const el = h('div', { className: cls, title },
        h('span', { className: 'mx-effect-icon' }, icon),
      );
      el.style.left = startPct + '%';
      el.style.width = widthPct + '%';
      effectsLane.appendChild(el);
    });
  }

  function renderXAxis() {
    const step = totalDur <= 60 ? 5 : totalDur <= 180 ? 10 : 20;
    for (let t = 0; t <= totalDur; t += step) {
      const mark = h('div', { className: 'mx-ruler-mark' });
      mark.style.left = (t / totalDur * 100) + '%';
      mark.dataset.time = t + 's';
      tlRuler.appendChild(mark);
    }
  }

  const effectLegendMeta = {
    frame_drop:         { label: 'Frame Drop',        cls: 'mx-effect-frame-drop',         icon: '⏸' },
    burst_noise:        { label: 'Burst Noise',        cls: 'mx-effect-burst-noise',        icon: '⚡' },
    out_of_turn_speech: { label: 'Out-of-Turn Speech', cls: 'mx-effect-out-of-turn-speech', icon: '🗣' },
    background_noise:   { label: 'Background Noise',   cls: 'mx-effect-background-noise',   icon: '🔊' },
    telephony:          { label: 'Telephony',           cls: 'mx-effect-telephony',          icon: '📞' },
  };

  function renderLegend() {
    const items = [
      { label: 'User Speech', html: '<span class="tl-legend-swatch"><span class="tl-legend-block tl-legend-block-user"></span></span>' },
      { label: 'Agent Speech', html: '<span class="tl-legend-swatch"><span class="tl-legend-block tl-legend-block-agent"></span></span>' },
    ];

    const activeTypes = new Set(D.effects.map(e => e.type));
    ['frame_drop', 'burst_noise', 'out_of_turn_speech', 'background_noise', 'telephony'].forEach(type => {
      if (!activeTypes.has(type)) return;
      const m = effectLegendMeta[type];
      items.push({
        label: m.label,
        html: `<span class="tl-legend-swatch"><span class="tl-legend-effect-swatch ${m.cls}">${m.icon}</span></span>`,
      });
    });

    items.forEach(item => {
      const el = document.createElement('div');
      el.className = 'tl-legend-item';
      el.innerHTML = item.html + '<span>' + item.label + '</span>';
      legendEl.appendChild(el);
    });
  }

  let userSpeechEls = [];
  let agentSpeechEls = [];

  function initTimeline() {
    trackUser.innerHTML = '';
    trackAgent.innerHTML = '';
    effectsLane.innerHTML = '';
    tlRuler.innerHTML = '';
    legendEl.innerHTML = '';

    if (D.speechEnvironment.backgroundNoise) renderNoise();
    userSpeechEls = renderSpeechBlocks(D.speech.user, trackUser, 'user');
    agentSpeechEls = renderSpeechBlocks(D.speech.agent, trackAgent, 'agent');
    renderEffects();
    renderXAxis();
    renderLegend();
    timeDisplay.textContent = `0:00 / ${fmt(totalDur)}`;
  }

  // ── Audio interaction ────────────────────
  function seekAudio(timeSec) {
    if (!audioEl) return;
    audioEl.currentTime = Math.max(0, Math.min(timeSec, audioEl.duration || totalDur));
    if (audioEl.paused) audioEl.play();
  }

  if (audioEl) {
    playBtn.addEventListener('click', () => {
      if (audioEl.paused) audioEl.play(); else audioEl.pause();
    });

    audioEl.addEventListener('play', () => {
      playIcon.style.display = 'none';
      pauseIcon.style.display = 'block';
      playhead.classList.add('active');
    });
    audioEl.addEventListener('pause', () => {
      playIcon.style.display = 'block';
      pauseIcon.style.display = 'none';
    });
    audioEl.addEventListener('ended', () => {
      playIcon.style.display = 'block';
      pauseIcon.style.display = 'none';
      playhead.classList.remove('active');
    });

    speedBtn.addEventListener('click', () => {
      speedIdx = (speedIdx + 1) % speeds.length;
      audioEl.playbackRate = speeds[speedIdx];
      speedBtn.textContent = speeds[speedIdx] + '×';
    });

    clickOverlay.addEventListener('click', e => {
      const rect = tlTracksArea.getBoundingClientRect();
      const waveW = rect.width - TL_LABEL_W;
      const x = e.clientX - rect.left - TL_LABEL_W;
      const ratio = Math.max(0, Math.min(1, x / waveW));
      seekAudio(ratio * totalDur);
    });

    // Playhead + active highlights
    let rafId = null;
    function updatePlayhead() {
      const t = audioEl.currentTime;
      const areaRect = tlTracksArea.getBoundingClientRect();
      const waveW = areaRect.width - TL_LABEL_W;
      const px = TL_LABEL_W + (t / totalDur) * waveW;
      playhead.style.left = px + 'px';
      timeDisplay.textContent = `${fmt(t)} / ${fmt(totalDur)}`;

      // Highlight active speech blocks
      userSpeechEls.forEach(el => {
        const inRange = t >= parseFloat(el.dataset.start) && t <= parseFloat(el.dataset.end);
        el.classList.toggle('active', inRange);
      });
      agentSpeechEls.forEach(el => {
        const inRange = t >= parseFloat(el.dataset.start) && t <= parseFloat(el.dataset.end);
        el.classList.toggle('active', inRange);
      });

      // Highlight active tick row (in ticks or overview mode)
      if (currentView === 'ticks' || currentView === 'overview') {
        let activeRow = null;
        tickView.rows.forEach(row => {
          const rowTime = parseFloat(row.dataset.startTime);
          if (rowTime <= t) activeRow = row;
        });
        const prev = tickView.tbody.querySelector('tr.playing');
        if (prev && prev !== activeRow) prev.classList.remove('playing');
        if (activeRow && activeRow !== prev) activeRow.classList.add('playing');
      }

      // Update expanded taps playhead
      updateExpandPlayhead();
    }

    function tick() {
      updatePlayhead();
      rafId = requestAnimationFrame(tick);
    }

    audioEl.addEventListener('play', () => { rafId = requestAnimationFrame(tick); });
    audioEl.addEventListener('pause', () => { cancelAnimationFrame(rafId); updatePlayhead(); });
    audioEl.addEventListener('ended', () => { cancelAnimationFrame(rafId); updatePlayhead(); });
    audioEl.addEventListener('seeked', () => { updatePlayhead(); });

    // Keyboard shortcuts (for timeline, ticks, and overview views; taps view has its own)
    document.addEventListener('keydown', e => {
      if (currentView === 'taps') return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.code === 'Space') { e.preventDefault(); audioEl.paused ? audioEl.play() : audioEl.pause(); }
      if (e.code === 'ArrowLeft') { e.preventDefault(); audioEl.currentTime = Math.max(0, audioEl.currentTime - 5); }
      if (e.code === 'ArrowRight') { e.preventDefault(); audioEl.currentTime = Math.min(audioEl.duration, audioEl.currentTime + 5); }
    });
  }

  // Tooltip on speech blocks and effects in timeline
  tlTracksArea.addEventListener('mousemove', e => {
    const target = e.target.closest('.tl-speech, .mx-effect');
    if (target && target.title) {
      tooltip.innerHTML = escapeHtml(target.title).replace(/\n/g, '<br>');
      tooltip.classList.add('show');
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 35) + 'px';
    } else {
      tooltip.classList.remove('show');
    }
  });
  tlTracksArea.addEventListener('mouseleave', () => tooltip.classList.remove('show'));

  // Click speech block to seek
  tlTracksArea.addEventListener('click', e => {
    const speech = e.target.closest('.tl-speech');
    if (speech && audioEl) {
      seekAudio(parseFloat(speech.dataset.start));
    }
  });

  // ── View Switching ───────────────────────
  function switchView(id) {
    currentView = id;
    Object.entries(viewBtns).forEach(([k, btn]) => btn.classList.toggle('active', k === id));

    ['overview', 'timeline', 'ticks', 'taps'].forEach(v => {
      const panel = document.getElementById('view-' + v);
      if (panel) panel.classList.remove('active');
    });

    if (id === 'overview') {
      overviewPanel.appendChild(timelineCard);
      overviewPanel.appendChild(tickView.card);
      overviewPanel.classList.add('active');
      expandBtn.style.display = '';
      requestAnimationFrame(initTimeline);
      if (tapsExpanded && expandState.tracks.length) {
        requestAnimationFrame(() => expandState.tracks.forEach(drawExpandedWaveform));
      }
    } else if (id === 'timeline') {
      timelinePanel.appendChild(timelineCard);
      timelinePanel.classList.add('active');
      expandBtn.style.display = 'none';
      expandSection.classList.remove('active');
      requestAnimationFrame(initTimeline);
    } else if (id === 'ticks') {
      tickPanel.appendChild(tickView.card);
      tickPanel.classList.add('active');
    } else if (id === 'taps') {
      tapsPanel.classList.add('active');
      if (mixerState.drawAllWaveforms) requestAnimationFrame(mixerState.drawAllWaveforms);
    }
  }

  // ── Init ─────────────────────────────────
  requestAnimationFrame(initTimeline);

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (currentView === 'timeline' || currentView === 'overview') initTimeline();
      if (tapsExpanded && expandState.tracks.length) {
        expandState.tracks.forEach(drawExpandedWaveform);
      }
    }, 200);
  });
})();
