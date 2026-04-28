const state = {
  root: null,
  runs: [],
  selectedPath: null,
  results: null,
  selectedSimId: null,
  detail: null,
  raw: null,
  prompts: null,
  activeTab: "conversation",
};

const el = {
  rootLabel: document.getElementById("rootLabel"),
  runSelect: document.getElementById("runSelect"),
  refreshButton: document.getElementById("refreshButton"),
  runSummary: document.getElementById("runSummary"),
  searchInput: document.getElementById("searchInput"),
  statusFilter: document.getElementById("statusFilter"),
  simulationList: document.getElementById("simulationList"),
  emptyState: document.getElementById("emptyState"),
  detailPane: document.getElementById("detailPane"),
  detailTitle: document.getElementById("detailTitle"),
  detailSubtitle: document.getElementById("detailSubtitle"),
  audioPlayer: document.getElementById("audioPlayer"),
  tabs: [...document.querySelectorAll(".tab")],
  panels: {
    conversation: document.getElementById("conversationTab"),
    ticks: document.getElementById("ticksTab"),
    task: document.getElementById("taskTab"),
    prompts: document.getElementById("promptsTab"),
    review: document.getElementById("reviewTab"),
    config: document.getElementById("configTab"),
    raw: document.getElementById("rawTab"),
  },
};

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function jsonBlock(value) {
  return `<div class="json-box"><pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre></div>`;
}

function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  }
  return url.toString();
}

async function fetchJson(path, params = {}) {
  const response = await fetch(api(path, params));
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

function statusClass(sim) {
  if (sim.success === true) return "pass";
  if (sim.success === false) return "fail";
  return "unknown";
}

function statusLabel(sim) {
  if (sim.success === true) return "pass";
  if (sim.success === false) return "fail";
  return "unknown";
}

function renderRunSelect() {
  el.runSelect.innerHTML = state.runs
    .map((run) => {
      const selected = run.path === state.selectedPath ? "selected" : "";
      return `<option value="${escapeHtml(run.path)}" ${selected}>${escapeHtml(run.name)}</option>`;
    })
    .join("");
}

function renderRunSummary() {
  const r = state.results;
  if (!r) {
    el.runSummary.innerHTML = "";
    return;
  }
  const info = r.info || {};
  const domain =
    (info.environment_info && info.environment_info.domain_name) ||
    "unknown domain";
  const agent =
    (info.agent_info && info.agent_info.implementation) || "unknown agent";
  const user =
    (info.user_info && info.user_info.implementation) || "unknown user";
  el.runSummary.innerHTML = `
    <h3>${escapeHtml(domain)}</h3>
    <div class="muted">${escapeHtml(agent)} / ${escapeHtml(user)}</div>
    <div class="metrics">
      <div class="metric"><b>${r.simulations_count}</b><span>simulations</span></div>
      <div class="metric"><b>${r.success_count}</b><span>passed</span></div>
      <div class="metric"><b>${r.failure_count}</b><span>failed</span></div>
    </div>
  `;
}

function filteredSimulations() {
  const query = el.searchInput.value.trim().toLowerCase();
  const filter = el.statusFilter.value;
  const simulations = state.results && state.results.simulations;
  return (simulations || []).filter((sim) => {
    const trial = sim.trial == null ? "" : sim.trial;
    const haystack = `${sim.task_id} ${sim.id} ${trial}`.toLowerCase();
    if (query && !haystack.includes(query)) return false;
    if (filter === "passed" && sim.success !== true) return false;
    if (filter === "failed" && sim.success !== false) return false;
    if (filter === "unknown" && sim.success !== null) return false;
    if (filter === "audio" && !sim.has_audio) return false;
    return true;
  });
}

function renderSimulationList() {
  const sims = filteredSimulations();
  if (!sims.length) {
    el.simulationList.innerHTML = `<div class="empty-state">No simulations match the filters.</div>`;
    return;
  }
  el.simulationList.innerHTML = sims
    .map((sim) => {
      const active = sim.id === state.selectedSimId ? "active" : "";
      const audio = sim.has_audio ? " audio" : "";
      const cost = sim.agent_cost == null ? "" : ` agent $${Number(sim.agent_cost).toFixed(4)}`;
      const trial = sim.trial == null ? "-" : sim.trial;
      const termination = sim.termination_reason == null ? "-" : sim.termination_reason;
      return `
        <button class="sim-row ${active}" data-sim-id="${escapeHtml(sim.id)}" type="button">
          <div>
            <div class="sim-title">Task ${escapeHtml(sim.task_id)}</div>
            <div class="sim-meta">trial ${escapeHtml(trial)} / ${escapeHtml(termination)}${audio}${escapeHtml(cost)}</div>
          </div>
          <span class="badge ${statusClass(sim)}">${statusLabel(sim)}</span>
        </button>
      `;
    })
    .join("");
}

function renderToolCalls(toolCalls) {
  if (!toolCalls || !toolCalls.length) return "";
  return toolCalls
    .map((tc) => `<div class="tool-box"><pre>${escapeHtml(JSON.stringify(tc, null, 2))}</pre></div>`)
    .join("");
}

function hasTimelineColumn(rows, key) {
  return rows.some((row) => row[key] != null && String(row[key]).trim() !== "");
}

function canSeekTimelineCell(row, key, value) {
  return (
    ["agent", "user", "user_transcript"].includes(key) &&
    value !== "-" &&
    typeof row.start_seconds === "number" &&
    el.audioPlayer.src
  );
}

function renderTimelineCell(row, col) {
  const value = row[col.key] == null || row[col.key] === "" ? "-" : row[col.key];
  const content = escapeHtml(value);
  if (canSeekTimelineCell(row, col.key, value)) {
    return `
      <td class="${col.cls}">
        <button class="timeline-seek" data-seek="${escapeHtml(row.start_seconds)}" type="button">
          <pre>${content}</pre>
        </button>
      </td>
    `;
  }
  return `<td class="${col.cls}"><pre>${content}</pre></td>`;
}

function seekTimelineAudio(seconds) {
  if (!Number.isFinite(seconds) || !el.audioPlayer.src) return;
  el.audioPlayer.currentTime = Math.max(0, seconds);
  const playPromise = el.audioPlayer.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
}

function renderTimeline() {
  const rows = (state.detail && state.detail.timeline) || [];
  if (!rows.length) {
    return false;
  }

  const columnDefs = [
    { key: "ticks", label: "Ticks", cls: "timeline-ticks", required: true },
    { key: "time", label: "Time", cls: "timeline-time", required: true },
    { key: "agent", label: "Agent", cls: "timeline-agent", required: true },
    { key: "user", label: "User", cls: "timeline-user", required: true },
    { key: "user_transcript", label: "Transcript", cls: "timeline-user" },
    { key: "tools", label: "Tools", cls: "timeline-tools", required: true },
    { key: "agent_turn", label: "Agent Turn", cls: "timeline-turn" },
    { key: "user_turn", label: "User Turn", cls: "timeline-turn", required: true },
  ];
  const columns = columnDefs.filter((col) => col.required || hasTimelineColumn(rows, col.key));

  el.panels.conversation.innerHTML = `
    <div class="timeline-table">
      <table>
        <thead>
          <tr>
            ${columns.map((col) => `<th class="${col.cls}">${escapeHtml(col.label)}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${rows
            .map((row) => `
              <tr>
                ${columns
                  .map((col) => renderTimelineCell(row, col))
                  .join("")}
              </tr>
            `)
            .join("")}
        </tbody>
      </table>
    </div>
  `;
  return true;
}

function renderMessages() {
  if (renderTimeline()) return;

  const messages = (state.detail && state.detail.messages) || [];
  if (!messages.length) {
    el.panels.conversation.innerHTML = `<div class="empty-state">No messages found.</div>`;
    return;
  }
  el.panels.conversation.innerHTML = messages
    .map((msg, index) => {
      const role = msg.role || msg.message_type || "message";
      const content = msg.content || "";
      const meta = [
        msg.timestamp,
        msg.turn_idx != null ? `turn ${msg.turn_idx}` : null,
        msg.chunk_id != null ? `chunk ${msg.chunk_id}` : null,
        msg.contains_speech === false ? "silent" : null,
        msg.is_final_chunk === false ? "streaming" : null,
      ]
        .filter(Boolean)
        .join(" / ");
      return `
        <article class="message">
          <div>
            <div class="role ${escapeHtml(role)}">${escapeHtml(role)}</div>
            <div class="message-meta">#${index + 1}</div>
          </div>
          <div>
            <div class="message-body">${escapeHtml(content)}</div>
            ${renderToolCalls(msg.tool_calls)}
            ${meta ? `<div class="message-meta">${escapeHtml(meta)}</div>` : ""}
          </div>
        </article>
      `;
    })
    .join("");
}

function previewChunk(chunk) {
  if (!chunk) return "";
  if (chunk.preview) return chunk.preview;
  if (chunk.content) return chunk.content;
  if (chunk.tool_calls) return JSON.stringify(chunk.tool_calls);
  if (chunk.is_audio) return "[audio chunk]";
  return "";
}

function renderTicks() {
  const ticks = (state.detail && state.detail.ticks) || [];
  if (!ticks.length) {
    el.panels.ticks.innerHTML = `<div class="empty-state">No expanded ticks stored for this simulation.</div>`;
    return;
  }
  el.panels.ticks.innerHTML = `
    <div class="tick-table">
      <table>
        <thead>
          <tr>
            <th>Tick</th>
            <th>User</th>
            <th>Assistant</th>
            <th>Tools</th>
            <th>Timing</th>
          </tr>
        </thead>
        <tbody>
          ${ticks
            .map((tick) => {
              const tools = [
                ...(tick.agent_tool_calls || []).map((tc) => `assistant: ${tc.name}`),
                ...(tick.user_tool_calls || []).map((tc) => `user: ${tc.name}`),
                ...(tick.agent_tool_results || []).map(() => "agent tool result"),
                ...(tick.user_tool_results || []).map(() => "user tool result"),
              ].join("\\n");
              const timing = [
                tick.tick_duration_seconds != null ? `${tick.tick_duration_seconds}s sim` : null,
                tick.wall_clock_duration_seconds != null
                  ? `${Number(tick.wall_clock_duration_seconds).toFixed(3)}s wall`
                  : null,
              ]
                .filter(Boolean)
                .join("\\n");
              return `
                <tr>
                  <td>${escapeHtml(tick.tick_id)}</td>
                  <td>${escapeHtml(previewChunk(tick.user_chunk) || tick.user_transcript || "")}</td>
                  <td>${escapeHtml(previewChunk(tick.agent_chunk))}</td>
                  <td><pre>${escapeHtml(tools)}</pre></td>
                  <td><pre>${escapeHtml(timing)}</pre></td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderTask() {
  const task = state.detail && state.detail.task;
  if (!task) {
    el.panels.task.innerHTML = `<div class="empty-state">No task data found.</div>`;
    return;
  }
  el.panels.task.innerHTML = `
    <div class="two-col">
      <section>
        <h3>User Scenario</h3>
        ${jsonBlock(task.user_scenario || {})}
      </section>
      <section>
        <h3>Evaluation Criteria</h3>
        ${jsonBlock(task.evaluation_criteria || {})}
      </section>
    </div>
    <h3>Description</h3>
    ${jsonBlock(task.description || {})}
    <h3>Initial State</h3>
    ${jsonBlock(task.initial_state || {})}
  `;
}

function renderReview() {
  const detail = state.detail || {};
  el.panels.review.innerHTML = `
    <div class="two-col">
      <section>
        <h3>Reward</h3>
        ${jsonBlock(detail.reward_info || {})}
      </section>
      <section>
        <h3>Authentication</h3>
        ${jsonBlock(detail.auth_classification || {})}
      </section>
    </div>
    <h3>Full Review</h3>
    ${jsonBlock(detail.review || {})}
    <h3>User-Only Review</h3>
    ${jsonBlock(detail.user_only_review || {})}
    <h3>Hallucination Check</h3>
    ${jsonBlock(detail.hallucination_check || {})}
  `;
}

function renderConfig() {
  const detail = state.detail || {};
  el.panels.config.innerHTML = `
    <div class="two-col">
      <section>
        <h3>Run Info</h3>
        ${jsonBlock((state.results && state.results.info) || {})}
      </section>
      <section>
        <h3>Simulation Info</h3>
        ${jsonBlock(detail.info || {})}
      </section>
    </div>
    <h3>Speech Environment</h3>
    ${jsonBlock(detail.speech_environment || {})}
    <h3>Effect Timeline</h3>
    ${jsonBlock(detail.effect_timeline || {})}
  `;
}

function promptBlock(text) {
  if (!text) return `<div class="empty-state">Not available.</div>`;
  return `<div class="prompt-block"><pre>${escapeHtml(text)}</pre></div>`;
}

function toolDefsHtml(tools) {
  if (!tools || !Object.keys(tools).length)
    return `<div class="empty-state">Not available.</div>`;
  return Object.values(tools)
    .map(
      (t) => `
        <div class="tool-entry">
          <div class="tool-name">${escapeHtml(t.name)}</div>
          <div class="tool-doc"><pre>${escapeHtml(t.doc || "")}</pre></div>
          ${t.params ? `<div class="tool-schema"><pre>${escapeHtml(JSON.stringify(t.params, null, 2))}</pre></div>` : ""}
        </div>`,
    )
    .join("");
}

async function renderPrompts() {
  if (!state.selectedSimId) return;
  if (!state.prompts) {
    el.panels.prompts.innerHTML = `<div class="empty-state">Loading prompts...</div>`;
    try {
      state.prompts = await fetchJson(
        `/api/prompts/${encodeURIComponent(state.selectedSimId)}`,
        { path: state.selectedPath },
      );
    } catch (err) {
      el.panels.prompts.innerHTML = `<div class="empty-state">Failed to load prompts: ${escapeHtml(err.message)}</div>`;
      return;
    }
  }
  const p = state.prompts;
  el.panels.prompts.innerHTML = `
    <h3>Agent — Instruction</h3>
    ${promptBlock(p.agent_instruction)}
    <h3>Agent — Domain Policy</h3>
    ${promptBlock(p.agent_policy)}
    <h3>Agent — Tools</h3>
    <div class="tool-list">${toolDefsHtml(p.agent_tools)}</div>
    <h3>User — Global Guidelines</h3>
    ${promptBlock(p.user_guidelines)}
    <h3>User — Task Scenario</h3>
    ${promptBlock(p.user_scenario ? JSON.stringify(p.user_scenario, null, 2) : null)}
    ${p.user_tools ? `<h3>User — Tools</h3><div class="tool-list">${toolDefsHtml(p.user_tools)}</div>` : ""}
  `;
}

async function renderRaw() {
  if (!state.selectedSimId) return;
  if (!state.raw) {
    el.panels.raw.innerHTML = `<div class="empty-state">Loading raw simulation JSON...</div>`;
    state.raw = await fetchJson(`/api/raw/${encodeURIComponent(state.selectedSimId)}`, {
      path: state.selectedPath,
    });
  }
  el.panels.raw.innerHTML = jsonBlock(state.raw);
}

function renderDetailShell() {
  const summary = state.detail && state.detail.summary;
  if (!summary) return;
  el.emptyState.classList.add("hidden");
  el.detailPane.classList.remove("hidden");
  el.detailTitle.textContent = `Task ${summary.task_id}`;
  const trial = summary.trial == null ? "-" : summary.trial;
  const termination =
    summary.termination_reason == null ? "-" : summary.termination_reason;
  el.detailSubtitle.textContent = `simulation ${summary.id} / trial ${trial} / ${termination}`;

  if (state.detail.audio_url) {
    el.audioPlayer.src = api(state.detail.audio_url, { path: state.selectedPath });
    el.audioPlayer.classList.remove("hidden");
  } else {
    el.audioPlayer.removeAttribute("src");
    el.audioPlayer.classList.add("hidden");
  }
}

async function renderActiveTab() {
  for (const tab of el.tabs) {
    const active = tab.dataset.tab === state.activeTab;
    tab.classList.toggle("active", active);
  }
  for (const [name, panel] of Object.entries(el.panels)) {
    panel.classList.toggle("hidden", name !== state.activeTab);
  }
  if (state.activeTab === "conversation") renderMessages();
  if (state.activeTab === "ticks") renderTicks();
  if (state.activeTab === "task") renderTask();
  if (state.activeTab === "prompts") await renderPrompts();
  if (state.activeTab === "review") renderReview();
  if (state.activeTab === "config") renderConfig();
  if (state.activeTab === "raw") await renderRaw();
}

async function loadSimulation(simId) {
  state.selectedSimId = simId;
  state.raw = null;
  state.prompts = null;
  renderSimulationList();
  el.detailPane.classList.add("hidden");
  el.emptyState.classList.remove("hidden");
  el.emptyState.textContent = "Loading simulation...";
  state.detail = await fetchJson(`/api/simulations/${encodeURIComponent(simId)}`, {
    path: state.selectedPath,
  });
  renderDetailShell();
  await renderActiveTab();
}

async function loadResults(path) {
  state.selectedPath = path;
  state.selectedSimId = null;
  state.detail = null;
  state.raw = null;
  renderRunSelect();
  el.rootLabel.textContent = `Root: ${state.root}`;
  el.runSummary.innerHTML = `<div class="muted">Loading run...</div>`;
  el.simulationList.innerHTML = "";
  el.detailPane.classList.add("hidden");
  el.emptyState.classList.remove("hidden");
  el.emptyState.textContent = "Select a simulation to inspect the conversation, ticks, task, and audio.";
  state.results = await fetchJson("/api/results", { path });
  renderRunSummary();
  renderSimulationList();
}

async function init() {
  const initial = await fetchJson("/api/state");
  state.root = initial.root;
  state.runs = initial.runs || [];
  state.selectedPath = initial.default_path;
  renderRunSelect();
  if (!state.runs.length) {
    el.rootLabel.textContent = `No results found under ${state.root}`;
    return;
  }
  await loadResults(state.selectedPath);
}

el.runSelect.addEventListener("change", (event) => loadResults(event.target.value));
el.refreshButton.addEventListener("click", () => loadResults(state.selectedPath));
el.searchInput.addEventListener("input", renderSimulationList);
el.statusFilter.addEventListener("change", renderSimulationList);
el.simulationList.addEventListener("click", (event) => {
  const row = event.target.closest("[data-sim-id]");
  if (row) loadSimulation(row.dataset.simId);
});
el.panels.conversation.addEventListener("click", (event) => {
  const seekButton = event.target.closest("[data-seek]");
  if (seekButton) seekTimelineAudio(Number(seekButton.dataset.seek));
});
for (const tab of el.tabs) {
  tab.addEventListener("click", async () => {
    state.activeTab = tab.dataset.tab;
    await renderActiveTab();
  });
}

init().catch((error) => {
  console.error(error);
  el.rootLabel.textContent = "Failed to load results";
  el.emptyState.textContent = error.message;
});
