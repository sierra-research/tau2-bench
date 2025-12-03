const simulationInput = document.getElementById("simulation-input");
const judgeInput = document.getElementById("judge-input");
const clearBtn = document.getElementById("clear-btn");
const datasetSelect = document.getElementById("dataset-select");
const simulationListEl = document.getElementById("simulation-list");
const simulationDetailsEl = document.getElementById("simulation-details");
const judgeDetailsEl = document.getElementById("judge-details");
const judgeFilterEl = document.getElementById("judge-filter");
const searchInput = document.getElementById("search");
const messageTemplate = document.getElementById("message-template");

const state = {
  simulations: [],
  tasksById: {},
  judgeRecords: [],
  recordsByTask: new Map(),
  selectedSimulationId: null,
  labelSet: new Set(),
};
let datasetsConfig = [];

async function initDatasets() {
  try {
    const url = new URL("./data/datasets.json", document.baseURI).toString();
    const res = await fetch(url);
    if (!res.ok) {
      console.warn("Failed to fetch datasets.json", res.status);
      return;
    }
    const json = await res.json();
    if (!Array.isArray(json)) return;
    datasetsConfig = json;
    json.forEach((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = dataset.label;
      datasetSelect.appendChild(option);
    });
  } catch (err) {
    console.warn("Failed to load datasets.json", err);
  }
}

simulationInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const data = await readJsonFile(file);
    ingestSimulationData(data);
    renderSimulationList();
    renderSimulationDetails();
  } catch (err) {
    alert(`Failed to read simulation file: ${err.message}`);
  }
});

judgeInput.addEventListener("change", async (event) => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  try {
    await ingestJudgeFiles(files);
    renderJudgeFilter();
    renderJudgeDetails();
  } catch (err) {
    alert(`Failed to read judge file: ${err.message}`);
  }
});

clearBtn.addEventListener("click", () => {
  clearState();
  renderSimulationList();
  renderSimulationDetails();
  renderJudgeFilter();
  renderJudgeDetails();
});

searchInput.addEventListener("input", () => renderSimulationList());
judgeFilterEl.addEventListener("change", () => renderJudgeDetails());
datasetSelect.addEventListener("change", async (event) => {
  const datasetId = event.target.value;
  if (!datasetId) {
    clearState();
    renderSimulationList();
    renderSimulationDetails();
    renderJudgeFilter();
    renderJudgeDetails();
    return;
  }
  const dataset = datasetsConfig.find((d) => d.id === datasetId);
  if (!dataset) return;
  try {
    await loadDataset(dataset);
  } catch (err) {
    alert(`Failed to load dataset: ${err.message}`);
  }
});

document.addEventListener("DOMContentLoaded", initDatasets);

function readJsonFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = reader.result;
        resolve(JSON.parse(text));
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function ingestSimulationData(data) {
  state.simulations = data.simulations || [];
  state.tasksById = {};
  for (const task of data.tasks || []) {
    if (task.id) state.tasksById[task.id] = task;
  }
  state.selectedSimulationId =
    state.simulations.length > 0 ? state.simulations[0].id : null;
}

async function ingestJudgeFiles(files) {
  resetJudgeState();

  for (const file of files) {
    const text = await readTextFile(file);
    const parsed = tryParseJudgePayloads(text);
    parsed.forEach((entry) => {
      addJudgeRecord(entry, file.name);
    });
  }
}

async function ingestJudgeFilesFromUrls(urls) {
  resetJudgeState();
  for (const url of urls) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch ${url}`);
    }
    const text = await res.text();
    const parsed = tryParseJudgePayloads(text);
    parsed.forEach((entry) => addJudgeRecord(entry, url));
  }
}

function tryParseJudgePayloads(text) {
  text = text.trim();
  if (!text) return [];
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed;
    if (Array.isArray(parsed.entries)) return parsed.entries;
    return [parsed];
  } catch (_) {
    return text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  }
}

function addJudgeRecord(raw, sourceName) {
  const normalized = normalizeJudgeRecord(raw, sourceName);
  if (!normalized.taskId) return;

  state.judgeRecords.push(normalized);
  state.labelSet.add(normalized.label);

  if (!state.recordsByTask.has(normalized.taskId)) {
    state.recordsByTask.set(normalized.taskId, []);
  }
  state.recordsByTask.get(normalized.taskId).push(normalized);
}

function normalizeJudgeRecord(entry, sourceName) {
  const simId =
    entry.simulation_id ||
    entry.simulationId ||
    entry.simulationID ||
    null;
  const rawResponse = entry.judge_response || entry.judgeResponse || "";
  const prompt =
    entry.judge_prompt ||
    entry.prompt ||
    entry.judgePrompt ||
    "";
  const systemPrompt =
    entry.judge_system_prompt ||
    entry.judgeSystemPrompt ||
    "";
  const model = entry.judge_model || entry.judgeModel || entry.model || "";
  const temperature =
    entry.judge_temperature ??
    entry.judgeTemperature ??
    entry.temperature ??
    null;
  const parsedResponse = parseJudgeResponse(rawResponse);
  const tau2Success =
    entry.tau2_success ??
    entry.tau2Success ??
    (typeof entry.tau2_reward === "number"
      ? Math.abs(entry.tau2_reward - 1) < 1e-6
      : null);

  const verdictRaw =
    entry.judge_verdict ||
    parsedResponse.verdict ||
    "unknown";
  const verdictInfo = normalizeVerdict(verdictRaw);

  return {
    simulationId: simId,
    taskId: entry.task_id || entry.taskId || null,
    label: entry.label || "default",
    tau2Reward: entry.tau2_reward ?? entry.reward ?? null,
    tau2Success,
    verdictRaw,
    verdictLabel: verdictInfo.label,
    verdictBool: verdictInfo.value,
    prompt,
    rawResponse,
    verdict: verdictRaw,
    justification:
      entry.judge_justification ||
      parsedResponse.justification ||
      "",
    issues:
      entry.judge_issues ||
      parsedResponse.issues ||
      [],
    source: sourceName || entry.source || "unknown",
    model,
    systemPrompt,
    temperature,
  };
}

function parseJudgeResponse(text) {
  if (!text) return { verdict: "", justification: "", issues: [] };
  try {
    const parsed = JSON.parse(text);
    return {
      verdict: parsed.verdict || "",
      justification: parsed.justification || "",
      issues: Array.isArray(parsed.issues) ? parsed.issues : [],
    };
  } catch (_) {
    return { verdict: "", justification: "", issues: [] };
  }
}

function normalizeVerdict(value) {
  if (!value) return { label: "UNKNOWN", value: null };
  const lower = value.toLowerCase();
  if (
    lower.includes("pass") ||
    lower.includes("satisfied") ||
    lower.includes("success") ||
    lower.includes("resolved")
  ) {
    return { label: "PASS", value: true };
  }
  if (
    lower.includes("fail") ||
    lower.includes("unsatisfied") ||
    lower.includes("error") ||
    lower.includes("rejected")
  ) {
    return { label: "FAIL", value: false };
  }
  return { label: value.toUpperCase(), value: null };
}

function renderSimulationList() {
  simulationListEl.innerHTML = "";
  const searchTerm = searchInput.value.trim().toLowerCase();

  const items = state.simulations.filter((sim) => {
    const matchId = sim.id?.toLowerCase().includes(searchTerm);
    const matchTask = sim.task_id
      ?.toLowerCase()
      .includes(searchTerm);
    return matchId || matchTask;
  });

  if (!items.length) {
    simulationListEl.innerHTML =
      '<p class="empty-state">No simulations loaded.</p>';
    return;
  }

  for (const sim of items) {
    const el = document.createElement("div");
    el.className = "simulation-item";
    if (sim.id === state.selectedSimulationId) el.classList.add("active");
    el.innerHTML = `
      <div class="badge">${sim.task_id || "unknown task"}</div>
      <div>ID: ${sim.id}</div>
      <div class="meta-line">
        Reward: ${sim.reward_info?.reward ?? "?"} ·
        Trial: ${sim.trial ?? "?"}
      </div>
    `;
    el.addEventListener("click", () => {
      state.selectedSimulationId = sim.id;
      renderSimulationList();
      renderSimulationDetails();
      renderJudgeDetails();
    });
    simulationListEl.appendChild(el);
  }
}

function renderSimulationDetails() {
  simulationDetailsEl.innerHTML = "";
  const sim = state.simulations.find(
    (s) => s.id === state.selectedSimulationId
  );

  if (!sim) {
    simulationDetailsEl.innerHTML =
      '<p class="empty-state">Select a simulation to view details.</p>';
    return;
  }

  const task = state.tasksById[sim.task_id] || {};
  const info = document.createElement("div");
  info.className = "simulation-meta";
  info.innerHTML = `
    <table class="meta-table">
      <tbody>
        <tr><th>Simulation ID</th><td>${sim.id}</td></tr>
        <tr><th>Task ID</th><td>${sim.task_id || "n/a"}</td></tr>
        <tr><th>Reward</th><td>${sim.reward_info?.reward ?? "?"}</td></tr>
        <tr><th>Termination</th><td>${sim.termination_reason || "n/a"}</td></tr>
        <tr><th>Agent Cost</th><td>${sim.agent_cost ?? "?"}</td></tr>
        <tr><th>User Cost</th><td>${sim.user_cost ?? "?"}</td></tr>
      </tbody>
    </table>
    <div class="scenario">
      <h3>User Scenario</h3>
      <p>${task.user_scenario?.instructions?.reason_for_call || "Reason not provided."}</p>
    </div>
  `;

  const messagesContainer = document.createElement("div");
  messagesContainer.className = "messages";
  for (const message of sim.messages || []) {
    const node = messageTemplate.content.cloneNode(true);
    node.querySelector(".role").textContent = message.role;
    node.querySelector(".timestamp").textContent =
      message.timestamp || "";
    const contentEl = node.querySelector(".content");
    const toolCalls = message.tool_calls || [];
    const baseContent = message.content || "";
    if (toolCalls.length) {
      const toolInfo = toolCalls
        .map(
          (call) =>
            `[tool] ${call.requestor || "assistant"} -> ${
              call.name
            } ${JSON.stringify(call.arguments)}`
        )
        .join("\n");
      contentEl.textContent = [baseContent, toolInfo]
        .filter(Boolean)
        .join("\n\n");
    } else {
      contentEl.textContent = baseContent;
    }
    messagesContainer.appendChild(node);
  }

  simulationDetailsEl.appendChild(info);
  simulationDetailsEl.appendChild(messagesContainer);
}

function renderJudgeFilter() {
  judgeFilterEl.innerHTML = "";
  const option = document.createElement("option");
  option.value = "all";
  option.textContent = "All Labels";
  judgeFilterEl.appendChild(option);

  Array.from(state.labelSet)
    .sort()
    .forEach((label) => {
      const opt = document.createElement("option");
      opt.value = label;
      opt.textContent = label;
      judgeFilterEl.appendChild(opt);
    });
}

function renderJudgeDetails() {
  judgeDetailsEl.innerHTML = "";
  const simId = state.selectedSimulationId;
  if (!simId) {
    judgeDetailsEl.innerHTML =
      '<p class="empty-state">Select a simulation to view judge outputs.</p>';
    return;
  }

  const sim = state.simulations.find((s) => s.id === simId);
  if (!sim) {
    judgeDetailsEl.innerHTML =
      '<p class="empty-state">Select a simulation to view judge outputs.</p>';
    return;
  }

  const taskId = sim.task_id;
  const records = taskId ? state.recordsByTask.get(taskId) || [] : [];
  const filter = judgeFilterEl.value;
  const filtered =
    filter === "all"
      ? records
      : records.filter((r) => r.label === filter);

  if (!records.length) {
    judgeDetailsEl.innerHTML =
      '<p class="empty-state">No judge outputs loaded for this task.</p>';
    return;
  }

  if (!filtered.length) {
    judgeDetailsEl.innerHTML =
      '<p class="empty-state">No outputs match the current label filter.</p>';
    return;
  }

  for (const record of filtered) {
    const card = document.createElement("article");
    const judgeVerdict = record.verdictLabel || "UNKNOWN";
    const tau2Verdict =
      record.tau2Success === null
        ? "UNKNOWN"
        : record.tau2Success
        ? "PASS"
        : "FAIL";
    const agreement =
      record.tau2Success === null
        ? null
        : record.tau2Success === record.verdictBool;
    let borderClass = "neutral";
    if (agreement === null) {
      borderClass = "neutral";
    } else if (agreement) {
      borderClass = "agree";
    } else {
      borderClass = "disagree";
    }
    card.dataset.agreement = borderClass;
    card.className = "judge-card";
    card.innerHTML = `
      <header>
        <strong>${record.label}</strong>
        <span class="badge">${judgeVerdict}</span>
      </header>
      <p><strong>Source:</strong> ${record.source}</p>
      <p class="agreement ${
        agreement === null ? "" : agreement ? "agree" : "disagree"
      }">
        <strong>TAU2:</strong> ${tau2Verdict}
        · <strong>Judge:</strong> ${judgeVerdict}
        ${
          agreement === null
            ? ""
            : agreement
            ? "<span aria-label='agree'>✅</span>"
            : "<span aria-label='disagree'>⚠️</span>"
        }
      </p>
      ${
        record.model
          ? `<p><strong>Judge Model:</strong> ${record.model}${
              record.temperature !== null
                ? ` (T=${record.temperature})`
                : ""
            }</p>`
          : ""
      }
      <p><strong>Reward:</strong> ${record.tau2Reward ?? "?"}</p>
      <p><strong>Justification:</strong> ${record.justification || "—"}</p>
      ${
        record.issues && record.issues.length
          ? `<p><strong>Issues:</strong> ${record.issues.join(", ")}</p>`
          : ""
      }
      ${
        record.prompt
          ? `<details><summary>Prompt</summary><pre>${escapeHtml(
              record.prompt
            )}</pre></details>`
          : ""
      }
      ${
        record.systemPrompt
          ? `<details><summary>System Prompt</summary><pre>${escapeHtml(
              record.systemPrompt
            )}</pre></details>`
          : ""
      }
      ${
        record.rawResponse
          ? `<details><summary>Raw Judge Response</summary><pre>${escapeHtml(
              record.rawResponse
            )}</pre></details>`
          : ""
      }
    `;
    judgeDetailsEl.appendChild(card);
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function clearState() {
  state.simulations = [];
  state.tasksById = {};
  resetJudgeState();
  state.selectedSimulationId = null;
  simulationInput.value = "";
  judgeInput.value = "";
  searchInput.value = "";
  if (datasetSelect) {
    datasetSelect.value = "";
  }
}

function resetJudgeState() {
  state.judgeRecords = [];
  state.recordsByTask = new Map();
  state.labelSet = new Set();
}

async function loadDataset(dataset) {
  clearState();
  datasetSelect.value = dataset.id;
  const simRes = await fetch(
    new URL(dataset.simulation, document.baseURI).toString()
  );
  if (!simRes.ok) throw new Error(`Failed to fetch ${dataset.simulation}`);
  const simData = await simRes.json();
  ingestSimulationData(simData);
  renderSimulationList();
  renderSimulationDetails();

  if (dataset.judge_files && dataset.judge_files.length > 0) {
    const urls = dataset.judge_files.map((url) =>
      new URL(url, document.baseURI).toString()
    );
    await ingestJudgeFilesFromUrls(urls);
  } else {
    resetJudgeState();
  }
  renderJudgeFilter();
  renderJudgeDetails();
}


