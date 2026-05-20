from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau2.utils.utils import DATA_DIR


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_json_script(value: Any) -> str:
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def _simulation_result_candidates(row: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    simulation_save_to = row.get("simulation_save_to")
    if simulation_save_to:
        candidates.append(DATA_DIR / "simulations" / simulation_save_to / "results.json")

    raw_path = row.get("simulation_result_path")
    if raw_path:
        path = Path(raw_path)
        candidates.append(path)
        marker = "/data/simulations/"
        if marker in raw_path:
            candidates.append(
                DATA_DIR / "simulations" / raw_path.split(marker, maxsplit=1)[1]
            )

    return candidates


def _augment_simulation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    metrics_by_id: dict[str, dict[str, int]] = {}
    seen_paths: set[Path] = set()
    for row in rows:
        for candidate in _simulation_result_candidates(row):
            if candidate in seen_paths or not candidate.exists():
                continue
            seen_paths.add(candidate)
            payload = _read_json(candidate)
            for simulation in payload.get("simulations", []):
                simulation_id = simulation.get("id")
                if not simulation_id:
                    continue
                metrics_by_id[simulation_id] = {
                    "message_count": len(simulation.get("messages") or []),
                    "tick_count": len(simulation.get("ticks") or []),
                }

    if not metrics_by_id:
        return rows

    augmented: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        next_row.update(metrics_by_id.get(row.get("simulation_id"), {}))
        augmented.append(next_row)
    return augmented


def _file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _parse_run_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    text = path.read_text(encoding="utf-8", errors="ignore")
    status_matches = list(
        re.finditer(
            r"Status: (?P<complete>\d+)/(?P<total>\d+) complete\. Avg reward: (?P<reward>[^\n]+)",
            text,
        )
    )
    task_matches = list(
        re.finditer(
            r"(?P<started>\d+)/(?P<total>\d+) \(trial (?P<trial>\d+)/(?P<trials>\d+)\)\. Running task",
            text,
        )
    )
    running_matches = re.findall(r"\.0\((\d+)s\)", text[-12000:])

    latest_status: dict[str, Any] | None = None
    if status_matches:
        match = status_matches[-1]
        latest_status = {
            "complete": int(match.group("complete")),
            "total": int(match.group("total")),
            "avg_reward": match.group("reward").strip(),
        }

    latest_task: dict[str, Any] | None = None
    if task_matches:
        match = task_matches[-1]
        latest_task = {
            "started": int(match.group("started")),
            "total": int(match.group("total")),
            "trial": int(match.group("trial")),
            "trials": int(match.group("trials")),
        }

    return {
        "exists": True,
        "file": _file_state(path),
        "latest_status": latest_status,
        "latest_task": latest_task,
        "max_running_seconds": max((int(value) for value in running_matches), default=0),
        "tail": "\n".join(text.splitlines()[-24:]),
    }


def build_responses_report(exp_dir: str | Path) -> Path:
    exp_path = Path(exp_dir)
    if not exp_path.is_absolute():
        exp_path = DATA_DIR / "exp" / "responses" / exp_path
    exp_path.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(exp_path / "manifest.json")
    results = _read_csv_rows(exp_path / "results.csv")
    simulations = _augment_simulation_rows(_read_csv_rows(exp_path / "simulations.csv"))
    run_state = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "files": {
            "manifest": _file_state(exp_path / "manifest.json"),
            "results": _file_state(exp_path / "results.csv"),
            "simulations": _file_state(exp_path / "simulations.csv"),
            "run_log": _file_state(exp_path / "run.log"),
        },
        "log": _parse_run_log(exp_path / "run.log"),
    }

    html = _render_html(
        manifest=manifest,
        results=results,
        simulations=simulations,
        run_state=run_state,
        exp_name=manifest.get("exp_name") or exp_path.name,
    )
    out = exp_path / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def _render_html(
    *,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
    run_state: dict[str, Any],
    exp_name: str,
) -> str:
    manifest_json = _safe_json_script(manifest)
    results_json = _safe_json_script(results)
    simulations_json = _safe_json_script(simulations)
    run_state_json = _safe_json_script(run_state)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Responses OFAT Sweep Report</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f4f6f3;
        --panel: #ffffff;
        --ink: #17211d;
        --muted: #62716a;
        --line: #d9e0d8;
        --soft: #eef3ec;
        --soft-2: #f8faf7;
        --accent: #245f83;
        --green: #1f7855;
        --gold: #98711f;
        --red: #b4463d;
        --purple: #635b91;
        --shadow: 0 12px 28px rgba(23, 33, 29, 0.07);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.45;
      }}
      main {{
        max-width: 1480px;
        margin: 0 auto;
        padding: 30px 24px 56px;
      }}
      h1, h2, h3, p {{ margin: 0; }}
      header {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 24px;
        align-items: end;
        margin-bottom: 18px;
      }}
      .eyebrow {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
        text-transform: uppercase;
      }}
      h1 {{
        font-size: clamp(31px, 4vw, 54px);
        line-height: 1;
        letter-spacing: 0;
        max-width: 920px;
      }}
      .subtitle {{
        color: var(--muted);
        font-size: 16px;
        margin-top: 13px;
        max-width: 900px;
      }}
      .meta, .chips, .legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .meta {{ justify-content: flex-end; max-width: 560px; }}
      .pill, .chip, .legend-item {{
        border: 1px solid var(--line);
        border-radius: 999px;
        background: #fff;
        color: #394741;
        font-size: 13px;
        padding: 7px 11px;
        white-space: nowrap;
      }}
      .chip {{
        cursor: pointer;
        user-select: none;
      }}
      .chip input {{ margin-right: 6px; }}
      .chip input[type="radio"] {{ margin-right: 6px; }}
      .chip.active {{
        border-color: rgba(36, 95, 131, 0.45);
        background: #eaf2f6;
      }}
      .grid {{ display: grid; gap: 16px; }}
      .cards {{
        grid-template-columns: repeat(5, minmax(0, 1fr));
        margin-bottom: 16px;
      }}
      .card, .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
      }}
      .card {{ min-height: 108px; padding: 16px; }}
      .label {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }}
      .metric {{
        margin-top: 8px;
        font-size: 30px;
        line-height: 1;
        font-weight: 800;
        letter-spacing: 0;
      }}
      .hint {{
        margin-top: 8px;
        color: var(--muted);
        font-size: 13px;
      }}
      .muted {{ color: var(--muted); }}
      .status-grid {{
        display: grid;
        grid-template-columns: 1.2fr 1fr;
        gap: 16px;
      }}
      .progress-stack {{
        display: grid;
        gap: 12px;
      }}
      .progress-row {{
        display: grid;
        grid-template-columns: 132px minmax(0, 1fr) 92px;
        align-items: center;
        gap: 12px;
        font-size: 13px;
      }}
      .bar {{
        height: 12px;
        overflow: hidden;
        border-radius: 999px;
        background: #e8eee6;
        border: 1px solid var(--line);
      }}
      .bar > span {{
        display: block;
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, var(--accent), var(--green));
        border-radius: inherit;
      }}
      .status-copy {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--soft-2);
        padding: 12px;
        color: #34423c;
        font-size: 13px;
      }}
      .status-copy strong {{
        display: block;
        color: var(--ink);
        margin-bottom: 5px;
      }}
      .matrix {{
        display: grid;
        gap: 9px;
      }}
      .matrix-row {{
        display: grid;
        grid-template-columns: 135px minmax(0, 1fr);
        gap: 10px;
        align-items: stretch;
      }}
      .matrix-label {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
        padding-top: 7px;
        text-transform: uppercase;
      }}
      .matrix-cells {{
        display: grid;
        gap: 8px;
        grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      }}
      .matrix-cell {{
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent);
        border-radius: 8px;
        background: #fff;
        padding: 9px 10px;
        min-height: 82px;
      }}
      .matrix-cell.pending {{
        opacity: 0.72;
        background: repeating-linear-gradient(135deg, #fff, #fff 9px, #f3f6f1 9px, #f3f6f1 18px);
      }}
      .matrix-cell .value {{
        font-size: 13px;
        font-weight: 800;
      }}
      .matrix-cell .sub {{
        color: var(--muted);
        font-size: 12px;
        margin-top: 5px;
      }}
      .panel {{
        padding: 18px;
        min-width: 0;
        margin-bottom: 16px;
      }}
      .panel-head {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 14px;
      }}
      .panel h2 {{ font-size: 18px; line-height: 1.15; }}
      .panel-note {{
        color: var(--muted);
        font-size: 13px;
        margin-top: 5px;
      }}
      .controls {{
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) auto;
        gap: 14px;
        align-items: end;
      }}
      .control-block {{
        border: 1px solid var(--line);
        background: var(--soft-2);
        border-radius: 8px;
        padding: 12px;
      }}
      .control-title {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin-bottom: 9px;
        text-transform: uppercase;
      }}
      button {{
        border: 1px solid var(--line);
        background: #fff;
        color: var(--ink);
        border-radius: 8px;
        cursor: pointer;
        font: inherit;
        padding: 9px 12px;
      }}
      button:hover {{ border-color: rgba(36, 95, 131, 0.55); }}
      .charts {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }}
      .chart-card {{
        border: 1px solid var(--line);
        background: #fff;
        border-radius: 8px;
        padding: 14px 14px 10px;
        min-width: 0;
      }}
      .chart-title {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: baseline;
        margin-bottom: 8px;
      }}
      .chart-title h3 {{ font-size: 14px; }}
      .chart-title span {{ color: var(--muted); font-size: 12px; }}
      .chart {{
        min-height: 320px;
        width: 100%;
      }}
      svg {{ width: 100%; height: 100%; display: block; }}
      .axis, .tick {{ fill: var(--muted); font-size: 11px; }}
      .axis-title {{ fill: #34423c; font-size: 12px; font-weight: 700; }}
      .grid-line {{ stroke: #e7ece5; stroke-width: 1; }}
      .zero-line {{ stroke: #9aa79f; stroke-width: 1.2; stroke-dasharray: 4 4; }}
      .point {{
        cursor: pointer;
        stroke: #fff;
        stroke-width: 1.8;
      }}
      .point.selected {{
        stroke: #17211d;
        stroke-width: 3;
      }}
      .legend {{
        margin-top: 12px;
        align-items: center;
      }}
      .legend-item {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--soft-2);
      }}
      .dot {{
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
      }}
      .success-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }}
      .story-card, .bin-card, .heatmap-card {{
        border: 1px solid var(--line);
        background: #fff;
        border-radius: 8px;
        padding: 14px;
        min-width: 0;
      }}
      .story-card {{
        display: grid;
        gap: 12px;
        grid-column: 1 / -1;
      }}
      .story-kpis {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }}
      .story-kpi {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--soft-2);
        padding: 10px;
      }}
      .story-kpi strong {{
        display: block;
        font-size: 18px;
        line-height: 1.1;
      }}
      .bin-rows {{
        display: grid;
        gap: 9px;
        margin-top: 12px;
      }}
      .bin-row {{
        display: grid;
        grid-template-columns: 76px minmax(0, 1fr) 64px 62px;
        align-items: center;
        gap: 10px;
        font-size: 12px;
      }}
      .bin-track {{
        position: relative;
        height: 12px;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: #eef3ec;
      }}
      .bin-track span {{
        display: block;
        height: 100%;
        min-width: 1px;
        border-radius: inherit;
        background: linear-gradient(90deg, var(--gold), var(--green));
      }}
      .heatmap-card {{
        grid-column: 1 / -1;
      }}
      .heatmap {{
        display: grid;
        grid-template-columns: 76px repeat(7, minmax(62px, 1fr));
        gap: 5px;
        margin-top: 12px;
        min-width: 680px;
      }}
      .heatmap-scroll {{
        overflow-x: auto;
      }}
      .heat-label, .heat-cell {{
        border-radius: 7px;
        min-height: 52px;
        padding: 7px;
        font-size: 11px;
      }}
      .heat-label {{
        color: var(--muted);
        display: flex;
        align-items: center;
        font-weight: 800;
      }}
      .heat-cell {{
        border: 1px solid var(--line);
        background: var(--soft-2);
      }}
      .heat-cell.empty {{
        opacity: 0.5;
        background: repeating-linear-gradient(135deg, #fff, #fff 8px, #f3f6f1 8px, #f3f6f1 16px);
      }}
      .heat-cell strong {{
        display: block;
        font-size: 12px;
      }}
      .tables {{
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 16px;
      }}
      .table-wrap {{
        overflow: auto;
        border: 1px solid var(--line);
        border-radius: 8px;
        max-height: 620px;
      }}
      .table-wrap.expanded {{
        max-height: 720px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        min-width: 980px;
      }}
      table.task-expanded {{
        min-width: 1900px;
      }}
      th {{
        background: var(--soft-2);
        color: #34423c;
        font-size: 11px;
        letter-spacing: 0.05em;
        position: sticky;
        top: 0;
        text-align: left;
        text-transform: uppercase;
        z-index: 1;
      }}
      th, td {{
        border-bottom: 1px solid var(--line);
        padding: 9px 10px;
        vertical-align: top;
      }}
      tr.overall td {{
        background: #e8f0e7;
        border-bottom: 2px solid #b9c9b6;
        font-weight: 800;
      }}
      tr.selected-row td {{ background: #fff7df; }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 12px;
      }}
      .right {{ text-align: right; }}
      .empty {{
        border: 1px dashed var(--line);
        border-radius: 8px;
        color: var(--muted);
        padding: 22px;
        text-align: center;
      }}
      #tooltip {{
        position: fixed;
        display: none;
        pointer-events: none;
        z-index: 20;
        max-width: 360px;
        border: 1px solid rgba(23, 33, 29, 0.18);
        background: rgba(255, 255, 255, 0.98);
        color: var(--ink);
        border-radius: 8px;
        box-shadow: 0 16px 36px rgba(23, 33, 29, 0.16);
        padding: 11px 12px;
        font-size: 12px;
      }}
      #tooltip strong {{ display: block; font-size: 13px; margin-bottom: 5px; }}
      #tooltip .muted {{ color: var(--muted); }}
      #tooltip .mini-grid {{
        display: grid;
        grid-template-columns: auto auto;
        column-gap: 14px;
        row-gap: 2px;
        margin-top: 7px;
      }}
      @media (max-width: 1100px) {{
        header, .controls, .charts, .status-grid, .success-grid {{ grid-template-columns: 1fr; }}
        .meta {{ justify-content: flex-start; }}
        .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .story-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      }}
      @media (max-width: 640px) {{
        main {{ padding: 22px 14px 40px; }}
        .cards {{ grid-template-columns: 1fr; }}
        .story-kpis {{ grid-template-columns: 1fr; }}
        .progress-row, .matrix-row {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <div>
          <div class="eyebrow">Responses API hyperparameter sweep</div>
          <h1>{exp_name}</h1>
          <p class="subtitle">
            One-factor-at-a-time view of performance, latency, cost, and token usage.
            Scatter points aggregate across selected domains into one point per hyperparameter configuration.
          </p>
        </div>
        <div class="meta" id="meta"></div>
      </header>

      <section class="grid cards" id="cards"></section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Run Progress</h2>
            <p class="panel-note">Global run completion across every domain and hyperparameter config. Partial active configs appear here before they flush to the result tables.</p>
          </div>
        </div>
        <div class="status-grid">
          <div class="progress-stack" id="progress-bars"></div>
          <div class="status-copy" id="run-status"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Interactive Selection</h2>
            <p class="panel-note">Filter domains and hyperparameter families. Default charts show absolute values; enable baseline deltas for lift and tradeoff analysis.</p>
          </div>
          <button id="clear-selection">Clear selected point</button>
        </div>
        <div class="controls">
          <div class="control-block">
            <div class="control-title">Domains</div>
            <div class="chips" id="domain-filters"></div>
          </div>
          <div class="control-block">
            <div class="control-title">Hyperparameter families</div>
            <div class="chips" id="family-filters"></div>
          </div>
          <div class="control-block">
            <div class="control-title">View mode</div>
            <label class="chip"><input type="checkbox" id="delta-toggle" />Show deltas vs baseline</label>
            <label class="chip"><input type="radio" name="latency-metric" data-latency-metric="mean" value="mean" />Mean latency</label>
            <label class="chip"><input type="radio" name="latency-metric" data-latency-metric="median" value="median" />Median latency</label>
            <label class="chip"><input type="checkbox" id="task-details-toggle" />Expanded task metrics</label>
          </div>
        </div>
        <div class="legend" id="legend"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>OFAT Coverage Matrix</h2>
            <p class="panel-note">Filtered view: each cell is one hyperparameter point aggregated across the currently selected domains. Hatched cells mean at least one selected domain has not committed that hyperparameter point yet.</p>
          </div>
        </div>
        <div class="matrix" id="ofat-matrix"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Performance, Latency, Cost, Tokens</h2>
            <p class="panel-note">Six pairwise 2D plots. Hover for config, domain breakdown, and measured cost/latency; click a point to scope the task table to that config.</p>
          </div>
        </div>
        <div class="charts" id="charts"></div>
      </section>

      <section class="panel" id="success-diagnostics-panel">
        <div class="panel-head">
          <div>
            <h2>Success vs Latency And Steps</h2>
            <p class="panel-note">Binned task success rate by wall-clock latency and message-step count. Success means full reward; steps use message count when raw results are available, otherwise LLM-call count.</p>
          </div>
        </div>
        <div class="success-grid" id="success-diagnostics"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Aggregate Table</h2>
            <p class="panel-note">One row per hyperparameter point across selected domains. The overall row aggregates all visible simulations.</p>
          </div>
        </div>
        <div class="table-wrap"><table id="config-table"></table></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Task Table</h2>
            <p class="panel-note">Task-level view keeps domain and task identity. Click a chart point to show only that configuration. Expanded metrics add tail latency and token distribution columns to the right.</p>
          </div>
        </div>
        <div class="table-wrap" id="task-table-wrap"><table id="task-table"></table></div>
      </section>
    </main>
    <div id="tooltip"></div>
    <script id="manifest-data" type="application/json">{manifest_json}</script>
    <script id="results-data" type="application/json">{results_json}</script>
    <script id="simulations-data" type="application/json">{simulations_json}</script>
    <script id="run-state-data" type="application/json">{run_state_json}</script>
    <script>
      const MANIFEST = JSON.parse(document.getElementById("manifest-data").textContent || "{{}}");
      const RESULTS = JSON.parse(document.getElementById("results-data").textContent || "[]");
      const SIMULATIONS = JSON.parse(document.getElementById("simulations-data").textContent || "[]");
      const RUN_STATE = JSON.parse(document.getElementById("run-state-data").textContent || "{{}}");

      const BASELINE = {{
        llm: MANIFEST.model || "gpt-5.4-mini",
        reasoning_effort: "medium",
        verbosity: "medium",
        web_search_mode: "off",
        service_tier: "default",
        responses_transport: "http",
        parallel_tool_calls: null,
      }};
      const COLORS = {{
        baseline: "#34423c",
        model: "#7f3f98",
        reasoning: "#245f83",
        verbosity: "#1f7855",
        web: "#98711f",
        service: "#635b91",
        transport: "#b4562f",
        parallel: "#58723a",
        variant: "#5f6472",
        combined: "#b4463d",
      }};
      const FAMILIES = ["baseline", "model", "reasoning", "verbosity", "web", "service", "transport", "parallel", "variant", "combined"];
      const LATENCY_BINS = [
        {{ label: "<30s", min: 0, max: 30 }},
        {{ label: "30-60s", min: 30, max: 60 }},
        {{ label: "1-2m", min: 60, max: 120 }},
        {{ label: "2-5m", min: 120, max: 300 }},
        {{ label: "5-10m", min: 300, max: 600 }},
        {{ label: "10-20m", min: 600, max: 1200 }},
        {{ label: "20m+", min: 1200, max: Infinity }},
      ];
      const STEP_BINS = [
        {{ label: "<10", min: 0, max: 10 }},
        {{ label: "10-20", min: 10, max: 20 }},
        {{ label: "20-40", min: 20, max: 40 }},
        {{ label: "40-80", min: 40, max: 80 }},
        {{ label: "80-120", min: 80, max: 120 }},
        {{ label: "120+", min: 120, max: Infinity }},
      ];
      const state = {{
        domains: new Set(),
        families: new Set(FAMILIES),
        delta: false,
        latencyMetric: "mean",
        showTaskDetails: false,
        selectedConfig: null,
      }};

      const toNumber = (value, fallback = 0) => {{
        const n = Number(value);
        return Number.isFinite(n) ? n : fallback;
      }};
      const fmtPct = value => `${{value.toFixed(1)}}%`;
      const fmtPp = value => `${{value >= 0 ? "+" : ""}}${{value.toFixed(1)}} pp`;
      const fmtSec = value => `${{value >= 0 ? "" : "-"}}${{Math.abs(value).toFixed(value < 10 && value > -10 ? 1 : 0)}}s`;
      const fmtUsd = value => `${{value < 0 ? "-" : ""}}$${{Math.abs(value).toFixed(value < 0.1 && value > -0.1 ? 4 : 3)}}`;
      const fmtTokens = value => `${{value >= 0 ? "" : "-"}}${{Math.abs(value) >= 1000 ? (Math.abs(value) / 1000).toFixed(1) + "k" : Math.abs(value).toFixed(0)}}`;
      const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));

      function configKey(row) {{
        return [
          row.llm || MANIFEST.model || "",
          row.mode || "default",
          row.reasoning_effort,
          row.verbosity,
          row.web_search_mode,
          row.service_tier,
          row.responses_transport || "http",
          row.parallel_tool_calls ?? "",
          row.variant || "baseline",
        ].join("|");
      }}

      function configLabel(row) {{
        const prefix = row.variant && row.variant !== "baseline" ? `${{row.variant}}: ` : "";
        const model = row.llm && row.llm !== BASELINE.llm ? `${{row.llm}} / ` : "";
        const transport = row.responses_transport && row.responses_transport !== "http" ? ` / ${{row.responses_transport}}` : "";
        const parallel = row.parallel_tool_calls !== null && row.parallel_tool_calls !== undefined && row.parallel_tool_calls !== "" ? ` / parallel ${{row.parallel_tool_calls}}` : "";
        return `${{prefix}}${{model}}${{row.reasoning_effort}} / ${{row.verbosity}} / web ${{row.web_search_mode}} / ${{row.service_tier}}${{transport}}${{parallel}}`;
      }}

      function familyFor(row) {{
        if (row.llm && row.llm !== BASELINE.llm) return "model";
        if (row.responses_transport && row.responses_transport !== BASELINE.responses_transport) return "transport";
        if (row.parallel_tool_calls !== null && row.parallel_tool_calls !== undefined && row.parallel_tool_calls !== "") return "parallel";
        if (row.variant && row.variant !== "baseline") return "variant";
        const diffs = [
          row.reasoning_effort !== BASELINE.reasoning_effort ? "reasoning" : null,
          row.verbosity !== BASELINE.verbosity ? "verbosity" : null,
          row.web_search_mode !== BASELINE.web_search_mode ? "web" : null,
          row.service_tier !== BASELINE.service_tier ? "service" : null,
        ].filter(Boolean);
        if (diffs.length === 0) return "baseline";
        if (diffs.length === 1) return diffs[0];
        return "combined";
      }}

      function synthesizeSimulationsFromResults() {{
        return RESULTS.map(row => ({{
          ...row,
          simulation_id: row.name,
          task_id: "summary",
          trial: "0",
          reward: row.avg_reward,
          duration_seconds: row.avg_duration_seconds,
          estimated_total_cost_usd: row.avg_estimated_total_cost_usd,
          agent_total_tokens: row.avg_agent_total_tokens,
          user_total_tokens: row.avg_user_total_tokens,
        }}));
      }}

      function normalizedRows() {{
        const source = SIMULATIONS.length ? SIMULATIONS : synthesizeSimulationsFromResults();
        return source.map(row => {{
          const rawParallel = row.parallel_tool_calls;
          const parallelToolCalls = rawParallel === true || rawParallel === "True" || rawParallel === "true"
            ? true
            : rawParallel === false || rawParallel === "False" || rawParallel === "false"
              ? false
              : null;
          const normalized = {{
            ...row,
            domain: row.domain || "unknown",
            mode: row.mode || "default",
            llm: row.llm || MANIFEST.model || "",
            reasoning_effort: row.reasoning_effort || "medium",
            verbosity: row.verbosity || "medium",
            web_search_mode: row.web_search_mode || "off",
            service_tier: row.service_tier || "default",
            responses_transport: row.responses_transport || "http",
            parallel_tool_calls: parallelToolCalls,
            variant: row.variant || "baseline",
            task_id: row.task_id ?? "unknown",
            reward: toNumber(row.reward),
            duration: toNumber(row.duration_seconds),
            cost: toNumber(row.estimated_total_cost_usd),
            agentCost: toNumber(row.agent_estimated_total_cost_usd),
            userCost: toNumber(row.user_estimated_total_cost_usd),
            tokens: toNumber(row.agent_total_tokens) + toNumber(row.user_total_tokens),
            agentTokens: toNumber(row.agent_total_tokens),
            userTokens: toNumber(row.user_total_tokens),
            webCalls: toNumber(row.agent_web_search_calls),
            llmCalls: toNumber(row.agent_llm_calls) + toNumber(row.user_llm_calls),
            messageCount: toNumber(row.message_count || row.num_messages),
            infraError: String(row.termination_reason || "").toLowerCase().includes("error") ? 1 : 0,
          }};
          normalized.stepCount = normalized.messageCount || normalized.llmCalls;
          normalized.success = normalized.reward >= 1 ? 1 : 0;
          normalized.maxStep = String(row.termination_reason || "").toLowerCase().includes("max_steps") ? 1 : 0;
          normalized.key = configKey(normalized);
          normalized.family = familyFor(normalized);
          normalized.label = configLabel(normalized);
          return normalized;
        }});
      }}

      const ROWS = normalizedRows();
      const MANIFEST_CONFIGS = (MANIFEST.configs || []).map(config => ({{
        ...config,
        key: configKey(config),
        label: configLabel(config),
        family: familyFor(config),
      }}));
      const ALL_DOMAINS = Array.from(new Set([...(MANIFEST.domains || []), ...ROWS.map(row => row.domain)])).sort();
      ALL_DOMAINS.forEach(domain => state.domains.add(domain));

      function mean(values) {{
        const clean = values.filter(Number.isFinite);
        if (!clean.length) return 0;
        return clean.reduce((a, b) => a + b, 0) / clean.length;
      }}

      function sortedNumbers(values) {{
        return values.filter(Number.isFinite).sort((a, b) => a - b);
      }}

      function quantile(values, q) {{
        const clean = sortedNumbers(values);
        if (!clean.length) return 0;
        const pos = (clean.length - 1) * q;
        const base = Math.floor(pos);
        const rest = pos - base;
        if (clean[base + 1] === undefined) return clean[base];
        return clean[base] + rest * (clean[base + 1] - clean[base]);
      }}

      function maxValue(values) {{
        const clean = values.filter(Number.isFinite);
        return clean.length ? Math.max(...clean) : 0;
      }}

      function trimmedMean(values, fraction = 0.05) {{
        const clean = sortedNumbers(values);
        if (!clean.length) return 0;
        const trim = Math.floor(clean.length * fraction);
        const kept = trim > 0 && clean.length - trim > trim ? clean.slice(trim, clean.length - trim) : clean;
        return mean(kept);
      }}

      function distribution(values) {{
        return {{
          mean: mean(values),
          median: quantile(values, 0.5),
          p90: quantile(values, 0.9),
          p95: quantile(values, 0.95),
          max: maxValue(values),
          trimmed5: trimmedMean(values, 0.05),
        }};
      }}

      function selectedLatency(stats) {{
        return state.latencyMetric === "median" ? stats.median : stats.mean;
      }}

      function latencyMetricTitle() {{
        return state.latencyMetric === "median" ? "Median" : "Mean";
      }}

      function successRate(rows) {{
        return rows.length ? mean(rows.map(row => row.success)) * 100 : 0;
      }}

      function binRows(rows, bins, accessor) {{
        return bins.map(bin => {{
          const values = rows.filter(row => {{
            const value = accessor(row);
            return Number.isFinite(value) && value >= bin.min && value < bin.max;
          }});
          return {{
            ...bin,
            rows: values,
            n: values.length,
            success: successRate(values),
            avgReward: mean(values.map(row => row.reward)) * 100,
            maxSteps: sum(values.map(row => row.maxStep)),
          }};
        }});
      }}

      function firstNonEmptyBin(stats) {{
        return stats.find(item => item.n > 0) || null;
      }}

      function lastNonEmptyBin(stats) {{
        return stats.filter(item => item.n > 0).at(-1) || null;
      }}

      function bestSupportedBin(stats, minN) {{
        const supported = stats.filter(item => item.n >= minN);
        if (!supported.length) return null;
        return supported.reduce((best, item) => item.success > best.success ? item : best, supported[0]);
      }}

      function sum(values) {{
        return values.filter(Number.isFinite).reduce((a, b) => a + b, 0);
      }}

      function groupBy(rows, keyFn) {{
        const map = new Map();
        rows.forEach(row => {{
          const key = keyFn(row);
          if (!map.has(key)) map.set(key, []);
          map.get(key).push(row);
        }});
        return Array.from(map, ([key, values]) => [key, values]);
      }}

      function pct(value, total) {{
        return total > 0 ? Math.max(0, Math.min(100, (value / total) * 100)) : 0;
      }}

      function dateLabel(value) {{
        if (!value) return "unknown";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "unknown";
        return date.toLocaleString([], {{ dateStyle: "medium", timeStyle: "short" }});
      }}

      function domainExpectedConfigCounts() {{
        const counts = new Map();
        MANIFEST_CONFIGS.forEach(config => counts.set(config.domain, (counts.get(config.domain) || 0) + 1));
        if (!counts.size) {{
          groupBy(RESULTS, row => row.domain).forEach(([domain, values]) => counts.set(domain, values.length));
        }}
        return counts;
      }}

      function domainCompletedConfigCounts() {{
        const counts = new Map();
        groupBy(RESULTS, row => row.domain).forEach(([domain, values]) => counts.set(domain, values.length));
        return counts;
      }}

      function domainTaskCounts() {{
        const counts = new Map();
        RESULTS.forEach(row => {{
          const count = toNumber(row.num_simulations);
          if (count > 0) counts.set(row.domain, Math.max(counts.get(row.domain) || 0, count));
        }});
        groupBy(ROWS, row => row.domain).forEach(([domain, values]) => {{
          const tasks = new Set(values.map(row => row.task_id)).size;
          if (tasks > 0 && !counts.has(domain)) counts.set(domain, tasks);
        }});
        return counts;
      }}

      function expectedSimulationTotal() {{
        const taskCounts = domainTaskCounts();
        if (MANIFEST_CONFIGS.length) {{
          return MANIFEST_CONFIGS.reduce((total, config) => total + (taskCounts.get(config.domain) || 0), 0);
        }}
        return ROWS.length;
      }}

      function completedSimulationTotal() {{
        const fromResults = RESULTS.reduce((total, row) => total + toNumber(row.num_simulations), 0);
        return fromResults || ROWS.length;
      }}

      function missingConfigs() {{
        if (!MANIFEST_CONFIGS.length) return [];
        const completed = new Set(RESULTS.map(row => row.name));
        return MANIFEST_CONFIGS.filter(config => !completed.has(config.name));
      }}

      function progressMarkup(label, value, total, hint = "") {{
        const percentage = pct(value, total);
        return `
          <div class="progress-row">
            <div class="label">${{esc(label)}}</div>
            <div class="bar"><span style="width:${{percentage}}%"></span></div>
            <div class="right mono">${{value}} / ${{total || "?"}}</div>
            ${{hint ? `<div></div><div class="hint">${{hint}}</div><div></div>` : ""}}
          </div>
        `;
      }}

      function runTotals() {{
        const expectedConfigs = MANIFEST_CONFIGS.length || RESULTS.length;
        const completedConfigs = RESULTS.length;
        const expectedSims = expectedSimulationTotal();
        const completedSims = completedSimulationTotal();
        return {{
          expectedConfigs,
          completedConfigs,
          expectedSims,
          completedSims,
          missing: missingConfigs(),
        }};
      }}

      function aggregateConfigRows(rows) {{
        return groupBy(rows, row => row.key).map(([key, values]) => {{
          const first = values[0];
          const domains = Array.from(new Set(values.map(row => row.domain))).sort();
          const tasks = Array.from(new Set(values.map(row => `${{row.domain}}:${{row.task_id}}`)));
          const latencyStats = distribution(values.map(row => row.duration));
          const tokenStats = distribution(values.map(row => row.tokens));
          const domainBreakdown = groupBy(values, row => row.domain).map(([domain, domainRows]) => ({{
            domain,
            n: domainRows.length,
            performance: mean(domainRows.map(row => row.reward)) * 100,
            latency: selectedLatency(distribution(domainRows.map(row => row.duration))),
            cost: mean(domainRows.map(row => row.cost)),
          }}));
          return {{
            ...first,
            key,
            label: configLabel(first),
            family: first.family,
            domains,
            tasks: tasks.length,
            simulations: values.length,
            performance: mean(values.map(row => row.reward)) * 100,
            latency: selectedLatency(latencyStats),
            latencyMean: latencyStats.mean,
            latencyMedian: latencyStats.median,
            latencyP90: latencyStats.p90,
            latencyP95: latencyStats.p95,
            latencyMax: latencyStats.max,
            latencyTrimmed5: latencyStats.trimmed5,
            cost: mean(values.map(row => row.cost)),
            tokens: tokenStats.mean,
            tokensMean: tokenStats.mean,
            tokensMedian: tokenStats.median,
            tokensP90: tokenStats.p90,
            tokensP95: tokenStats.p95,
            tokensMax: tokenStats.max,
            totalCost: sum(values.map(row => row.cost)),
            totalWebCalls: sum(values.map(row => row.webCalls)),
            webUseRate: mean(values.map(row => row.webCalls > 0 ? 1 : 0)) * 100,
            domainBreakdown,
          }};
        }}).sort((a, b) => familyOrder(a.family) - familyOrder(b.family) || a.label.localeCompare(b.label));
      }}

      function aggregateTaskRows(rows) {{
        const scoped = state.selectedConfig ? rows.filter(row => row.key === state.selectedConfig) : rows;
        const overallLatency = distribution(scoped.map(row => row.duration));
        const overallTokens = distribution(scoped.map(row => row.tokens));
        const overall = {{
          domain: "OVERALL",
          task_id: "All selected tasks",
          simulations: scoped.length,
          configs: new Set(scoped.map(row => row.key)).size,
          performance: mean(scoped.map(row => row.reward)) * 100,
          latency: selectedLatency(overallLatency),
          latencyMean: overallLatency.mean,
          latencyMedian: overallLatency.median,
          latencyP90: overallLatency.p90,
          latencyP95: overallLatency.p95,
          latencyMax: overallLatency.max,
          latencyTrimmed5: overallLatency.trimmed5,
          maxSteps: sum(scoped.map(row => String(row.termination_reason || "").toLowerCase().includes("max_steps") ? 1 : 0)),
          timeoutRuns: sum(scoped.map(row => String(row.termination_reason || "").toLowerCase().includes("timeout") ? 1 : 0)),
          cost: mean(scoped.map(row => row.cost)),
          tokens: overallTokens.mean,
          tokensMean: overallTokens.mean,
          tokensMedian: overallTokens.median,
          tokensP90: overallTokens.p90,
          tokensP95: overallTokens.p95,
          tokensMax: overallTokens.max,
          agentTokensMean: mean(scoped.map(row => row.agentTokens)),
          userTokensMean: mean(scoped.map(row => row.userTokens)),
          overall: true,
        }};
        const tasks = groupBy(scoped, row => `${{row.domain}}|${{row.task_id}}`).map(([key, values]) => {{
          const [domain, task_id] = key.split("|");
          const latencyStats = distribution(values.map(row => row.duration));
          const tokenStats = distribution(values.map(row => row.tokens));
          return {{
            domain,
            task_id,
            simulations: values.length,
            configs: new Set(values.map(row => row.key)).size,
            performance: mean(values.map(row => row.reward)) * 100,
            latency: selectedLatency(latencyStats),
            latencyMean: latencyStats.mean,
            latencyMedian: latencyStats.median,
            latencyP90: latencyStats.p90,
            latencyP95: latencyStats.p95,
            latencyMax: latencyStats.max,
            latencyTrimmed5: latencyStats.trimmed5,
            maxSteps: sum(values.map(row => String(row.termination_reason || "").toLowerCase().includes("max_steps") ? 1 : 0)),
            timeoutRuns: sum(values.map(row => String(row.termination_reason || "").toLowerCase().includes("timeout") ? 1 : 0)),
            cost: mean(values.map(row => row.cost)),
            tokens: tokenStats.mean,
            tokensMean: tokenStats.mean,
            tokensMedian: tokenStats.median,
            tokensP90: tokenStats.p90,
            tokensP95: tokenStats.p95,
            tokensMax: tokenStats.max,
            agentTokensMean: mean(values.map(row => row.agentTokens)),
            userTokensMean: mean(values.map(row => row.userTokens)),
          }};
        }}).sort((a, b) => a.domain.localeCompare(b.domain) || String(a.task_id).localeCompare(String(b.task_id)));
        return [overall, ...tasks];
      }}

      function familyOrder(family) {{
        const index = FAMILIES.indexOf(family);
        return index === -1 ? 99 : index;
      }}

      function filteredRows() {{
        return ROWS.filter(row => state.domains.has(row.domain) && state.families.has(row.family));
      }}

      function baselineAggregate(configRows) {{
        return configRows.find(row => row.family === "baseline") || null;
      }}

      function displayRows(configRows) {{
        if (!state.delta) return configRows;
        const baseline = baselineAggregate(configRows);
        if (!baseline) return configRows;
        return configRows.map(row => ({{
          ...row,
          performanceDisplay: row.performance - baseline.performance,
          latencyDisplay: row.latency - baseline.latency,
          costDisplay: row.cost - baseline.cost,
          tokensDisplay: row.tokens - baseline.tokens,
        }}));
      }}

      function withDisplayMetrics(configRows) {{
        return configRows.map(row => ({{
          ...row,
          performanceDisplay: row.performanceDisplay ?? row.performance,
          latencyDisplay: row.latencyDisplay ?? row.latency,
          costDisplay: row.costDisplay ?? row.cost,
          tokensDisplay: row.tokensDisplay ?? row.tokens,
        }}));
      }}

      function renderMeta() {{
        const points = (MANIFEST.points || []).length || new Set(ROWS.map(row => row.key)).size;
        const runLimits = [];
        if (MANIFEST.max_steps) runLimits.push(`max steps ${{MANIFEST.max_steps}}`);
        if (MANIFEST.max_duration_seconds) runLimits.push(`timeout ${{fmtSec(MANIFEST.max_duration_seconds)}}`);
        document.getElementById("meta").innerHTML = [
          `model ${{esc(MANIFEST.model || "unknown")}}`,
          `shape ${{esc(MANIFEST.shape || "unknown")}}`,
          `${{points}} hyperparameter points`,
          `${{ALL_DOMAINS.length}} domains in output`,
          ...runLimits,
        ].map(value => `<span class="pill">${{value}}</span>`).join("");
      }}

      function renderFilters() {{
        document.getElementById("domain-filters").innerHTML = ALL_DOMAINS.map(domain => `
          <label class="chip ${{state.domains.has(domain) ? "active" : ""}}">
            <input type="checkbox" data-domain="${{esc(domain)}}" ${{state.domains.has(domain) ? "checked" : ""}} />${{esc(domain)}}
          </label>
        `).join("");
        document.getElementById("family-filters").innerHTML = FAMILIES.map(family => `
          <label class="chip ${{state.families.has(family) ? "active" : ""}}">
            <input type="checkbox" data-family="${{family}}" ${{state.families.has(family) ? "checked" : ""}} />${{family}}
          </label>
        `).join("");
        document.getElementById("delta-toggle").checked = state.delta;
        document.getElementById("delta-toggle").closest(".chip")?.classList.toggle("active", state.delta);
        document.querySelectorAll("[data-latency-metric]").forEach(input => {{
          input.checked = input.dataset.latencyMetric === state.latencyMetric;
          input.closest(".chip")?.classList.toggle("active", input.checked);
        }});
        document.getElementById("task-details-toggle").checked = state.showTaskDetails;
        document.getElementById("task-details-toggle").closest(".chip")?.classList.toggle("active", state.showTaskDetails);
      }}

      function renderLegend() {{
        document.getElementById("legend").innerHTML = FAMILIES.map(family => `
          <span class="legend-item"><span class="dot" style="background:${{COLORS[family]}}"></span>${{family}}</span>
        `).join("") + `<span class="legend-item">Point size = selected-task count</span>`;
      }}

      function renderCards(rows, configRows) {{
        const domains = new Set(rows.map(row => row.domain)).size;
        const taskCount = new Set(rows.map(row => `${{row.domain}}:${{row.task_id}}`)).size;
        const latencyStats = distribution(rows.map(row => row.duration));
        const tokenStats = distribution(rows.map(row => row.tokens));
        const alternateLatency = state.latencyMetric === "mean"
          ? `median ${{fmtSec(latencyStats.median)}}`
          : `mean ${{fmtSec(latencyStats.mean)}}`;
        const cards = [
          ["Performance", fmtPct(mean(rows.map(row => row.reward)) * 100), `${{rows.length}} simulations`],
          [`${{latencyMetricTitle()}} latency`, fmtSec(selectedLatency(latencyStats)), `${{alternateLatency}} · p95 ${{fmtSec(latencyStats.p95)}}`],
          ["Mean cost", fmtUsd(mean(rows.map(row => row.cost))), `total ${{fmtUsd(sum(rows.map(row => row.cost)))}}`],
          ["Mean tokens", fmtTokens(tokenStats.mean), `median ${{fmtTokens(tokenStats.median)}} · p95 ${{fmtTokens(tokenStats.p95)}}`],
          ["Coverage", `${{domains}} domains`, `${{taskCount}} distinct tasks, ${{configRows.length}} configs`],
        ];
        document.getElementById("cards").innerHTML = cards.map(([label, metric, hint]) => `
          <div class="card"><div class="label">${{label}}</div><div class="metric">${{metric}}</div><div class="hint">${{hint}}</div></div>
        `).join("");
      }}

      function renderProgress() {{
        const totals = runTotals();
        const domainExpected = domainExpectedConfigCounts();
        const domainCompleted = domainCompletedConfigCounts();
        const domainBars = ALL_DOMAINS.map(domain => {{
          const complete = domainCompleted.get(domain) || 0;
          const expected = domainExpected.get(domain) || complete;
          return progressMarkup(domain, complete, expected);
        }}).join("");

        document.getElementById("progress-bars").innerHTML = [
          progressMarkup("Configs", totals.completedConfigs, totals.expectedConfigs, `${{totals.missing.length}} remaining`),
          progressMarkup("Simulations", totals.completedSims, totals.expectedSims),
          domainBars,
        ].join("");

        const latest = RUN_STATE.log?.latest_status;
        const latestTask = RUN_STATE.log?.latest_task;
        const fileUpdated = RUN_STATE.files?.results?.modified_at;
        const activeLine = latest
          ? `Active config status: ${{latest.complete}}/${{latest.total}} complete, avg reward ${{esc(latest.avg_reward)}}.`
          : "No active run status was found in run.log.";
        const taskLine = latestTask
          ? `Latest started task: ${{latestTask.started}}/${{latestTask.total}}. Longest visible active task: ${{fmtSec(RUN_STATE.log?.max_running_seconds || 0)}}.`
          : "";
        document.getElementById("run-status").innerHTML = `
          <strong>${{totals.completedConfigs >= totals.expectedConfigs ? "Run complete" : "Run in progress"}}</strong>
          <div>${{activeLine}}</div>
          <div>${{taskLine}}</div>
          <div class="hint">results.csv last updated ${{dateLabel(fileUpdated)}}. Report generated ${{dateLabel(RUN_STATE.generated_at)}}.</div>
        `;
      }}

      function renderOfatMatrix(configRows) {{
        const byKey = new Map(configRows.map(row => [row.key, row]));
        const selectedDomainCount = state.domains.size;
        const expectedPoints = Array.from(
          (MANIFEST_CONFIGS.length ? MANIFEST_CONFIGS : configRows).reduce((map, config) => {{
            if (!config.domain || state.domains.has(config.domain)) map.set(config.key, config);
            return map;
          }}, new Map()).values()
        );
        const expectedByFamily = groupBy(expectedPoints, row => row.family)
          .filter(([family]) => state.families.has(family))
          .sort((a, b) => familyOrder(a[0]) - familyOrder(b[0]));
        const completedNames = new Set(RESULTS.map(row => row.name));

        if (!expectedByFamily.length) {{
          document.getElementById("ofat-matrix").innerHTML = `<div class="empty">No expected hyperparameter points match the current filters.</div>`;
          return;
        }}

        document.getElementById("ofat-matrix").innerHTML = expectedByFamily.map(([family, configs]) => {{
          const cells = configs.map(config => {{
            const row = byKey.get(config.key);
            const expectedForPoint = MANIFEST_CONFIGS.filter(item => item.key === config.key && state.domains.has(item.domain));
            const completedForPoint = expectedForPoint.filter(item => completedNames.has(item.name));
            const pending = expectedForPoint.length > completedForPoint.length;
            const border = COLORS[family] || COLORS.combined;
            const metrics = row
              ? `${{fmtPct(row.performance)}} · ${{fmtSec(row.latency)}} · ${{fmtUsd(row.cost)}}`
              : "Waiting for committed result";
            const coverage = expectedForPoint.length
              ? `${{completedForPoint.length}}/${{expectedForPoint.length}} selected domains committed`
              : `${{row?.domains?.length || 0}} domains`;
            return `
              <div class="matrix-cell ${{pending ? "pending" : ""}}" style="border-left-color:${{border}}" title="${{esc(config.label)}}">
                <div class="value">${{esc(config.label)}}</div>
                <div class="sub">${{metrics}}</div>
                <div class="sub">${{coverage}}</div>
              </div>
            `;
          }}).join("");
          return `
            <div class="matrix-row">
              <div class="matrix-label">${{esc(family)}}</div>
              <div class="matrix-cells">${{cells}}</div>
            </div>
          `;
        }}).join("") + `
          <div class="hint">This matrix is scoped to ${{selectedDomainCount}} selected domain${{selectedDomainCount === 1 ? "" : "s"}}. Use the domain filters above to switch between telecom-only and all-domain coverage.</div>
        `;
      }}

      function metricFormat(metric, value) {{
        if (metric === "performance") return state.delta ? fmtPp(value) : fmtPct(value);
        if (metric === "latency") return fmtSec(value);
        if (metric === "cost") return fmtUsd(value);
        if (metric === "tokens") return fmtTokens(value);
        return value.toFixed(2);
      }}

      function metricLabel(metric) {{
        const base = {{
          performance: "performance",
          latency: `${{state.latencyMetric}} latency`,
          cost: "cost",
          tokens: "tokens",
        }}[metric];
        return state.delta ? `${{base}} delta vs baseline` : base;
      }}

      function axisTicks(min, max, count = 4) {{
        if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
        if (min === max) {{
          const pad = Math.abs(min) || 1;
          min -= pad * 0.5;
          max += pad * 0.5;
        }}
        const step = (max - min) / count;
        return Array.from({{ length: count + 1 }}, (_, i) => min + step * i);
      }}

      function scale(value, min, max, outMin, outMax) {{
        if (min === max) return (outMin + outMax) / 2;
        return outMin + ((value - min) / (max - min)) * (outMax - outMin);
      }}

      function tipFor(row) {{
        const domains = row.domainBreakdown.map(item =>
          `<div>${{esc(item.domain)}}</div><div>${{fmtPct(item.performance)}} / ${{fmtSec(item.latency)}} / ${{fmtUsd(item.cost)}}</div>`
        ).join("");
        return `
          <strong>${{esc(row.label)}}</strong>
          <div class="muted">${{esc(row.family)}} point · ${{row.simulations}} simulations · ${{row.tasks}} tasks</div>
          <div class="mini-grid">
            <div>Performance</div><div>${{fmtPct(row.performance)}}</div>
            <div>${{latencyMetricTitle()}} latency</div><div>${{fmtSec(row.latency)}}</div>
            <div>Latency p95</div><div>${{fmtSec(row.latencyP95)}}</div>
            <div>Cost</div><div>${{fmtUsd(row.cost)}} avg, ${{fmtUsd(row.totalCost)}} total</div>
            <div>Mean tokens</div><div>${{fmtTokens(row.tokens)}}</div>
            <div>Token p95</div><div>${{fmtTokens(row.tokensP95)}}</div>
            <div>Web usage</div><div>${{fmtPct(row.webUseRate)}} · ${{row.totalWebCalls}} calls</div>
          </div>
          <div class="mini-grid">${{domains}}</div>
        `;
      }}

      function renderScatter(container, rows, spec) {{
        const width = 640, height = 360;
        const m = {{ top: 18, right: 24, bottom: 56, left: 70 }};
        const plotW = width - m.left - m.right;
        const plotH = height - m.top - m.bottom;
        const xField = `${{spec.x}}Display`;
        const yField = `${{spec.y}}Display`;
        const xVals = rows.map(row => row[xField]);
        const yVals = rows.map(row => row[yField]);
        let xMin = Math.min(...xVals), xMax = Math.max(...xVals);
        let yMin = Math.min(...yVals), yMax = Math.max(...yVals);
        const xPad = (xMax - xMin || Math.abs(xMax) || 1) * 0.08;
        const yPad = (yMax - yMin || Math.abs(yMax) || 1) * 0.08;
        xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
        const xTicks = axisTicks(xMin, xMax);
        const yTicks = axisTicks(yMin, yMax);
        const zeroX = state.delta && xMin < 0 && xMax > 0 ? scale(0, xMin, xMax, m.left, m.left + plotW) : null;
        const zeroY = state.delta && yMin < 0 && yMax > 0 ? scale(0, yMin, yMax, m.top + plotH, m.top) : null;
        const maxTasks = Math.max(...rows.map(row => row.tasks), 1);
        const points = rows.map(row => {{
          const x = scale(row[xField], xMin, xMax, m.left, m.left + plotW);
          const y = scale(row[yField], yMin, yMax, m.top + plotH, m.top);
          const r = 5 + 7 * Math.sqrt(row.tasks / maxTasks);
          const selected = state.selectedConfig === row.key ? " selected" : "";
          return `<circle class="point${{selected}}" cx="${{x}}" cy="${{y}}" r="${{r}}" fill="${{COLORS[row.family]}}" data-key="${{esc(row.key)}}" data-tip="${{esc(tipFor(row))}}"></circle>`;
        }}).join("");
        container.innerHTML = `
          <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{esc(spec.title)}}">
            ${{xTicks.map(tick => {{
              const x = scale(tick, xMin, xMax, m.left, m.left + plotW);
              return `<line class="grid-line" x1="${{x}}" x2="${{x}}" y1="${{m.top}}" y2="${{m.top + plotH}}"></line><text class="tick" x="${{x}}" y="${{height - 31}}" text-anchor="middle">${{metricFormat(spec.x, tick)}}</text>`;
            }}).join("")}}
            ${{yTicks.map(tick => {{
              const y = scale(tick, yMin, yMax, m.top + plotH, m.top);
              return `<line class="grid-line" x1="${{m.left}}" x2="${{m.left + plotW}}" y1="${{y}}" y2="${{y}}"></line><text class="tick" x="${{m.left - 10}}" y="${{y + 4}}" text-anchor="end">${{metricFormat(spec.y, tick)}}</text>`;
            }}).join("")}}
            ${{zeroX ? `<line class="zero-line" x1="${{zeroX}}" x2="${{zeroX}}" y1="${{m.top}}" y2="${{m.top + plotH}}"></line>` : ""}}
            ${{zeroY ? `<line class="zero-line" x1="${{m.left}}" x2="${{m.left + plotW}}" y1="${{zeroY}}" y2="${{zeroY}}"></line>` : ""}}
            <text class="axis-title" x="${{m.left + plotW / 2}}" y="${{height - 6}}" text-anchor="middle">${{metricLabel(spec.x)}}</text>
            <text class="axis-title" x="20" y="${{m.top + plotH / 2}}" transform="rotate(-90 20 ${{m.top + plotH / 2}})" text-anchor="middle">${{metricLabel(spec.y)}}</text>
            ${{points}}
          </svg>
        `;
      }}

      function renderCharts(rows) {{
        const specs = [
          {{ id: "perf-latency", title: "Performance vs latency", x: "latency", y: "performance" }},
          {{ id: "perf-cost", title: "Performance vs cost", x: "cost", y: "performance" }},
          {{ id: "latency-cost", title: "Latency vs cost", x: "latency", y: "cost" }},
          {{ id: "perf-tokens", title: "Performance vs tokens", x: "tokens", y: "performance" }},
          {{ id: "latency-tokens", title: "Latency vs tokens", x: "tokens", y: "latency" }},
          {{ id: "cost-tokens", title: "Cost vs tokens", x: "tokens", y: "cost" }},
        ];
        const container = document.getElementById("charts");
        if (!rows.length) {{
          container.innerHTML = `<div class="empty">No hyperparameter points match the current filters.</div>`;
          return;
        }}
        container.innerHTML = specs.map(spec => `
          <div class="chart-card">
            <div class="chart-title"><h3>${{spec.title}}</h3><span>${{state.delta ? "delta view" : "absolute view"}}</span></div>
            <div class="chart" id="${{spec.id}}"></div>
          </div>
        `).join("");
        specs.forEach(spec => renderScatter(document.getElementById(spec.id), rows, spec));
      }}

      function renderBinChart(title, stats) {{
        const maxN = Math.max(...stats.map(item => item.n), 1);
        return `
          <div class="bin-card">
            <div class="chart-title"><h3>${{title}}</h3><span>bars = success rate</span></div>
            <div class="bin-rows">
              ${{stats.map(item => `
                <div class="bin-row" title="${{esc(`${{item.label}}: ${{fmtPct(item.success)}} success, n=${{item.n}}, max-step=${{item.maxSteps}}`)}}">
                  <div class="mono">${{esc(item.label)}}</div>
                  <div class="bin-track"><span style="width:${{Math.max(1, item.success)}}%"></span></div>
                  <div class="right">${{item.n ? fmtPct(item.success) : "n/a"}}</div>
                  <div class="right muted">n=${{item.n}}</div>
                </div>
              `).join("")}}
            </div>
            <div class="hint">Largest bucket count: ${{maxN}}. Max-step runs are included in the bucket tooltip.</div>
          </div>
        `;
      }}

      function successStoryMarkup(rows, latencyStats, stepStats) {{
        const minN = Math.max(8, Math.ceil(rows.length * 0.01));
        const firstLatency = firstNonEmptyBin(latencyStats);
        const bestLatency = bestSupportedBin(latencyStats, minN);
        const tailLatency = lastNonEmptyBin(latencyStats);
        const firstStep = firstNonEmptyBin(stepStats);
        const bestStep = bestSupportedBin(stepStats, minN);
        const tailStep = lastNonEmptyBin(stepStats);
        const maxStepRows = rows.filter(row => row.maxStep);
        const overallSuccess = successRate(rows);
        const peakBeatsTail = bestLatency && tailLatency && bestLatency.success > tailLatency.success + 5;
        const earlyImproves = firstLatency && bestLatency && bestLatency.success > firstLatency.success + 2;
        const verdict = peakBeatsTail
          ? `${{earlyImproves ? "The current data supports the expected shape:" : "The current data partly supports the expected shape:"}} success peaks before the tail, then drops sharply for very long tasks.`
          : "The current data does not show a clear late-task collapse after a supported peak; check the bins and domain filters.";
        return `
          <div class="story-card">
            <div>
              <h3>Observed Pattern</h3>
              <p class="panel-note">${{verdict}} Latency peak: ${{bestLatency ? `${{esc(bestLatency.label)}} at ${{fmtPct(bestLatency.success)}}` : "not enough data"}}; longest latency bucket: ${{tailLatency ? `${{esc(tailLatency.label)}} at ${{fmtPct(tailLatency.success)}}` : "none"}}. Step peak: ${{bestStep ? `${{esc(bestStep.label)}} at ${{fmtPct(bestStep.success)}}` : "not enough data"}}; longest step bucket: ${{tailStep ? `${{esc(tailStep.label)}} at ${{fmtPct(tailStep.success)}}` : "none"}}.</p>
            </div>
            <div class="story-kpis">
              <div class="story-kpi"><div class="label">Overall success</div><strong>${{fmtPct(overallSuccess)}}</strong><div class="hint">${{rows.length}} simulations</div></div>
              <div class="story-kpi"><div class="label">Best latency bucket</div><strong>${{bestLatency ? fmtPct(bestLatency.success) : "n/a"}}</strong><div class="hint">${{bestLatency ? `${{bestLatency.label}}, n=${{bestLatency.n}}` : "No supported bin"}}</div></div>
              <div class="story-kpi"><div class="label">Tail latency bucket</div><strong>${{tailLatency ? fmtPct(tailLatency.success) : "n/a"}}</strong><div class="hint">${{tailLatency ? `${{tailLatency.label}}, n=${{tailLatency.n}}` : "No tail bin"}}</div></div>
              <div class="story-kpi"><div class="label">Max-step success</div><strong>${{fmtPct(successRate(maxStepRows))}}</strong><div class="hint">${{maxStepRows.length}} max-step runs</div></div>
            </div>
          </div>
        `;
      }}

      function heatColor(success, n) {{
        if (!n) return "";
        const alpha = 0.16 + 0.72 * (success / 100);
        return `background: rgba(31, 120, 85, ${{alpha.toFixed(2)}}); border-color: rgba(31, 120, 85, 0.35); color: ${{success > 62 ? "#fff" : "var(--ink)"}};`;
      }}

      function renderSuccessHeatmap(rows) {{
        const header = `<div class="heat-label"></div>${{LATENCY_BINS.map(bin => `<div class="heat-label">${{esc(bin.label)}}</div>`).join("")}}`;
        const body = STEP_BINS.map(stepBin => {{
          const cells = LATENCY_BINS.map(latencyBin => {{
            const values = rows.filter(row =>
              row.stepCount >= stepBin.min && row.stepCount < stepBin.max &&
              row.duration >= latencyBin.min && row.duration < latencyBin.max
            );
            const rate = successRate(values);
            const maxSteps = sum(values.map(row => row.maxStep));
            if (!values.length) return `<div class="heat-cell empty" title="${{esc(`${{stepBin.label}} steps, ${{latencyBin.label}} latency: no data`)}}"></div>`;
            return `
              <div class="heat-cell" style="${{heatColor(rate, values.length)}}" title="${{esc(`${{stepBin.label}} steps, ${{latencyBin.label}} latency: ${{fmtPct(rate)}} success, n=${{values.length}}, max-step=${{maxSteps}}`)}}">
                <strong>${{fmtPct(rate)}}</strong>
                <span>n=${{values.length}}</span>
              </div>
            `;
          }}).join("");
          return `<div class="heat-label">${{esc(stepBin.label)}}</div>${{cells}}`;
        }}).join("");
        return `
          <div class="heatmap-card">
            <div class="chart-title"><h3>Success heatmap: latency x steps</h3><span>darker = higher success</span></div>
            <div class="heatmap-scroll"><div class="heatmap">${{header}}${{body}}</div></div>
            <div class="hint">Rows are message-step buckets; columns are latency buckets. Empty hatched cells have no simulations under the current filters.</div>
          </div>
        `;
      }}

      function renderSuccessDiagnostics(rows) {{
        const container = document.getElementById("success-diagnostics");
        if (!rows.length) {{
          container.innerHTML = `<div class="empty">No simulations match the current filters.</div>`;
          return;
        }}
        const latencyStats = binRows(rows, LATENCY_BINS, row => row.duration);
        const stepStats = binRows(rows, STEP_BINS, row => row.stepCount);
        container.innerHTML = [
          successStoryMarkup(rows, latencyStats, stepStats),
          renderBinChart("Success rate by latency", latencyStats),
          renderBinChart("Success rate by steps", stepStats),
          renderSuccessHeatmap(rows),
        ].join("");
      }}

      function renderConfigTable(rows, rawRows) {{
        const overallLatency = distribution(rawRows.map(row => row.duration));
        const overallTokens = distribution(rawRows.map(row => row.tokens));
        const overall = {{
          label: "OVERALL",
          family: "overall",
          domains: Array.from(new Set(rawRows.map(row => row.domain))).sort(),
          simulations: rawRows.length,
          tasks: new Set(rawRows.map(row => `${{row.domain}}:${{row.task_id}}`)).size,
          performance: mean(rawRows.map(row => row.reward)) * 100,
          latency: selectedLatency(overallLatency),
          latencyMean: overallLatency.mean,
          latencyMedian: overallLatency.median,
          latencyP95: overallLatency.p95,
          cost: mean(rawRows.map(row => row.cost)),
          tokens: overallTokens.mean,
          tokensMedian: overallTokens.median,
          tokensP95: overallTokens.p95,
          totalCost: sum(rawRows.map(row => row.cost)),
          totalWebCalls: sum(rawRows.map(row => row.webCalls)),
          overall: true,
        }};
        const tableRows = [overall, ...rows];
        document.getElementById("config-table").innerHTML = `
          <thead><tr>
            <th>Config</th><th>Family</th><th>Domains</th><th class="right">Tasks</th><th class="right">Runs</th>
            <th class="right">Performance</th><th class="right">${{latencyMetricTitle()}} latency</th><th class="right">Mean cost</th>
            <th class="right">Total cost</th><th class="right">Mean tokens</th><th class="right">Web calls</th>
          </tr></thead>
          <tbody>
            ${{tableRows.map(row => `
              <tr class="${{row.overall ? "overall" : state.selectedConfig === row.key ? "selected-row" : ""}}">
                <td class="mono">${{esc(row.label)}}</td>
                <td>${{esc(row.family)}}</td>
                <td>${{esc((row.domains || []).join(", "))}}</td>
                <td class="right">${{row.tasks}}</td>
                <td class="right">${{row.simulations}}</td>
                <td class="right">${{fmtPct(row.performance)}}</td>
                <td class="right">${{fmtSec(row.latency)}}</td>
                <td class="right">${{fmtUsd(row.cost)}}</td>
                <td class="right">${{fmtUsd(row.totalCost)}}</td>
                <td class="right">${{fmtTokens(row.tokens)}}</td>
                <td class="right">${{row.totalWebCalls}}</td>
              </tr>
            `).join("")}}
          </tbody>
        `;
      }}

      function renderTaskTable(rows) {{
        const tableRows = aggregateTaskRows(rows);
        const expanded = state.showTaskDetails;
        document.getElementById("task-table-wrap").classList.toggle("expanded", expanded);
        document.getElementById("task-table").innerHTML = `
          <thead><tr>
            <th>Domain</th><th>Task</th><th class="right">Configs</th><th class="right">Runs</th>
            <th class="right">Performance</th><th class="right">${{latencyMetricTitle()}} latency</th><th class="right">Mean cost</th><th class="right">Mean tokens</th>
            ${{expanded ? `
              <th class="right">Latency mean</th><th class="right">Latency median</th><th class="right">Latency p90</th><th class="right">Latency p95</th>
              <th class="right">Latency max</th><th class="right">Latency trimmed 5%</th><th class="right">Max-step runs</th><th class="right">Timeout runs</th>
              <th class="right">Token median</th><th class="right">Token p90</th><th class="right">Token p95</th><th class="right">Token max</th>
              <th class="right">Agent token mean</th><th class="right">User token mean</th>
            ` : ""}}
          </tr></thead>
          <tbody>
            ${{tableRows.map(row => `
              <tr class="${{row.overall ? "overall" : ""}}">
                <td>${{esc(row.domain)}}</td>
                <td class="mono">${{esc(row.task_id)}}</td>
                <td class="right">${{row.configs}}</td>
                <td class="right">${{row.simulations}}</td>
                <td class="right">${{fmtPct(row.performance)}}</td>
                <td class="right">${{fmtSec(row.latency)}}</td>
                <td class="right">${{fmtUsd(row.cost)}}</td>
                <td class="right">${{fmtTokens(row.tokens)}}</td>
                ${{expanded ? `
                  <td class="right">${{fmtSec(row.latencyMean)}}</td>
                  <td class="right">${{fmtSec(row.latencyMedian)}}</td>
                  <td class="right">${{fmtSec(row.latencyP90)}}</td>
                  <td class="right">${{fmtSec(row.latencyP95)}}</td>
                  <td class="right">${{fmtSec(row.latencyMax)}}</td>
                  <td class="right">${{fmtSec(row.latencyTrimmed5)}}</td>
                  <td class="right">${{row.maxSteps}}</td>
                  <td class="right">${{row.timeoutRuns}}</td>
                  <td class="right">${{fmtTokens(row.tokensMedian)}}</td>
                  <td class="right">${{fmtTokens(row.tokensP90)}}</td>
                  <td class="right">${{fmtTokens(row.tokensP95)}}</td>
                  <td class="right">${{fmtTokens(row.tokensMax)}}</td>
                  <td class="right">${{fmtTokens(row.agentTokensMean)}}</td>
                  <td class="right">${{fmtTokens(row.userTokensMean)}}</td>
                ` : ""}}
              </tr>
            `).join("")}}
          </tbody>
        `;
        document.getElementById("task-table").classList.toggle("task-expanded", expanded);
      }}

      function render() {{
        renderMeta();
        renderFilters();
        renderLegend();
        const rawRows = filteredRows();
        const configRows = withDisplayMetrics(displayRows(aggregateConfigRows(rawRows)));
        renderCards(rawRows, configRows);
        renderProgress();
        renderOfatMatrix(configRows);
        renderCharts(configRows);
        renderSuccessDiagnostics(rawRows);
        renderConfigTable(configRows, rawRows);
        renderTaskTable(rawRows);
      }}

      document.addEventListener("change", event => {{
        const domain = event.target.dataset?.domain;
        const family = event.target.dataset?.family;
        if (domain) {{
          event.target.checked ? state.domains.add(domain) : state.domains.delete(domain);
          state.selectedConfig = null;
          render();
        }}
        if (family) {{
          event.target.checked ? state.families.add(family) : state.families.delete(family);
          if (state.families.size === 0) state.families.add(family);
          state.selectedConfig = null;
          render();
        }}
        if (event.target.id === "delta-toggle") {{
          state.delta = event.target.checked;
          render();
        }}
        if (event.target.dataset?.latencyMetric) {{
          state.latencyMetric = event.target.dataset.latencyMetric;
          render();
        }}
        if (event.target.id === "task-details-toggle") {{
          state.showTaskDetails = event.target.checked;
          render();
        }}
      }});

      document.addEventListener("click", event => {{
        const point = event.target.closest?.(".point");
        if (point) {{
          const key = point.dataset.key;
          state.selectedConfig = state.selectedConfig === key ? null : key;
          render();
        }}
      }});

      document.getElementById("clear-selection").addEventListener("click", () => {{
        state.selectedConfig = null;
        render();
      }});

      const tooltip = document.getElementById("tooltip");
      document.addEventListener("mousemove", event => {{
        const point = event.target.closest?.(".point");
        if (!point) {{
          tooltip.style.display = "none";
          return;
        }}
        tooltip.innerHTML = point.dataset.tip;
        tooltip.style.display = "block";
        const left = Math.min(window.innerWidth - 380, event.clientX + 14);
        tooltip.style.left = `${{Math.max(12, left)}}px`;
        tooltip.style.top = `${{event.clientY + 14}}px`;
      }});

      try {{
        render();
      }} catch (error) {{
        document.querySelector("main").innerHTML = `<div class="empty"><strong>Could not render report.</strong><br>${{esc(error.message)}}</div>`;
        console.error(error);
      }}
    </script>
  </body>
</html>
"""
