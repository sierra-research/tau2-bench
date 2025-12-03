# Judge Viewer

A local web UI for viewing τ² simulation results alongside LLM-judge outputs.

## Quick Start

```bash
cd tools/judge_viewer
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

## Usage

1. **Pick a dataset** from the dropdown to load a pre-configured simulation + judge outputs bundle.  
   *or* **manually upload**:
   - A simulation JSON from `tau2 run`
   - One or more `.summary.json` files from `run_custom_judge.py` (Cmd/Ctrl+click to select multiple)

2. **Browse** – Select a simulation on the left. The center pane shows the conversation transcript; the right pane shows all judge verdicts for that task with prompt details and agreement status.

## Notes

- Judge records are matched to simulations by `task_id`.
- Pre-configured datasets are defined in `data/datasets.json`. Add your own files under `data/` and update the config.
- All parsing is client-side; files stay on your machine.
