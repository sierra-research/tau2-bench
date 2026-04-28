# tau2 Result Viewer Website

Browser-based local viewer for tau2 simulation results. It reads the same
`results.json` / directory-format outputs used by `tau2 view`, then serves a
small web UI for inspecting:

- run-level metadata and pass/fail counts
- simulation list with search and filters
- conversation timeline grouped like `tau2 view`
- expanded full-duplex ticks for low-level debugging
- task details and evaluation criteria
- reward/review/authentication details
- verbose voice audio such as `audio/both.wav`

## Run

From the repository root:

```bash
uv run python result_view_website/server.py \
  --results data/simulations/boson_realtime_smoke \
  --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

By default, the server binds to `127.0.0.1`, so it is only reachable from the
same machine. If you are running it on a remote host and want to access it by
hostname, bind to all interfaces:

```bash
uv run python result_view_website/server.py \
  --results data/simulations/boson_realtime_smoke \
  --host 0.0.0.0 \
  --port 8765
```

Then open `http://<host>:8765/`. If the page still does not load, the port may
be blocked by firewall or network policy. In that case, use SSH port forwarding:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<host>
```

Then open `http://127.0.0.1:8765/` on your local machine.

`--results` can be one of:

- a run directory, for example `data/simulations/boson_realtime_smoke`
- a `results.json` file
- the parent simulations directory, for example `data/simulations`

When pointed at `data/simulations`, the UI lets you switch between runs.

## Notes

This viewer is read-only. It covers the main inspection path from `tau2 view`,
but does not implement the terminal viewer's note/task-issue creation actions.
