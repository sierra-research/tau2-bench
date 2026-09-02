# Parallel runner: a shared task queue for tau-bench, tau-voice, and tau-multi

Status: draft design (2026-08-04)

**Scope**: standard tau (text), tau-voice, and tau-multi (the multilingual
pool/preset drivers on `soham/tau-multilingual`). Hyper-tau is explicitly
out of current scope — the seam covers it structurally (see Coverage), and
it joins later once its own `tau2 run` integration lands. A separate
follow-up item, after this ships and the tau-multi drivers are ported:
**remove the tau-multi-specific orchestration** (`pool_driver.py`'s
subprocess-per-cell launching, host-wide process-table budget, cell flock
locks), keeping the pool *spec* and census/reporting logic, which become
producer-side code.

## Background: what exists today, and why it caps out

A `tau2 run` process executes simulations on a `ThreadPoolExecutor` bounded by
`--max-concurrency` (`src/tau2/runner/batch.py`). The bottleneck is not the
providers — it is Python compute in the main process. `tau2 bench-concurrency`
(private repo, `soham/tau-multilingual`, 2026-07-28) measured this directly:

- 1 process at concurrency 30: 272.7 sims/hr, ~100 ms asyncio tick overrun,
  and it could not even fill its own pool (21.6 of 30 slots busy).
- 6 processes at concurrency 10 each: 626.1 sims/hr at ~16 ms overrun,
  60 simulations genuinely in flight. Zero provider errors at any level.

So the ceiling is **per process**, and useful parallelism above ~10 means
**more processes**. The private repo's `tau2 pool` verb builds on that: a
driver loop launches one `tau2 run --auto-resume` subprocess per (language ×
arm) cell, each at concurrency 10, bounded by a host-wide budget of 8 cells,
with per-cell `flock` locks so two drivers never put two writers on one
results directory.

That works, but the unit of parallelism is a *cell* (a whole run config), and
the coordination is indirect: process-table scans, lock files, censusing
results.json between rounds. It is specific to the multilingual pool layout,
and the same need exists in plain tau-bench runs, tau-voice runs, and
hyper-tau. This design generalizes it.

## Goal

One thin, generic mechanism in the **public repo** that:

1. Separates **task production** from **task consumption**: a producer expands
   a run (or a grid of runs) into work units; worker processes pull units and
   execute them.
2. Workers **ask** for work (pull, not push), so the controller can enforce
   global policies at lease time — per-provider concurrency caps, host
   budgets, fair interleaving across runs.
3. Work-unit granularity lands in the **10 s – 10 min** range.
4. Stays adaptable to **multiple machines** later: one controller / task
   creator, workers elsewhere — without redesign.

Non-goals: distributed consensus, external queue infrastructure (Redis,
Celery, SQS), multi-controller setups. One controller, N workers, stdlib +
existing deps only.

## Core abstraction: the work unit

**One work unit = one simulation attempt**: `(run_id, task_id, trial, seed)`
plus a reference to the run's config. This is naturally in the target
granularity band — text sims run seconds to a couple of minutes, voice sims
minutes. Nothing in tau-bench, tau-voice, or hyper-tau needs a finer or
coarser unit; hyper-tau runs are still per-simulation `tau2 run` executions
over different task representations.

Seeds are computed by the **producer**, never the worker, using the stable
sha256-based `derive_task_seed()` (the salted-`hash()` bug was exactly a
cross-process determinism failure, fixed 2026-07-20). A unit is fully
determined by its fields; any worker executing it gets the same simulation.

```python
class WorkUnit(BaseModel):
    unit_id: str            # f"{run_id}/{task_id}/t{trial}"
    run_id: str             # keys into the controller's RunConfig registry
    task_id: str
    trial: int
    seed: int
    provider: str           # the resource this unit consumes (see limits)
    attempt: int = 0
```

A `WorkUnit` is **in-memory only and never persisted**. It is not a second
source of truth beside `SimulationRun` — it is derived state: the producer
computes the queue as (task list × trials) minus `done_runs` from the
checkpoint, exactly the diff `try_resume` computes today. `SimulationRun`
cannot play this role itself: it is the *result* (heavyweight — full message
transcript, audio refs) and does not exist until a sim finishes, while the
queue needs a small descriptor for work that has not happened yet, plus
scheduling fields (`attempt`, `provider`, lease state) that have no business
in the results schema. Conversely, persisting WorkUnits would create
placeholder records in the results tree — the same pathology as
infrastructure-error files convincing `--auto-resume` a cell was complete.

## Components

```
                 ┌──────────────────────────────────────────┐
                 │ Controller (one process)                  │
  RunConfig(s) ─▶│  Producer: run/pool/preset → WorkUnits    │
                 │  Queue + lease table (in memory)          │
                 │  Limits: per-provider caps, global cap    │
                 │  Sink: checkpoint writer (results.json)   │
                 └────────────▲──────────────┬───────────────┘
                              │ lease/result │ spawn (local mode)
                     ┌────────┴────────┐ ┌───┴─────────────┐
                     │ Worker process 1 │ │ Worker process N │  each holds up to
                     │  slots ≤ 10      │ │  slots ≤ 10      │  `--slots` sims in
                     └─────────────────┘ └─────────────────┘  flight (threads)
```

### Controller

- **Producer**: expands one or more `RunConfig`s into work units. `tau2 run`
  produces units for one run; a pool/preset/matrix driver produces units for
  many runs into the *same* queue. The queue is heterogeneous by design —
  interleaving across providers falls out of the lease policy instead of
  being orchestrated per-cell.
- **Queue + leases**: in-memory FIFO per provider with a lease table. A lease
  has a fixed 300s TTL kept alive by worker heartbeats (every ~30s from a
  dedicated thread — the worker's main loop can block for minutes posting
  multi-MB voice results, so heartbeating from it starves under load); an
  expired lease requeues the unit with `attempt += 1`, up to a retry cap.
- **Admission control**: at lease time the controller checks (a) the global
  in-flight cap (host budget analog), (b) the unit's per-provider cap
  (`--provider-limit openai=40,gemini=20`). This is the answer to "workers
  ask someone for a task": the *controller* decides which unit a worker gets,
  so policy lives in one place.
- **Single writer**: workers return the finished `SimulationRun`; the
  controller writes checkpoints via the existing `create_checkpoint_fns`
  machinery. One writer per results.json eliminates the entire class of
  problems the pool driver's flock/pid-file/process-table logic exists to
  contain. A result with `termination == infrastructure_error` is not
  checkpointed — it is requeued (up to the retry cap), which turns the
  pool's census→drop→resume repair loop into a one-line policy. "Not
  checkpointed" means no entry in results.json / no `simulations/<id>.json`
  record; the run directory and any per-attempt artifacts the worker wrote
  (`artifacts/task_<id>/sim_<id>/`, logs, audio) stay on disk for
  post-mortem, marked discarded via `sim_status.json` — the same convention
  the hallucination-retry path uses today. This mirrors current behavior:
  `try_resume` strips infra-error sims from the checkpoint but leaves their
  artifact directories in place.

### Workers

A worker is `tau2 worker --controller <addr> --slots 10`. Its loop:

1. Ask the controller for a lease (blocking long-poll, one request per free slot).
2. Build and run the simulation — the existing Layer 1/2 code
   (`build_orchestrator`, `run_simulation`) unchanged.
3. POST the result; repeat.

`--slots` defaults to the measured per-process sweet spot (10). Workers hold
no state worth preserving: kill one and its leases expire back into the
queue. Workers never touch results.json.

### Transport

The seam is a four-call protocol, defined once:

```python
class TaskSource(Protocol):
    def lease(self, worker_id: str) -> Optional[LeasedUnit]: ...
    def complete(self, unit_id: str, result: SimulationRun) -> None: ...
    def fail(self, unit_id: str, error: str) -> None: ...
    def heartbeat(self, unit_id: str) -> None: ...   # extends the lease
```

Two implementations:

- **`LocalTaskSource`** — plain in-process object. `tau2 run` without
  `--workers` keeps today's behavior exactly (the ThreadPool calls it
  directly); zero new moving parts for the common case.
- **`HttpTaskSource`** — the controller serves the same four calls over HTTP
  (FastAPI is already a dependency via the domain servers); the worker-side
  client is ~50 lines of `httpx`. In local multi-process mode the controller
  binds `127.0.0.1:<random port>` and spawns its own workers as subprocesses
  with the address in an env var. **Multi-machine later is only**: bind a
  routable address, add a bearer token, and start `tau2 worker` on other
  hosts pointed at it. No new protocol, no new code paths.

Result payloads are JSON (`SimulationRun` already serializes). Voice audio
artifacts are the one wrinkle: locally, workers share the filesystem and
write audio under the run dir as today; cross-machine, the worker streams the
artifact in the `complete` call (or writes to shared storage). This is
explicitly deferred — the protocol carries an `artifacts` field from day one
so it needs no version bump.

## CLI surface

```
tau2 run ... --workers 6                     # 6 processes × --max-concurrency = in flight
tau2 run ... --provider-limit openai=40      # cap at lease time
tau2 worker --controller http://host:8321 --slots 10   # remote/extra workers
```

- Per-worker slots reuse the existing `--max-concurrency` knob (default 10)
  rather than adding a second flag: `--workers 6 --max-concurrency 10` = 60
  in flight, and the flag keeps its exact current meaning when
  `--workers 0`. The standalone `tau2 worker` verb takes `--slots` since it
  has no run config of its own.
- `--workers 0` (default): today's in-process ThreadPool, unchanged.
- `--workers N`: controller mode; the run's own process does no simulation
  work, matching the finding that main-process compute is the bottleneck.
- Pool / preset / hyper-tau drivers stop launching subprocess-per-cell and
  instead register their runs with one controller and produce units. The
  pool's *census* logic (usable vs infrastructure_error, dedup, coverage)
  stays — it becomes the producer's "what units do I still owe" question at
  startup and after each round, computed from checkpoints exactly as now.

## Coverage: standard tau, tau-voice, hyper-tau

Standard tau (text) and tau-voice are covered **by construction**: both
execute through `run_tasks` in `runner/batch.py`, which is exactly where the
`TaskSource` seam goes. The hyper-tau branch's `batch.py` is unchanged from
main (its runner diff touches only `build.py` and `simulation.py`), so the
seam lands identically there.

Tau-multi is covered at both layers: its cells are ordinary `tau2 run`
invocations (so the seam applies), and its drivers (`run-preset`, `tau2
pool`) become producers in the driver-port step.

Hyper-tau is **out of current scope** but structurally covered, with two
paths for later:

1. **`tau2 run --domain hyper_*`** — hyper-tau's own README declares wiring
   hyper domains through `runner.batch.run_tasks` as its follow-up (the
   registered factory is currently a placeholder; `tau2 hyper-tau` is the
   working CLI). Once that wiring lands, a hyper episode flows through the
   same seam with zero parallel-runner work: one outer episode = one
   WorkUnit.
2. **The current `tau2 hyper-tau` orchestration** (outer_orchestrator,
   eval_transfer) runs its own ThreadPools and would be a driver port, same
   shape as `run_multiple.py` and the pool driver.

Two hyper-specific caveats, acknowledged for that later work rather than
solved here:

- **Granularity**: an outer episode (Developer iterating a policy, running
  inner evals) can exceed the 10-minute band. The protocol already carries
  `heartbeat`, so long units are safe — a hyper run just sets its lease TTL
  from its own ceiling. Splitting inner evals into sub-units is a possible
  later refinement, not needed for correctness.
- **Weight**: one hyper unit fans out inner tau sims inside the worker
  (`TAU2_HYPER_INNER_MAX_WORKERS`, default 32), so it consumes far more
  provider quota and CPU than one text sim. Leases carry an optional
  `weight` so a hyper unit can count as more than 1 against slots and
  provider caps; until inner evals are sub-units, that weight is an
  estimate.

## Persistence and resume

Persistence is the existing checkpoint mechanism, unchanged — the queue is
never persisted because it is recomputable from the checkpoint.

- **During a run**: the controller appends each finished sim through
  `create_checkpoint_fns` exactly as `run_tasks` does today (monolithic
  results.json rewrite, or per-sim file + index update in dir format).
- **On kill + restart**: the producer calls `try_resume`, which loads the
  checkpoint, verifies config/task compatibility, strips
  `infrastructure_error` sims, and returns `done_runs` keyed
  `(trial, task_id, seed)`. The producer emits WorkUnits only for cells not
  in `done_runs`. `--auto-resume` keeps its meaning (skip the prompt).
- **Grid runs resume per run, exactly as `run_multiple.py` does today**
  (`src/experiments/tau_voice/run_multiple.py`): each (domain × provider ×
  complexity) combo keeps its own results dir under the base dir, and
  re-invoking the driver re-registers every combo — `try_resume` per combo
  skips what is done. A killed grid resumes mid-grid with nothing special.
- **In-flight sims at kill time are lost**, same as today: a sim that never
  reached `save_fn` was never checkpointed, and its `(trial, task_id, seed)`
  cell is simply re-emitted on restart. Worker death is the cheaper case and
  does *not* lose the run: the lease TTL expires and the unit requeues while
  the controller keeps running.

### Multiple top-level runs

Supported, with one rule: **one controller per results directory**. The
preferred shape for "many runs at once" (a pool, a preset matrix, unrelated
experiments) is many runs registered in **one** controller — that is the
whole point of the heterogeneous queue, and it is what makes provider caps
and the host budget actually global. Two independent `tau2 run` invocations
with different `save_to` targets also work (each is its own controller), but
their caps are per-controller: two controllers each capped at
`openai=40` are 80 against the provider. So: fine to do, but split the
budgets yourself, or point both producers at one controller. Two controllers
on the *same* results dir is refused at registration by the same flock guard
the pool driver uses today.

## What stays the same

- **Storage locations: unchanged, for both levels.** The run-level
  results.json (monolithic or dir format with `simulations/<sim_id>.json` +
  index) stays where it is under `DATA_DIR/simulations/<save_to>`, and
  per-sim artifacts stay under `artifacts/task_<task_id>/sim_<sim_id>/`.
  The only delta is *which process* calls the existing save functions (the
  controller instead of each run process). Cross-machine workers later need
  an artifact-upload step, but the on-disk layout they land in is the same.
- `RunConfig` / `VoiceRunConfig` / checkpoint format: unchanged.
  `--auto-resume` still works — resume is just the producer not emitting
  units that already have usable results.
- `run_simulation`, orchestrator build, retries-within-a-sim: unchanged.
- Determinism: unchanged contract, now structurally enforced (seed computed
  once, in the unit).

## Failure model

| Failure | Handling |
|---|---|
| Worker dies mid-sim | Lease TTL expires → unit requeues (`attempt += 1`) |
| Sim ends `infrastructure_error` | Not checkpointed; requeued up to retry cap |
| Unit exhausts retries | Checkpointed as failed; counted in exit code |
| Controller dies | Workers' leases go nowhere; workers exit on connection loss. Restart resumes from checkpoints (producer re-owes unfinished units) |
| Two controllers on one results dir | Same flock guard as today's pool lock, kept as a belt-and-braces check at run registration |

## Test plan

### Automated

- `tests/test_runner/test_work_queue.py` — WorkQueue semantics, no I/O:
  FIFO lease/complete, per-provider caps gate leasing, global cap, lease-TTL
  expiry requeues with `attempt + 1`, attempt cap moves a unit to dead,
  `fail()` requeue/dead, heartbeat extends a lease, all-resolved signal,
  stale complete (after expiry-requeue) is ignored.
- `tests/test_runner/test_controller.py` — controller HTTP contract via
  in-process ASGI transport (no sockets, no subprocesses): lease returns
  unit + serialized config + task; complete writes exactly one checkpoint
  entry (controller is the single writer); an `infrastructure_error` result
  is requeued, then written once attempts are exhausted; `/fail` requeues
  then produces a placeholder infra sim when dead; duplicate/stale completes
  are ignored. Worker-loop test drives `tau2.runner.worker` against the same
  ASGI app with `run_unit` monkeypatched to return fabricated sims: all
  units complete, worker exits on done, results land in the checkpoint.
- `tests/test_runner/test_local_seam.py` — `run_tasks` with `workers=0` and
  `run_single_task` monkeypatched: N tasks × trials all execute, checkpoint
  written, a second invocation resumes and runs nothing. Guards the
  zero-behavior-change claim without LLM calls.

### Manual

1. Baseline: `tau2 run --domain mock --agent llm_agent --user user_simulator
   --num-tasks 2 --num-trials 2 --max-concurrency 2 --save-to <dir A>`
   (workers=0, requires LLM keys).
2. Same command with `--workers 2 --save-to <dir B>`: completes, per-task
   rewards comparable, results.json structure identical, worker logs appear
   under the run dir.
3. Kill the `--workers` run mid-flight (Ctrl+C / SIGTERM); re-run with
   `--auto-resume`: completed sims are skipped, the rest finish, no
   duplicate `(task_id, trial)` pairs in results.json.
4. `run_multiple.py --workers 2` smoke over a small grid (guarded by
   available provider keys); verify per-combo results dirs match the
   sequential layout.

## Rollout

1. **Extract the seam** — introduce `TaskSource` + `LocalTaskSource`, refactor
   `run_tasks` to consume it. Pure refactor, no behavior change (public repo).
2. **Controller + worker** — `HttpTaskSource`, `tau2 worker`, `--workers N`
   spawn path, lease TTL/retry, per-provider limits. Validate against
   `bench-concurrency` numbers: 6×10 should reproduce ~626 sims/hr.
3. **Port the drivers** — `run_multiple.py` (public, tau-voice) and
   pool/preset (private, tau-multilingual) become producers over the shared
   controller. `run_multiple.py` is the simplest port and the proof of the
   "multiple top-level runs" story: today it runs its grid combos
   *sequentially*, one `tau2 run --auto-resume` subprocess at a time; as a
   producer it registers all combos in one controller and the grid runs
   concurrently under the global caps.
4. **Remove the tau-multi-specific implementation** (separate item, after 3
   has soaked) — delete `pool_driver.py`'s subprocess-per-cell launching,
   the host-wide process-table budget (`live_cell_targets`), and the cell
   flock/pid machinery; keep `pools.py` (the spec is the record of what ran)
   and the census/dedup/coverage logic as producer-side code.
5. **(Later, out of scope) hyper-tau** — rides the seam once its
   `tau2 run --domain hyper_*` wiring lands; the `tau2 hyper-tau`
   orchestration port is its own item.
6. **(Later) multi-machine** — auth token, routable bind, artifact upload.
