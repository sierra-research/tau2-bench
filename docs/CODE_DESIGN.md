# tau2-bench Code Design Standards

This document codifies the design conventions of this codebase. It is descriptive of
the core repo (these patterns already hold there) and **prescriptive for all new
code**: match these standards or raise the divergence explicitly in review.

## Mandates

Two standing mandates govern how we build and change this codebase:

1. **This is a machine, not a pile of scripts.** The factory / judges / annotation
   pipeline must be able to churn out languages, runs, judgments, and annotation
   artifacts again and again, reliably.
   - Anything run more than once is a `tau2` CLI subcommand with a tested code path.
     `scripts/` holds only environment glue (paths, mounts), and shrinks toward zero.
   - LLM content generation happens through **fixed, versioned, in-code prompts**
     (module-level constants with observable `call_name`s), never through a coding
     agent improvising content or filling gaps at run time. Claude-Code skills may
     orchestrate CLI verbs; they must not author content.
   - Artifact generation is idempotent: same inputs + same code version → same
     outputs. Long jobs are resumable (re-run fills gaps; it does not redo work).
2. **Research code: rip out and redo over compat shims.** Results, packs, and
   annotation artifacts are regenerable. When code is below the standard in this
   document, rewrite it — do not wrap it, alias it, or preserve legacy schemas.
   No deprecation aliases, no migration shims, no "keep the old column names".

## Data modeling: pydantic everywhere

- Every data structure that crosses a function/module/process boundary is a
  **pydantic v2 `BaseModel`**. Fields use `Annotated[T, Field(description=...)]`
  with a real description. Closed sets are `Enum`s (str-backed so they serialize
  cleanly) or `Literal`s — never magic strings.
- Raw `dict`s do not cross module boundaries. A `dict[str, Any]` is acceptable
  only as a short-lived intermediate that is `model_validate`d before leaving the
  function that built it (e.g. `yaml.safe_load` → normalize → `model_validate`).
- **LLM outputs are validated, not trusted**: parse the reply, then
  `ResponseModel.model_validate(...)`. Coercion of known model quirks (stringly
  bools, missing enum fields) lives in `field_validator`s on the response model,
  not in ad-hoc helper functions. A reply that still fails validation is a loud
  ERROR outcome, never a silently-dropped item.
- Serialized contracts (CSV headers, sheet columns, JSON artifacts) are
  **generated from models**, not hand-maintained string lists. Ingest is
  `model_validate` per row.
- Canonical examples: `src/tau2/data_model/` (messages, tasks, simulation),
  `src/tau2/multilingual/schema.py` (LanguagePack with cross-ref validators).

## Configuration

- Defaults are `DEFAULT_<FEATURE>_<SETTING>` constants in `src/tau2/config.py`.
- Runtime knobs live on pydantic **RunConfig objects** (`TextRunConfig`,
  `VoiceRunConfig`, nested settings models like `JudgesConfig`) — never long flat
  argument lists threaded through call stacks.
- Config flows one way: `config.py` default → RunConfig → recorded on the result
  (`Info` / `*Info` models). Whatever influenced a result must be readable from
  the result.
- No environment variables as a third config channel for behavior toggles. Env
  vars are for credentials and machine-local paths only.

## LLM calls

- **One seam**: `tau2.utils.llm_utils.generate()`. It owns retries
  (`DEFAULT_MAX_RETRIES`), cost/usage accounting, per-call debug logging, and
  provider quirks (reasoning models, response bridging). Do not call
  `litellm.completion()` directly.
- Every call site passes a stable, greppable `call_name`. Call names are an
  observable contract: tests and recorded fixtures match on them — do not rename
  casually.
- Prompts are module-level constants (or pure functions of typed inputs) living
  next to their pipeline stage. Prompt text is calibrated material: version it
  (a `*_PROMPT_VERSION` tag, bumped on any retune) so a prompt change between
  reruns is visible on the results it produced. Do not hash prompt text — a
  version tag carries the same signal without the ceremony.
- Judges and generators record their `model`, `model_args`, and prompt version on
  the result they produce. They do not thread per-call cost onto results;
  cost/usage accounting lives inside `generate()` (logs), not in result schemas.

## CLI

- argparse, single `tau2` entry point. Each package owns its subcommand surface in
  its own `cli.py` exposing `add_<name>_args(parser)`, registered from
  `src/tau2/cli.py`. (See `tau2 factory` for the canonical shape.)
- No `argparse.REMAINDER` forwarding to module `main()`s.
- Verbs are pipeline stages. If a workflow step matters, it is reachable as a
  `tau2` verb — end-to-end flows (add a language, rerun the benchmark, rebuild
  annotation artifacts) must be executable purely via `tau2` commands.

## Results & artifacts

- Results are typed (`Results` / `SimulationRun`) with the dual json/dir format;
  large runs are consumed via streaming (`iter_simulations`), not whole-file loads.
- Result types live in `src/tau2/data_model/` as contracts; pillar packages
  populate them. Scoring axes (reward, nativeness, delivery) are **decoupled
  sibling fields** — post-hoc judges never fold into reward.
- **Human-facing annotation artifacts** (packets, sheets) carry a
  manifest/provenance written by the same code that writes the artifact, and
  ingest validates it. Machine-to-machine exports (e.g. judge verdict JSONLs)
  do not need manifests — the typed records themselves carry what downstream
  consumers read; do not add sidecar provenance files nobody consumes.
- Blind nativeness annotation reuses the interaction/audio call packet. Call-level
  factors label the whole call; utterance-level factors pin stable agent-turn ids
  and exact displayed text. Machine candidates are reviewed only in separate
  call- and utterance-level adjudication packets, never exposed in the blind form.

## Errors, logging, concurrency

- loguru everywhere; no stdlib `logging`, no bare `print` in library code.
- Fail loud and local: a failed unit of work (one judge call, one row) is recorded
  as an ERROR outcome and counted; it never masks as success, silently drops, or
  aborts the whole batch. Retries live inside `generate()`, not at call sites.
- Concurrency: `ThreadPoolExecutor` with explicit worker counts from config;
  per-worker event loops where async is involved (see `runner/batch.py`). Shared
  registries use double-checked locking (see `multilingual/loader.py`).

## Registries & package boundaries

- Global registries are dict-backed singletons with validation at registration
  time (`src/tau2/registry.py`).
- The three multilingual-era pillars are top-level siblings with one-directional
  dependencies:
  - `multilingual/` (language factory) — may not import evaluator/annotation;
    exposes typed state (`TranslationState`, `FactoryProvenance`).
  - `judges/` (runtime judging) — imports data_model; exposes typed verdict
    records via `judges.export` for downstream consumers.
  - `annotation/` (human-facing artifacts) — imports `judges.export`,
    `data_model`, and factory's typed surface; nothing imports annotation.
- Closed catalogs (e.g. judge factors) are guarded by coverage tests that fail on
  unknown or unreferenced entries.
- Nativeness judge factors use one catalog-owned base question/rule plus
  language-pack examples. Rubric prose is short, direct, and concrete: state what
  the agent may do, what violates the rule, and show natural and violating examples
  in the target language. Do not put research history, calibration notes, abstract
  linguistic prose, or evaluator jargon in runtime prompt fields.

## Testing

- pytest; fixtures in `conftest.py` load real domain/pack objects; tmp_path for
  filesystem work. Lint = ruff (E4, E7, E9, F, I); run via Makefile targets.
- LLM calls are mocked by monkeypatching `generate` at the consuming module and
  returning `SimpleNamespace(content=json.dumps(payload))` — tests then exercise
  the real parse/validate path.
- Test fixtures for serialized contracts are built **from the models**, so tests
  cannot drift from the contract they guard.
- Every closed catalog / vocabulary gets a coverage-guard test; every pipeline
  gets an offline end-to-end test with recorded LLM fixtures keyed by `call_name`.
