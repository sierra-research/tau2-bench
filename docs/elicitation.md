# τ-Entity: English entity-capture benchmark

**τ-Entity** is the current name for the benchmark previously called
**elicitation** or **τ-elicitation**. The `intake` domain, CLI flags, and recorded
`elicitation` fields keep their existing names.

The τ-Entity benchmark measures whether a voice agent can collect an exact
missing value during an outbound callback. The cover stories are a clinic, an
auto service, and a hotel. These are record-capture tasks; the agent provides no
medical, automotive, or travel advice.

The domain and task set are both named `intake`. Its default strategy is
agent-directed: the agent chooses how to collect each value. The scaffolded
strategy is an ablation selected with `--elicitation-strategy scaffolded`.
Both strategies share the frozen English task values, database, normalization,
and final-record reward contract:

| Strategy | Agent behavior | Channels |
| --- | --- | --- |
| `agent_directed` (default) | Chooses when to repeat, verify, spell, or read back a value | Voice |
| `scaffolded` | Follows the supplied collection protocol; the voice policy requires spelling and explicit read-back confirmation | Voice and text |

Their tool schemas differ: scaffolded submission includes a confirmation
self-report; agent-directed submission omits it and its caller has private
spelling/read-back measurement tools. The policy files are
`agent_directed_policy.md`, `scaffolded_policy.md`, and
`scaffolded_policy_text.md` under `data/tau2/domains/intake/`.

Once a task is selected, its callback order is included in the agent's prompt
before the call starts: the record ID, person, organization, and ordered missing
fields with their descriptions. The live database is initialized from that same
task setup, with the requested fields blank. The agent collects their values
from the conversation; it does not call a work-order lookup tool.

The current elicitation prompt version, `1.5.0`, tells the voice user simulator
that it receives a transcript of the agent's audio. Agent-directed read-backs
ignore inaudible spacing and punctuation differences unless the agent explicitly
spells or describes those details aloud; the simulator still corrects spoken errors.
Both caller guidelines describe an outbound contact and keep lookups private.
Agent-directed logging does not require a read-back.
The caller preserves units from its lookup: “50 mg” is spoken as “fifty M, G,”
while “50 milligrams” is spoken as “fifty milligrams.”

The `intake` package, data directory, task-ID prefix, and `tau2 intake-tasks`
command identify the common task collection. Strategy selection configures the
matching policy, caller instructions, and submission schema at runtime; it does
not select a different domain or task set.

## Task selection and scoring

The `base` split is the full 200-task benchmark: ten entity families, two
difficulty tiers, and ten tasks per family/tier. Each task has one missing
field. The families are person names, codes, phone numbers, dates, times,
addresses, properties, email addresses, medications, and coined names.

Run the full `base` split for a reported result. A `--num-tasks` prefix is only
a smoke test: tasks are grouped by family, so a prefix does not represent the
whole benchmark. Use explicit `--task-ids` to select a smoke sample across
families and tiers. This release does not include composition or staged bands.
Easy and hard tiers are both included by default; use `--task-ids` for an
explicit selection of tasks from either tier.

Reward compares the final database with the database produced by replaying the
golden submission. The write tool normalizes accepted surface variants; wrong
values, omitted fields, and missing submissions fail. The released
normalization preserves unit wording: `mg`, `milligram`, and `milligrams` are
distinct. Scoring version `1.2.0` also preserves decimal points within numbers:
`0.5 mg` and `05 mg` are different values. It does not rewrite dates or times;
only surrounding whitespace is removed. Submit dates as `YYYY-MM-DD` and times
as `H:MM AM/PM`, without a leading hour zero. For example, `June 12, 2025` does
not match `2025-06-12`, and `14:30` does not match `2:30 PM`. The agent must
convert information provided during the conversation into the required field
format and ask for missing or ambiguous components instead of inventing them.
Email punctuation, changed digits, and AM/PM remain identity-bearing.
Pronunciation and spelling rules affect elicitation runs only; they do not
alter airline, retail, telecom, or banking behavior.

## Install and run

The domain, frozen tasks, and generator use core dependencies. Voice execution
uses the existing voice extra and the credentials required by the selected
agent provider, caller language model, ElevenLabs TTS, and Deepgram STT.

```bash
uv sync --extra voice --extra dev
uv run tau2 run --domain intake --audio-native --benchmark none \
  --audio-native-provider openai --audio-native-model gpt-realtime-2 \
  --reasoning-effort minimal \
  --user-llm gpt-5.5 --user-llm-args '{"temperature": 0.0, "reasoning_effort": "xhigh"}' \
  --complication-rate 1 \
  --task-split-name base --num-trials 1 --seed 42 \
  --save-to elicitation-example
```

This runs the default agent-directed benchmark. To run the scaffolded ablation,
add `--elicitation-strategy scaffolded` to the same invocation and choose a
different `--save-to` name. Text elicitation also requires
`--elicitation-strategy scaffolded`.

Pin the provider, model, reasoning effort, caller model, speech complexity,
seed, strategy, and complication settings when comparing runs. Use
`tau2 run --help` for the flags supported by the checked-out runtime. The example
explicitly requests caller model `gpt-5.5` with `xhigh` reasoning. Without
explicit caller settings, τ-Entity uses `gpt-5.5` with `low` reasoning and
temperature zero.
The default maximum voice duration is 240 seconds; override it explicitly
with `--max-steps-seconds` when needed.

Strategy and complication settings live in the typed `RunConfig.elicitation`
object: `strategy` is `agent_directed` by default or `scaffolded` for the ablation.
Both strategies default to complication rate `1`.

Complication kind weights come from the fixed, family-specific catalog.
Each call samples at most one complication. Eligibility depends on the entity
family, its value, and the channel: for example, pronunciation substitutions
require a curated variant, and spelling complications apply to voice calls.
A sampled spelling complication is exercised only if a spelling sequence
actually occurs.

`--complication-rate` rescales the bank's full distribution before checking
whether each kind is feasible for the task and channel. Infeasible kinds retain
their probability mass as a clean call, so rate `1` does not guarantee a
complication on every call. The catalog's relative kind weights are preserved.
`--complication-rate 0` requests clean calls. Explicit elicitation settings are
rejected for other domains.

Results use main's ordinary simulation output. Re-run the same invocation
with `--auto-resume` and the same `--save-to` to resume it, and inspect it with:

```bash
uv run tau2 view --file data/simulations/elicitation-example/results.json
```

The recorded `Info.elicitation` contains the resolved strategy and complication
settings. `Info.elicitation_scoring_version` and `Info.elicitation_prompt_version`
identify the normalization and prompt contracts. `Info.elicitation_catalog_version`
identifies the stimulus distribution, including calls without a sampled complication.
`Info.elicitation_spell_detection_version` records the detector used for
agent-directed caller-effort events.
Resume rejects changed benchmark conditions even with `--auto-resume`.
Each simulation's
`complication` records the exact sampled stimulus, parameters, catalog
version, and pronunciation substitution when present. Interpret scores with
these versions and conditions; older scoring or caller conditions are a
different benchmark setting.
Post-hoc review uses the simulation's recorded task-specific policy and injected
caller instruction; it does not resample complications from the current catalog.

An offline generation or verification run needs no credentials, downloads,
pre-generated audio, pronunciation dictionary installation, or language pack.
Voice runs synthesize caller audio through main's ordinary provider path.
Elicitation voice/persona draws use a stable task-ID digest, making the seeded
draw reproducible across workers and resume. These released draws can differ
from historical research runs that used process-dependent task-ID hashing;
exact historical audio equivalence is not claimed. Complication sampling
retains the source algorithm and catalog version.

## Reproduce and verify the frozen tasks

These optional maintainer commands check or regenerate the frozen task
artifacts. Ordinary evaluations use the shipped tasks directly and do not
require this step.

```bash
uv run tau2 intake-tasks enumerate
uv run tau2 intake-tasks verify
uv run tau2 intake-tasks freeze --out /tmp/elicitation-reproduction
uv run tau2 intake-tasks verify --data-dir /tmp/elicitation-reproduction
```

The default seed is `20260824`; generator version `7.0.1` and call-frame version
`4.5.0` preserve the source freeze. The checked-in `tasks.manifest.json`
records the seed, versions, bank SHA-256 values, sampled cells, and per-task
draws. The generator uses fixed templates and seeded random streams, with no
LLM calls. It validates every task by replaying the real domain toolkits before
writing any artifact. Repeating a freeze leaves unchanged files untouched.
`verify` reconstructs the recorded draw and fails on artifact drift without
repairing files.

`freeze` without `--out` targets the checked-in domain files. Review any intended
change to the banks, task generator, or call frame, update its version, and
regenerate visibly; do not hand-edit generated task JSON. Use a separate
output directory for experiments with another seed.

The generator retains all fifteen small source bank files because validation
and provenance cover that complete input set. The five additional banks
(amounts, insurance plans, vehicles, rate plans, and shops) contribute no
tasks to the released selection. Bank authoring, raw name corpora, online
registry screens, and research analysis are outside this release. See
`data/tau2/domains/intake/ENTITY_PROVENANCE.md` for the source snapshot.

## Validation

The offline tests cover byte-for-byte reproduction, bank schema and catalog
coverage, seed changes, split and manifest drift, normalization, all 200 golden
submissions, and incorrect or incomplete submissions. Run the focused domain
suite, then the core and affected voice suites before release:

```bash
uv run pytest tests/test_domains/test_intake
make test
make test-voice
make check-all
```

A live release smoke should select easy and hard tasks from every family and
exercise clean calls, spelling requests, and pronunciation complications with
explicit settings. Record the invocation and resulting artifact; an offline
pass does not establish provider connectivity or speech quality.
