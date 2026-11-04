# Published language packs

To create a language or extend its domains, see [Language authoring](language-authoring.md).

> Terms such as *pack*, *cell*, *binding revision* and *pin* are defined in the
> [glossary](glossary.md).

The ordinary runner accepts `--language` for both text and voice runs:

```bash
tau2 run --domain retail --language es
tau2 languages list
tau2 languages show es
tau2 languages validate es
```

English and every other language use the same published-pack contract and runtime
path. Voice runs default to every language in the released benchmark manifest.
For custom/text runs, omitting `--language` uses ordinary domain tasks and persona
selection. Select custom voice mode with `--benchmark none`.

## Banking pilot binding

The reviewed Hindi, Korean, Mandarin, Portuguese and Spanish banking regional bundles are available as the
separate `banking` binding. It is pinned to each unchanged V2 language
pack and is excluded from the released benchmark manifest. A custom run resolves
the binding from its language and domain. Selection loads published artifacts; it does not generate
localization content or edit `pack.yaml`:

```bash
tau2 run --audio-native --benchmark none --domain banking_knowledge \
  --language ko --retrieval-config bm25 \
  --audio-native-model gpt-realtime-2 --reasoning-effort low --num-tasks 5
```

Use `hi`, `ko`, `zh`, `pt` or `es`. New banking runs select the versioned
`banking-dates-v1` extension. It adds voice-only guidance that interprets scenario
dates as MM/DD/YYYY and speaks the same calendar date in local order: day–month–year
for Hindi, Spanish and Portuguese; year–month–day for Korean and Mandarin. The
same versioned extension gives the debit-card ordering document a domestic
shipping rule for India, South Korea, China, Spain or Brazil, respectively. It
preserves the ban on international shipping. The shared English document remains
US-only, and the previously scored `banking` bundles remain unchanged. The
original `banking` extension remains loadable from a recorded
`RunConfig.pack_extension` when reproducing earlier runs. Neither extension
changes canonical dates, database values or tool formats. In Spanish, the
versioned extension also corrects four caller descriptions: the stated residence
in tasks 077, 091 and 097 now matches the Spanish address, and task 082 uses the
localized merchant name consistently. Gold actions and earlier scored bundles
remain unchanged. In Portuguese, task 082 consistently names Academia Corpo
Pleno Premium and task 097's stated residence now matches its São Paulo address.
These two caller-text edits also have their own versioned bundle revision. The
Korean and Mandarin bundles also correct name spacing and distinct-person name
collisions. Native review of the five replacement Korean names and two
replacement Mandarin names is complete.
Source-bound correction records are retained as regression fixtures under
`tests/test_multilingual/fixtures/banking_name_review/` and tested against the
versioned name maps. Earlier scored bundles and their review status are unchanged.
The extension records its identity and the exact regional
bundle digests in the run results. Banking knowledge-base prose remains English,
while mapped entity references and this shipping rule follow the regional bundle.
The agent guidance instructs the agent to answer
in the selected language. Released packs and benchmark cells keep their digests.

To publish additional approved bundles without editing a released pack, first use
`tau2 factory compile-approved-regional` with the exact map approval, source
inventory and task metadata, then pass the compiled bundles to
`tau2 factory publish-pack-extension`. The factory writes the extension manifest
last, after its metadata and regional bundles are complete.

## Intake pilot binding

The reviewed Hindi, Korean, Mandarin, Portuguese and Spanish intake regional bundles have a separate
`intake` binding. A custom run resolves it from the language and domain, as for banking,
without changing banking's artifacts or the released benchmark:

```bash
tau2 run-pass3 --pass3-mode crossed --audio-native --benchmark none \
  --domain intake --language ko --save-to ko_intake_rcs \
  --audio-native-model gpt-realtime-2 --reasoning-effort low --num-tasks 5 --dry-run
```

Use `hi`, `ko`, `zh`, `pt` or `es`. Remove `--dry-run` only when ready to start live calls.
Selecting the binding loads already approved artifacts; it does not generate
localization content. The normal intake strategy and complication-rate options
remain available. The three crossed conditions share task framing and caller
assignment while sampling their voice effects independently.

## Construction boundary

A run combines three independent inputs:

- Run conditions: models, reasoning settings, duration, task selection and acoustic
  complexity.
- Domain: business policy, tools, database schema and success criteria.
- Language: localized tasks and identities, reviewed persona content, voices,
  greetings, examples, inserts and language-specific provider settings.

The runtime loads and validates the selected language/domain binding, resolves the
canonical task and gender-matched persona, and renders every supported prompt
variant before constructing an environment or provider. Missing language content
fails locally; a language call cannot inherit an English greeting.

Language voice selection is supplied before acoustic configuration is constructed.
Canonical task IDs select the same prebuilt acoustic conditions across languages.
When sampling is needed, stable canonical seeds and the established random draw
order preserve background assets and effect settings. The selected pack supplies
voice identity, interruption behavior, inserts and backchannels directly. Empty
insert lists remain empty. The existing scheduler and rate/10 in-turn tic coupling
are unchanged.

## Published schema

`LanguagePack` in `src/tau2/data_model/language.py` is a strict Pydantic model.
Schema version 3 is mode-neutral: the top level declares a reviewed greeting,
the agent language instruction, named personas and domain bindings. Everything
only a voice call needs lives in optional `voice:` sections:

- `pack.voice`: an optional spoken greeting override, the spelling section, the
  explicit speech-insert rate and the backchannel level.
- `persona.voice`: the provisioned voice ID, interrupt tendency, voice-only
  `pragmatics_clauses`, backchannel and non-directed phrases, vocal tics, the voice
  prompt examples and any spelled-entity tables.

A persona's top-level `pragmatics_clauses` are mode-neutral and render in text and
voice runs. `persona.voice.pragmatics_clauses` holds reviewed behavior that only
applies to spoken calls: phone openings and closings (`'여보세요'`, `'le llamo'`),
listening idioms, overlapping or interrupting the agent and continuers while the
agent talks. Voice runs render these after the mode-neutral clauses; text runs never
render them. A clause that mixes both is split, with the reviewed native utterances
kept verbatim.

A pack without voice sections is a valid text-only pack; voice runs refuse it
before any provider work. Voice sections are all-or-nothing across the pack and
its personas. Unknown fields and unsupported schema versions fail validation.
Short-phrase inventories are bounded (backchannel phrases: at most 2 entries of 2
words; non-directed phrases: 4 × 4), measured per language by
`src/tau2/utils/brevity.py`.

Each domain binding declares a revision and a task source:

- `source: file` (default): a pack-relative `task_metadata_file` (exact task IDs
  mapped to `canonical_task_id`, optional `caller_gender` and optional
  `persona_id`) and exactly one of `regional_bundle_file` (a published regional
  scenario bundle: localized database, policies and tasks; localized languages use
  this) or `tasks_file` (a JSON list of ordinary `Task` models run against the
  canonical database). The population is closed: only those exact tasks run.
- `source: registry`: the ordinary registry task set and splits. An optional
  `task_metadata_file` supplies reviewed metadata (for example caller gender) for
  unchanged registry tasks; adding a `tasks_file` also supplies reviewed same-ID
  replacement tasks, which always need that metadata. Other tasks receive a
  canonical default. The population is open, like any tau2 task set: an edited
  or caller-supplied task runs under its registry metadata and is pinned by its
  digest.

English is the shipped default pack (`language: en`) and binds its domains to the
registry, so English runs use the same persona, scaffold and provenance path as
every other language. English airline is metadata-only (registry tasks); retail
and telecom ship reviewed replacements that differ from the registry only by the
retail "Name details" caller block and the telecom caller-name swap, which drift
tests enforce.

The task scenario owns caller identity. The persona owns speaking behavior,
gender and a provisioned voice ID. Gender-matched assignment uses a stable
SHA-256 of run seed and canonical task ID; explicit assignments take precedence.

Every persona declares gender guidance, and a cross-lingual pack's personas
declare a `language_directive` (its origin facts); English personas have none.
The text and voice scaffolds live in one template source,
`src/tau2/user/guidelines.py` (`GUIDELINE_TEMPLATE_VERSION`). Text scaffolds never
presuppose a phone call. Domain guidance, persona clauses and final reminders
retain their reviewed placement.

### Voice ID overrides

`TAU2_VOICE_ID_<LANG>_<PERSONA>` replaces one pack persona's provisioned voice,
for example `TAU2_VOICE_ID_ES_MALE=...` for the `male` persona of `es`. The key is
upper-cased and hyphens become underscores (`TAU2_VOICE_ID_PT_BR_FEMALE`). The run
logs a warning and records the substituted voice in each task's language context;
results produced with an override are not comparable to released runs. English
runs use the same pattern (`TAU2_VOICE_ID_EN_MAMADOU_DIALLO`,
`TAU2_VOICE_ID_EN_PRIYA_PATIL`); see [voice-personas.md](voice-personas.md).

Pack-provided Portuguese spelling tables correct consistent PT-BR letter-name
mispronunciation in TTS-bound text. They apply only to explicit character-separated
spelling (such as `R, E, N, A`) and comma-separated native letter names (such as
`érre, é, êne, á`). Ordinary numbers, phone numbers and unsplit identifiers remain
unchanged: `120 segundos`, `VAAOXJ` and `mia_li_3668` are not automatically spelled.
The original message text stays unchanged; the delivered-text record captures the
normalized TTS input.

This intentionally narrows the previous normalizer, which also expanded unsplit
identifiers and some ordinary numbers. Opted-in calls record
`spelling_normalization_version: explicit-spelling-v1` in their language context.
Resume and benchmark validation reject an older context without that revision;
use a new run directory after this change.
Provider synthesis language and selected voice are explicit before construction.

## Run conditions and provenance

Selecting a language does not change the customer model, reasoning settings or
call-duration limit. Those are ordinary run configuration fields. Native providers
retain their configured agent voice; LiveKit resolves a multilingual speech profile
as described below.
For OpenAI Realtime, an omitted input-transcription language resolves from the
selected pack without mutating a caller-supplied audio configuration.

Provider settings are resolved before connection:

- OpenAI Realtime receives the selected transcription language and configured voice.
- xAI receives the pack's transcription language hint, including `es-ES` and
  `pt-BR`. English retains the provider's automatic detection behavior.
- Gemini receives the regional speech-language code for the supported
  `gemini-live-2.5-flash` half-cascade model. Native-audio models receive the
  language instructions in their system prompt; their API does not support that
  speech-language setting.
- LiveKit resolves the chosen preset into a complete, recorded cascaded config.
  Language runs use Deepgram Nova-3 recognition with an explicit language and
  ElevenLabs Flash v2.5 synthesis with a matching language hint and the Aria voice.
  The same speech models and voice are used for English and all five additional
  languages; the preset's LLM and turn-taking settings are retained.

LiveKit's effective STT, LLM and TTS settings are stored in
`audio_native_config.cascaded_config`. The caller's request is retained separately
in `audio_native_config.cascaded_request`: `CascadedConfigRequest(preset="openai-thinking")`
selects a preset, while `CascadedConfigRequest(config=...)` supplies a complete
component override. Explicit components must support the selected language;
incompatible settings fail before a call starts. Resolution copies the request for
each language, so generated defaults never become overrides for another language.
Saved runs retain both the request and its effective settings.
Custom runs with `--benchmark none` resolve English like any other language, so they use the same multilingual speech components.
Install the voice extra for the ElevenLabs LiveKit plugin; it uses the repository's
`ELEVENLABS_API_KEY` (with the plugin's `ELEVEN_API_KEY` accepted as a fallback).

The default `tau2 run --audio-native` entrypoint supplies benchmark conditions from
`config.py`: GPT-5.5 with low reasoning, a 2400-second call limit and Marin.
The caller uses the explicit `openai/gpt-5.5` route with temperature left unset.
The shared resolver supplies these defaults when omitted and preserves explicit
overrides. Select released cells with repeated `--language` and `--domain` flags.
Python callers pass an unlocalized `VoiceRunConfig` template to `BenchmarkRequest`.
LiveKit templates
use the selected cascaded preset's LLM and the runtime's multilingual speech
components. `--dry-run` resolves and prints all configurations without calls.
GPT Live retains its own backend and voice configuration. Agent gender is resolved
from the actual selected voice through the shared catalog, including LiveKit's
selected TTS component. Known gender is supplied to the agent prompt and recorded
with the selected voice on each simulation; unknown voices have unknown gender.

Pack/task/metadata digests are frozen before workers start. Each task's resolved
persona, prompt content/template revision, voice and artifact provenance are
recorded in `Info.language_tasks` and its simulation's `language_context`.
Language runs have no run-wide English `global_simulation_guidelines`; their
resolved prompt content is per task. Resume rejects changed content or conditions.
Automatic and saved-conversation reviews reconstruct the caller's guidelines
from that recorded context, including its mode, tool availability and reminder.
Both text scaffolds include the persona's gender, behavior and text-domain
guidance exactly once. Voice scaffolds preserve their reviewed content.

All ordinary voice runs share the agent instructions: authenticate according to
domain policy, respond naturally, and follow policy. Cascaded agents retain their
transcription and spoken-format guidance. τ-Entity keeps its explicit prompt
overrides and domain defaults. `Info.voice_instruction_prompt_version` records
the shared agent prompt revision; it is not a selectable compatibility mode.

## Offline prompt review

```bash
tau2 languages render-prompts --output /tmp/voice-review --config voice-config.json
```

The prompt review chooses one actual task per language/domain/persona, uses the
same agent and caller prompt functions as a run, and writes `index.html`, plain-text prompts,
`review.json` and its generated schema. Repeated `--language` and `--domain`
filters narrow the review. `--config` accepts an ordinary `VoiceRunConfig` JSON
template, preserving explicit provider, model, voice, seed and duration settings.
Without it the renderer uses ordinary run defaults. Add `--benchmark voice` to
review the same resolved benchmark conditions and frozen tasks as execution.
No orchestrator, caller executor or provider session is created; no LLM or TTS call is started.
Source hashes cover the template and construction code actually consumed.

Offline tests cover strict content validation, identical English/localized
construction, explicit run conditions, canonical acoustics, prompt rendering,
resume guards, provider configuration and separate telecom databases. Live voice
capability and synthesis remain a separate release smoke check.

See [Running the benchmark](running-the-benchmark.md) for matrix execution, coverage and submissions.

## Authoring and benchmark integration

The [language factory](language-authoring.md) publishes schema 3 packs with explicit
prompt content. Benchmark presets supply conditions for each cell and distinguish
language hints from shared model, voice, reasoning and acoustic settings, including
the recorded LiveKit component configuration.

Language contribution PRs include a README following the
[submission template](language-submission-template.md). Agent judges are optional;
the [judge guide](judge-validation.md) covers independent criteria authoring,
candidate execution, held-out validation and explicit production registration.

## Retail caller identities

Packs declare `name_order` as `given_first` or `family_first`. The task factory
adds explicit name roles and the configured spoken order from each retail
caller's structured record. For family-first packs: "Your first name is {first_name}, and your last name
is {last_name}. When asked for your full name, give your last name first and
your first name second: {full_name}." Korean and Mandarin use the same wording
as tau multi. Given-first packs explicitly instruct the caller to give the first
name first and the last name second. Names are supplied even when the original
task mentions only a user ID. The `regional-release-data-v16` benchmark frame retains
all task IDs and the same frozen 50-task selection per domain while
selecting the [regional database bundles](database-localization.md).
