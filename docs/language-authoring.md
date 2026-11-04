# Language authoring

`tau2 factory` builds a reviewed language pack for airline, retail and telecom.
The pack uses the same [language-pack schema](language-packs.md) that ordinary
runs load. The factory generates localized entities, databases, tasks and
language prompts together, and all of them go through one content review before
any voice work. [Regional database localization](database-localization.md)
describes the maps and runtime bundles this produces.

> Terms such as *pack*, *cell*, *binding revision* and *pin* are defined in the
> [glossary](glossary.md).

Each stage reads a typed JSON request and writes a new, immutable artifact.
Only the stages listed under [Credentials and cost](#credentials-and-cost) call
a model or a voice provider. Request models live beside their stage in
`tau2.multilingual.factory`, and `model_json_schema()` lists their fields.

The factory only builds voice-capable packs: every draft persona carries speech
content, and `finalize` requires provisioned voices and a listening approval.
To ship a text-only language, write the pack by hand. It uses the same schema
with the pack-level and per-persona `voice:` sections omitted (see
[language packs](language-packs.md)). Text runs accept it, and voice runs refuse
it before any provider work.

```sh
tau2 factory --help
tau2 factory <verb> --help
```

## Credentials and cost

| Stage | Provider | Key | Main cost driver |
| --- | --- | --- | --- |
| `autoform` | generation model | provider key, e.g. `OPENAI_API_KEY` | 1 call |
| `draft` (pack content) | generation model | provider key | 1 call, plus at most 1 repair call (`factory_draft_repair`) with the validation error if the reply fails validation |
| `draft` (`localization.mode="generate"`) | generation model | provider key | 1 call per 32 authored entries per domain (roughly 270 for airline, 190 for retail and 5 for telecom), plus repair calls for rejected batches, plus 1 airport-catalog call for airline |
| `draft` (`localization.semantic_review`) | generation model | provider key | Optional: 1 call per task, 50 per domain |
| `prepare-voices` (`config.mode="design"`) | ElevenLabs | `ELEVENLABS_API_KEY` | 2 voice designs + 2 saved voices |
| `prepare-voices` (`config.mode="existing"`) | ElevenLabs | `ELEVENLABS_API_KEY` | 2 TTS auditions |
| `validate-voices --check-provider` | ElevenLabs | `ELEVENLABS_API_KEY` | 2 lookups |
| smoke `tau2 run` | agent + user simulator providers | the run's keys | 2 live calls per domain |

Generation goes through the shared `generate()` API, so the key is the one your
model's provider needs (`OPENAI_API_KEY` for `gpt-*` models). Every request pins
its own `model` and `reasoning_effort` (`minimal`, `low`, `medium`, `high` or
`xhigh`). Authoring requests do not inherit the benchmark's model settings.
Everything else runs offline: `init-request`, `inventory-database`, `validate`,
`review-content`, `approve`, `stage-pack` and `add-domain` with reuse inputs,
`reuse-voices`, `audition`, `validate-voices`, `finalize` and
`refresh-regional-bundle`.

## Create a language

Stages print one JSON line: the output path and the `sha256` digest. The next
gate or request uses that digest.

```sh
# 1. Draft the author form, then review and edit author-form.json by hand.
tau2 factory autoform --request form-request.json --output author-form.json

# 2. Write a valid draft request: frozen tasks, caller genders and inventories.
tau2 factory init-request --language fr --domains airline retail telecom \
  --form author-form.json \
  --reviewed-by "Ada Reviewer" --model gpt-5.5 --reasoning-effort medium \
  --country FR --currency EUR --currency-scale 1 --time-zone Europe/Paris \
  --output fr-request

# 3. Build and review all content (prints the draft digest and review page).
tau2 factory draft --request fr-request/draft-request.json --output draft.json
tau2 factory approve --gate content --artifact draft.json \
  --digest <draft sha256> --reviewed-by "Ada Reviewer" --output content-approval.json

# 4. Provision voices, listen, and approve the listening digest.
tau2 factory prepare-voices --request voices-request.json --output voice-assets
tau2 factory audition --request voice-assets/assets.json --output auditions
tau2 factory approve --gate listening --artifact voice-assets/assets.json \
  --digest <listening_sha256> --reviewed-by "Ada Reviewer" --output listening-approval.json

# 5. Publish, install and check.
tau2 factory finalize --request finalize-request.json --output published/fr
cp -R published/fr data/tau2/multilingual/fr
tau2 languages validate fr
tau2 run --audio-native --benchmark none --language fr --domain retail --num-tasks 2
```

1. **Author form.** `autoform` turns an `AutoformRequest` (identity, hints,
   optional `persona_names`, pinned generation settings) into an
   `AuthorFormArtifact`. Review the locale, registers and the two persona
   sketches (one male, one female). Set `identity.name_order` to `given_first`
   or `family_first`; the pack keeps it.
2. **Draft request.** `init-request` writes `draft-request.json` and the files
   it references into a new directory. This request encodes these requirements:
   - **Frozen tasks.** Each domain uses the 50 tasks in
     `data/tau2/task_subsets/<domain>_50.json`, in that order. These are the
     reviewed English tasks shipped under `data/tau2/multilingual/en/`.
     Publication rejects any other cohort.
   - **`caller_gender`.** Every task's metadata carries the reviewed English
     caller gender. The factory routes each task to the persona of that gender.
     An explicit `persona_id` must agree with it.
   - **Task IDs.** Airline and retail IDs are `<canonical id>_<language>` (for
     example `6_fr`). Telecom keeps the canonical ID. `metadata` maps each ID to
     its `canonical_task_id`, so the runtime never parses IDs.
   - **`voice_guidelines`.** This per-domain field starts empty. Write it by
     hand if voice mode needs domain-specific spoken conventions. No stage
     generates it.
   - **Source inventory.** Each domain references the inventory of this
     checkout's canonical databases, frozen tasks and policies. It is identified
     by per-document sha256 digests, not a commit; publication rejects a request
     whose inventory differs from the checkout's.
   - **`localization`.** `init-request` emits `mode: "generate"` with your
     regional settings (country, currency, fixed decimal `currency_scale` from
     USD, IANA time zone). It adds `airport_generation` for airline and the
     shared `caller_roles` for telecom. Add English `settings.notes` to state
     regional constraints for the generator.
   Retail tasks gain explicit given- and family-name facts inside the factory
   (see [Customer name facts](#customer-name-facts)).
3. **Content review.** `draft` compiles the regional database, tasks, reference
   actions, identities, policies and airport labels. It then drafts the
   language and persona content with the fixed `public-pack-draft-v9` prompt.
   Compilation requires complete source coverage and preserved schemas,
   relationships, task cohorts and lookup outcomes. `draft.review/index.html`
   shows the language prompts, full tasks, database content and entity
   mappings. It also has a table of deterministic flags, each with its
   before/after text: prose that differs after masking entity values and
   name-derived customer IDs, and source values still present in localized text
   (matched as whole tokens). Flags can be false positives, and passing them
   does not establish geographic or linguistic correctness. A reviewer approves
   **the one whole-draft digest** printed by `draft` and shown on the page.
   `review-content` re-renders the page, and `validate` re-checks and reprints
   the digest.
4. **Voices.** `prepare-voices` takes a `VoiceAssetsRequest` whose `review` is
   the content approval. For new voices use `config.mode="design"` with
   explicit design settings and a `VoiceSpec` per persona: an acoustic brief
   plus native audition text of at least 100 characters. For account voices
   use `config.mode="existing"` with a persona-to-voice-ID map and
   `AuditionSpec` values. `audition` exports the samples and an `index.json`.
   Its `listening_sha256` covers exactly what was heard: voice settings,
   specs, voice IDs and sample hashes. The listening approval binds that
   digest.
5. **Publish.** `finalize` re-checks the draft, both approvals and every audio
   file. It then writes `pack.yaml`, the task and metadata files, the regional
   bundles and `provenance.json` in one atomic step. Drafts, mappings,
   approvals and audition evidence stay in the authoring workspace.

### Smoke gate

After installing the pack, run the ordinary benchmark command on two tasks per
domain before a PR:

```sh
tau2 run --audio-native --benchmark none --language <code> --domain <domain> --num-tasks 2
```

Add `--dry-run` first to confirm the selection offline. The factory has no smoke
verb of its own: `tau2 run` is already the verb, and a wrapper would duplicate
its provider, concurrency and resume options. Record the command, providers and
result directory in the pack README.

## Request files and references

A request can reference a prior artifact instead of inlining it (a draft can
exceed 60 MB):

```json
{"$artifact": "draft.json", "sha256": "<digest printed by draft>"}
```

Paths are relative to the request file. The digest is the same `content_digest`
the stage printed. If the file was edited or replaced, resolution fails before
the stage runs. References work anywhere in any stage request. Directory fields
(`assets_root`, `previous_assets_root`) are also relative to the request file.

<!-- draft-request-example -->
```json
{
  "form": {
    "identity": {
      "language": "fr",
      "display_name": "French",
      "script": "latn",
      "name_order": "given_first"
    },
    "locale_notes": "Metropolitan French; dates as 14 mars 2026; prices with a decimal comma.",
    "preferences": "Everyday spoken French with vous by default.",
    "personas": [
      {"name": "Luc", "gender": "male", "sketch": "Plain, direct wording."},
      {"name": "Anne", "gender": "female", "sketch": "Warm, careful wording."}
    ]
  },
  "revision": "fr-v1",
  "generation": {"model": "gpt-5.5", "reasoning_effort": "medium"},
  "speech_insert_events_per_minute": 1.0,
  "backchannel_level": "medium",
  "domains": [
    {
      "domain": "retail",
      "revision": "fr-retail-v1",
      "tasks": {"$artifact": "tasks/retail.json", "sha256": "<digest>"},
      "metadata": {"$artifact": "tasks/retail.metadata.json", "sha256": "<digest>"},
      "reviewed_by": "Ada Reviewer",
      "voice_guidelines": "",
      "localization": {
        "mode": "generate",
        "request": {
          "source": {"$artifact": "inventory/retail.json", "sha256": "<digest>"},
          "settings": {
            "language": "fr",
            "country_code": "FR",
            "currency_code": "EUR",
            "currency_scale": "1",
            "time_zone": "Europe/Paris",
            "name_order": "given_first",
            "notes": ["Fixed benchmark price scale; not a market exchange rate."]
          },
          "revision": "fr-retail-v1",
          "generation": {"model": "gpt-5.5", "reasoning_effort": "medium"}
        }
      }
    }
  ]
}
```

A `tests` check validates this example against `DraftRequest` after binding its
references to files regenerated by `init-request`.

`voices-request.json` and `finalize-request.json` reference earlier outputs:

```json
{
  "draft": {"$artifact": "draft.json", "sha256": "<draft sha256>"},
  "review": {"$artifact": "content-approval.json", "sha256": "<approval sha256>"},
  "config": {"mode": "design", "provider": "elevenlabs", "model": "eleven_ttv_v3",
             "seed": 42, "loudness": 0.5, "guidance_scale": 38},
  "specs": {
    "Luc": {"description": "<acoustic brief>", "audition_text": "<native text, 100+ characters>"},
    "Anne": {"description": "<acoustic brief>", "audition_text": "<native text, 100+ characters>"}
  }
}
```

```json
{
  "draft": {"$artifact": "draft.json", "sha256": "<draft sha256>"},
  "assets": {"$artifact": "voice-assets/assets.json", "sha256": "<assets sha256>"},
  "assets_root": "voice-assets",
  "review": {"$artifact": "listening-approval.json", "sha256": "<approval sha256>"}
}
```

## Approvals

`tau2 factory approve` writes a `ReviewApproval` (`content_sha256`,
`reviewed_by`). Nobody writes approval JSON by hand. The verb recomputes the
digest of the artifact on disk and refuses if it differs from the digest the
reviewer saw:

- `--gate content --artifact draft.json`: the whole-draft digest printed by
  `draft`, `validate` or `review-content`.
- `--gate listening --artifact voice-assets/assets.json`: the
  `listening_sha256` printed by `prepare-voices`, `reuse-voices` and
  `audition`.

An approval is an attribution record, not authentication. Keep approvals in the
review workspace, outside the release tree. Edits to entities, tasks, prompts or
review evidence change the draft digest and void the content approval.

## Voice asset files

Each generated sample is stored once under `audio/<sha256>.wav` or
`audio/<sha256>.mp3` in the asset directory. Version `voice-assets-v3` stores
the audio hash and format in its previews, recovery records and final manifest.
Those JSON files contain references, not encoded copies of the recordings.
Keep the directory together when moving or sharing it.

`audition` and `validate-voices` locate audio beside the supplied `assets.json`.
A `FinalizeRequest` supplies `assets_root`; a `ReuseAssetsRequest` supplies
`previous_assets_root`. These directory paths are relative to the request file
unless absolute. For example, a finalize request next to `voice-assets/` uses
`"assets_root": "voice-assets"`. Reuse copies the referenced recordings into its
output directory so that directory can be moved independently.

Audition export writes playable samples and a compact `index.json` containing
the listening digest, persona/voice mapping, sample filenames, formats and
hashes. Samples use hard links when the filesystem supports them; exports across
filesystems use verified copies. Validation, recovery, reuse and publication
verify the referenced audio before proceeding.

Older staged manifests with embedded recordings are rejected. They require
regeneration in a new output directory; existing files are never rewritten.

## Content rules

Every draft supplies a nonblank localized agent greeting and, for each of its two
personas, gender guidance, the cross-lingual `language_directive` origin facts,
and a complete `voice:` section with all named voice-example slots and phrase
inventories (schema version 3). The pack-level `voice:` section carries the
spelling section, speech-insert rate and the request's backchannel level.
Missing, unknown or stale fields fail validation. Persona `pragmatics_clauses`
render in text and voice runs, so they stay mode-neutral; behavior that only makes
sense on a spoken call (phone openings and closings, listening idioms, overlap,
interruption, continuers while the agent talks) goes in the persona's
`voice.pragmatics_clauses`, which only voice runs render.

The factory renders the four supported prompt variants (text/voice, with/without
user tools) through the runtime's normal renderer before publication.

The customer simulator produces text. Instructions may describe vocabulary,
register, native orthography and ordinary number/date/symbol words. Accent,
pronunciation, pitch, volume, pauses and delivery coaching belong in the separate
voice-asset brief. Spell character by character only after an explicit spelling
request; add localized word anchors only after a separate request for anchors.
Repeating a value or encountering a failed lookup does not authorize spelling.

Authentication remains in the domain policy and shared agent instruction. Language
content must not add lookup procedures, identity requirements, retry tactics or
spelling-on-failure guidance. Typed validation checks known retired instruction
patterns in author forms, nested prompt content, manual drafts, domain additions,
pack imports and final publication. It does not replace native-language review.

## Customer name facts

For every retail task, the compiler appends explicit given-name and family-name
facts to the customer scenario, with spoken order selected by `name_order`. Values come from the
structured caller record selected by `task_callers`, never from splitting a
display name. Korean and Chinese use `family_first`; other shipped languages use
`given_first`. Both conventions supply names explicitly, including tasks whose
source scenario gives only a user ID.

The compiler adds the fixed fact block before the task draft is reviewed. Rebuilds
are idempotent, and missing, stale or ambiguous identity facts fail publication
validation. Imported retail tasks must resolve one caller from structured task
references, successful identity lookups, an explicitly materialized record, or an
unambiguous source name declaration. Preserve canonical task IDs, tool arguments,
lookup behavior and all other scenario text.

Agent clauses remain authored in the pack and assembled by the existing
multilingual runtime with the actual agent voice context. Task localization does
not add agent instructions.

## Extend an existing pack

`stage-pack` imports a published pack and its exact task bundles into a new
draft revision without generation. `add-domain` adds another supported domain to
a staged draft. Both accept the same `localization` input as `draft`, compile
regional output and write the same review page. Use `mode: "reuse"` to build
from an existing compact map offline: supply `source` (the inventory the map was
drafted against), `mapping`, `airport_catalog` for airline and `caller_roles`
for telecom, preferably as `$artifact` references.

When persona content is unchanged, `reuse-voices` rebinds the reviewed voices to
the new draft. Like `prepare-voices`, its `--output` is a directory that receives
`assets.json` and the audio. Nothing audible changes, so the listening digest is
unchanged and the previous listening approval also covers the reused assets.
The new draft still needs its own content approval. A pack created outside the
factory can use `config.mode="existing"` with `prepare-voices` to obtain
audition evidence for its account voices.

Source packs stay unchanged. Changed tasks, prompts, settings or voice requests
need new output paths.

## Refresh shipped regional maps

Shipped maps pin the sha256 digest of every source document they were drafted
against. When code moves
the inventoried tool text without changing databases, tasks or policies, reusing
a map fails with "Localization source manifest differs". Rebind each map to the
current checkout and republish its bundle:

```sh
tau2 factory refresh-regional-bundle --language es --domain retail \
  --revision es-retail-refresh-1 --output refreshed/es
cp -R refreshed/es/localization refreshed/es/regional data/tau2/multilingual/es/
```

The verb makes no model calls. Only tool-text evidence may differ. The installed
map must carry the published bundle's revision, and for airline the installed
airport catalog must be the one the bundle pins (`airport_catalog_sha256`); the
refreshed catalog's `map_sha256` is rebound to the refreshed map. Review-only map
edits (entry notes) are allowed, because the verb refuses unless the republished
bundle's runtime content matches the installed one exactly, apart from the
provenance digests and revision. It records `regional-source-evidence-v1` refresh
provenance. The bundle's `localization_map_sha256` changes, and so does the
scenario digest in run fingerprints. Refreshing a released language therefore
needs a release-manifest update (`tau2 benchmarks pin voice` and
`tau2 benchmarks pin text`). Frozen results keep their original pins.

## Retries and recovery

Identical reruns reuse staged output without new generation calls. Changed
inputs or prompt versions refuse reuse. Generated localization checkpoints live
in `draft.localization/`. If entity generation fails validation, `draft` writes
`draft.localization/<domain>/mapping.failed.json` with the full proposal and
error, and no draft.
Checkpoints are keyed by the whole generation request: an identical rerun
replays without calls, and changing an input (for example regional `notes`)
regenerates all batches for that domain. To repair only the failing entries,
run [`revise-database-localization`](database-localization.md#selective-repair)
on the failed file and feed its output back with `mode: "reuse"`. If the
generated pack content fails validation, `draft` makes one repair call with the
validation error; a second failure stops with that error and writes nothing.
Finalization publishes atomically. An identical publication is a no-op, and a different bundle cannot overwrite an existing
destination.

Voice creation records its intent before contacting the provider. If the
outcome is uncertain, inspect the account and run `record-created-voice` with
the confirmed voice ID before resuming. The factory never guesses which remote
voice was created. `validate-voices` checks local audio and provenance, and
`--check-provider` also checks account access.

Optional [agent judge authoring](judge-validation.md) supports naturalness
extensions, adapted catalog diagnostics and new diagnostics. Reviewed candidates
run separately until held-out validation and explicit promotion.

## Verification

Offline tests cover typed generation replies, all prompt variants, pack imports,
domain extension, entity and lookup consistency, request references, approvals,
asset reuse and recovery, and publication through the runtime loader. They mock
generation and speech providers. A new language still needs a real generated
draft, native-speaker review, voice auditions and the smoke gate before
benchmark use.

## Submit a language PR

Every contributed language must include `README.md` beside `pack.yaml`, following
the [submission template](language-submission-template.md). The README records the
locale, domain/task coverage, personas and voices, exact review evidence, executed
checks, smoke results and known limitations. Keep transient staging artifacts out
of the PR; retain reproducible source requests and references to reviewed evidence.

A **user-only** submission needs no new diagnostic judge. A **user-and-judge**
submission lists each judge separately as **unvalidated** or **validated**, with
its version and applicability. Mixed statuses are allowed. Follow
[judge validation](judge-validation.md) for evidence and production promotion;
writing "validated" in a README never activates a judge. New language packs do
not automatically change the official benchmark's frozen language/task frame.

### Review actual prompts before the PR

Review the content page before finalizing. After installing the finalized bundle
under `data/tau2/multilingual/<language>/`, run:

```sh
tau2 languages validate <language>
tau2 languages render-prompts --language <language> --output /tmp/language-review
tau2 tasks verify
```

`languages validate` prints the pack path and, per domain, the task count and
the artifact digests that runs pin. Open the rendered `index.html` and inspect
the complete agent and customer prompts and the task/persona assignments.
Finalizing into a local directory does not release or merge a pack.

### Offline submissions

A contributor without provider keys can run `init-request`, validate requests,
reuse supplied artifacts and open a draft PR. The PR must clearly mark missing
generation, native review, voice review or smoke evidence. Do not invent
approvals or claim completed onboarding.

The repository skills [add-a-language](../.agents/skills/add-a-language/SKILL.md)
and [add-a-domain](../.agents/skills/add-a-domain/SKILL.md) follow these steps.
The latter localizes another existing supported domain. It does not implement
new domain policies or tools.
