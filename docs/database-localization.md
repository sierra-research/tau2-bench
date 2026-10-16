# Regional database localization

Regional maps adapt the complete airline, retail and telecom databases to a
country. They also cover the domain's frozen 50 tasks and their expected actions,
initial state and policy references. The canonical databases and task IDs remain
the source of truth. Maps do not add languages to the official benchmark.

Maps are built by the integrated [language factory](language-authoring.md):
`draft`, `stage-pack` and `add-domain` take `DomainBundle.localization` in
explicit `generate` or `reuse` mode, compile the regional database, tasks and
policies, and present the result at the one whole-draft content review. There is
no separate map or airport approval.

Each authored value records its original value, native-script form, corresponding
romanization, and English explanation. Romanization represents the same regional
value in Latin script; it is not an English translation. Frozen task IDs, protocol
enums, dates, counts and task reward axes are preserved. Customer IDs can be
rewritten explicitly with a recorded canonical-to-regional correspondence.
Monetary fields use an explicit fixed benchmark scale, not a live exchange rate.

## Released voice benchmark

`tau2 run --audio-native` uses English, Spanish, Portuguese, Hindi, Chinese and
Korean across airline, retail and telecom: 18 cells, each with the same frozen
50 tasks for its domain, or 900 calls at one trial. The five non-English languages
use published regional databases, tasks and policies in every domain. English
keeps its existing control tasks and prompts, with its canonical databases,
policies and runtime rules pinned too.

Use `--dry-run` to inspect the selection before running. `--domain`, `--language`
and `--num-tasks` select smaller runs; they do not change the official score's
denominator. The released overall score requires the complete cohort. Adding a
language pack does not change that cohort.

## Inventory

```sh
tau2 factory inventory-database --domain retail --output retail-source.json
```

The inventory snapshots the canonical database, the frozen-50 tasks, policies
and inventoried tool text for one checkout, identified by per-document sha256
digests. Every value has exactly one owner;
unclassified values become explicit preserve-only protocol entries.
`tau2 factory init-request` writes these inventories for a new language and
references them from the draft request by path and digest. Run the factory from
the release checkout, with `TAU2_DATA_DIR` pointing to that checkout's data
directory when using an isolated worktree.

## Generation (`mode: "generate"`)

`localization.request` is a `RegionalDraftRequest`: the inventory (`source`),
regional `settings` (language, country, currency, fixed `currency_scale`, IANA
time zone, name order and English `notes`), a new map `revision`, pinned
`generation` settings, optional reference files, `batch_size` (default 32) and
`max_concurrency` (default 3). Airline also needs `airport_generation`, telecom
uses `caller_roles`.

Generation uses the fixed `regional-database-localization-v2` prompt
(`factory_database_localization`; rejected batches are retried with bounded
validation feedback as `factory_database_localization_repair`). Names and
airports precede cities, then states, address details, and dependent prose.
State mappings use the source city context; task fragments include their owning
task and referenced products. Prose and policy remain English with mapped
entities. Validation catches wholesale translation of long English instructions
and changes to protected record IDs or scenario clock literals; these checks do
not establish semantic quality.

Unchanged successful batches reuse their checkpoints under
`<draft>.localization/<domain>/`. If final validation fails, the factory writes
`<draft>.localization/<domain>/mapping.failed.json` with the complete proposal
and error, and no draft.
Checkpoint identity includes the whole request, so an identical rerun replays
the same result without new calls, and any input change (for example regional
`notes`, or a new `revision`) regenerates every batch.
To fix a few entries without regenerating the rest,
use [selective repair](#selective-repair).
Emails and customer IDs are then derived deterministically from the regional
names (`regional-email-v1`, `regional-user-ids-v1`): emails use short numeric
suffixes and are collision-checked, and keep the mailbox domain of the map's
already-localized emails (the source domain until any exist; several domains
fail); customer IDs keep the source numeric suffix
and update database keys and linked task, initialization and evaluation
references. Explicit caller-name declarations grounded in the source customer's
full name and user ID are synchronized with the regional record; ambiguous
declarations require review.

Airline also needs reviewed airport city labels. The map owns IATA
correspondence; a map-bound catalog (`AirportCatalog`, bound by `map_sha256`)
provides native, romanized and English city labels. Telecom uses
`telecom-caller-roles.json`: the account being renamed and male/female name
sources. The build reuses their regional names while preserving task gender
assignments, birth date, phone and account conditions. Each renamed caller is
written into its task (the `set_user_info` action, caller prose and an
`initialization_data` customer override); rotated emails use the regional
database's single mailbox domain. The bundle's `caller_identities` record each
task's name source and final identity as provenance; the runtime does not read
them.

## Selective repair

`tau2 factory revise-database-localization` redrafts only the entries you select
and keeps every other mapping exactly. It reads a `DatabaseRevisionCommand`
JSON file; paths inside it are relative to that file:

```json
{
  "previous_file": "../draft.localization/retail/mapping.failed.json",
  "previous_kind": "failed",
  "revision": "de-retail-v1-fix1",
  "generation": {"model": "gpt-5.5", "reasoning_effort": "low"},
  "redraft_ids": ["address:<entry id>", "prose:<entry id>"],
  "feedback_files": ["feedback.md"]
}
```

```sh
tau2 factory revise-database-localization --request revise-request.json \
  --output retail-map-fix1.json --checkpoints revise-checkpoints/
```

- `previous_kind` is `map` (a compact map, the default), `failed` (a
  `*.failed.json` from a failed draft or revision) or `snapshot` (an expanded
  map).
  A failed or snapshot input must match the current inventory, unless you set
  `refresh_inventory: true` to rebind it explicitly.
- `revision` must be new. `redraft_ids` lists exact entry IDs from the previous
  artifact's `source.entries`, and `redraft_categories` selects whole
  categories. Include dependent prose that mentions a changed value: an
  unselected entry never changes. Automatic entries (emails, derived customer
  IDs) cannot be selected.
- `feedback_files` are review facts, for example "source 98178 is the
  deliberately wrong postal code: map it to a value different from 23552". They
  go to the fixed revision prompt
  (`factory_database_localization_revision`) as evidence, not as permission to
  change unselected records. `reference_files`, `batch_size` and
  `max_concurrency` work as in generation.

The output is a compact map that passed the same final domain validation as a
draft. If validation fails again, the verb writes the proposal and error beside
the output (`retail-map-fix1.failed.json` here), which can be the
`previous_file` of the next revision.

To use the repaired map in the factory, switch that domain's
`localization` to `mode: "reuse"` (below) with `mapping` pointing to the
output, and rerun `draft` to a new output path. The reference digest is the
canonical JSON digest of the file:

```sh
python -c "import json,sys; from tau2.multilingual.factory.request_files import json_digest; print(json_digest(json.load(open(sys.argv[1]))))" retail-map-fix1.json
```

Domains still in `generate` mode replay from their checkpoints without new
calls.

## Reuse (`mode: "reuse"`)

Supply `source` (the inventory the map was drafted against), `mapping` (the
compact map), `airport_catalog` for airline and `caller_roles` for telecom. The
build makes no model calls. `source` must reproduce the map's source manifest
exactly; see [Refresh shipped regional maps](language-authoring.md#refresh-shipped-regional-maps)
when code has moved the inventoried tool text.

## Review

The content review page includes every original/native/romanized mapping,
meaning, rationale and source location, and a table of deterministic flags with
before/after text: prose that still differs after masking entity values and
name-derived customer IDs, source values that remain as whole tokens in localized
text, and intentionally preserved values. Review names, addresses and postal
codes together, route relationships, quantities, currency references, deliberate
incorrect values, and caller identity/persona consistency. Structural checks
cannot establish linguistic quality or unchanged task difficulty. Optional
`semantic_review` settings add one model review per task as evidence, never as
approval.

## Customer name facts

For every retail map, compilation appends explicit first/given-name and
last/family-name roles to each caller's known information. The spoken order
follows `name_order`: Korean and Chinese use family-first; the other published
languages use given-first. The names come from the effective localized
customer record, including task initialization changes, using the same name-clarification
helper as ordinary language-pack tasks. Native and romanized compilations use
their corresponding name fields. This does not reorder the display name or
change authentication actions, evaluation arguments or rewards. Opening name
declarations that were correct in the source must match the localized customer,
even when the prose has no user ID. Deliberate wrong-first authentication details
remain intact. A supplied current postal code must match the localized profile;
a name/postal-code lookup that uniquely identifies the source caller must remain
unique. Goals that use the saved profile address must use that same localized
address. Missing, ambiguous or stale caller facts fail validation. The English HTML review includes
the compiled caller instructions for every affected task, alongside the authored
mapping rows.

## Submit a regional draft

Commit compact maps under
`data/tau2/multilingual/<language>/localization/<domain>.json`, the published
bundles under `regional/<domain>.json`, and a short README recording scope,
generation settings, validation, review status and unresolved regional choices.
Every compact map records `"status": "published"`; human review is recorded in that
README, not in the map.
Keep expanded inventories, checkpoints and generated HTML outside Git; the
factory recreates them from compact maps and the pinned source checkout. The
compact format retains authored before/after values and source hashes, and
reconstructs deterministic preservation and numeric scaling from versioned rules.

## Runtime bundles

Released bundles contain the final runtime data and `localization_map_sha256`;
airline bundles also record `airport_catalog_sha256`. Reviewer notes and approval
receipts are not part of runtime data. Original tool-code evidence remains
recorded; the executable regional rules carry a separate version and
implementation fingerprint.

Set each pack domain's `regional_bundle_file` to its published artifact. A bundle
contains matching DB, tasks, caller metadata, policies and regional settings.
Localized fees, airport catalogs, currency labels and clocks use instance-local
domain settings. Every constructed environment receives independent DB objects,
including predicted and gold evaluation replay.

Publication records the `source-wall-clock-v1` rule: airline and telecom policies
keep the canonical scenario date and wall-clock time, with the regional timezone
label. Their tools use that same clock. Converting only the policy timestamp would
change billing or booking decisions while database and task dates stayed fixed.
Pack staging and finalization preserve the complete regional artifact and its
source and review provenance.

Database localization defaults on for bindings that publish a regional bundle.
The explicit comparison uses the canonical DB and English control scenarios while
retaining the requested conversation language and persona assignment:

```sh
tau2 run --audio-native --benchmark none --language ko --domain retail --no-localize-db --reasoning-effort low
```

English remains the control. Ordinary language packs without a regional bundle
retain their published task behavior. The official voice manifest pins the regional
bindings; canonical comparisons require `--benchmark none` and cannot claim that
released score. Installed languages do not expand the frozen scoring cohort.

The agent receives one mode-derived tool-value instruction. Tool names, argument
keys, enums and identifier formats follow their schemas. Regional mode uses native
names and addresses; canonical mode uses canonical values. Language packs contain
conversation-language and register guidance, without a competing English-only or
accent-stripping instruction.

Results record `localize_db`, complete scenario and map digests, and executable
runtime-rule fingerprints. Resume and saved-run re-evaluation reject changed
bindings before scoring. Existing frozen results are not rewritten. Human content
review and bounded live smoke tests remain required before a released score; offline
compilation and regression tests do not establish language quality.
