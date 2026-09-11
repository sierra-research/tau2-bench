# Adding a domain to the multilingual voice pipeline

How to take a tau2 domain that already runs in text (registered in
`src/tau2/registry.py`, tasks + policy + tools in place) and make it a
multilingual voice benchmark domain. Telecom (added after airline) is the
worked example throughout; `git log --follow src/tau2/multilingual/domain_profiles.py`
finds the PR series.

Most of the stack is already domain-agnostic and needs **no changes**: the
voice runner, shared acoustics, persona voices, nativeness/delivery judges,
task-set discovery (`task_sets.py` auto-registers
`data/tau2/multilingual/<lang>/<domain>_tasks_<suffix>.json`), annotation
packets, and the localization invariants. What a new domain needs is one
profile entry plus the per-domain data that profile points at.

## 0. Risk gate — smoke the domain first

Before writing any multilingual code:

1. `tau2 run --domain <d> --num-tasks 3` (text) — rewards must compute.
2. One or two tasks through the audio-native voice runner in English.

Domains with **dual-control user tools** (telecom's simulated phone) exercise
paths airline never did — the voice user sim must drive device tools while
speaking. Verify this works before investing in localization.

## 1. Domain profile — the single registration point

Add an entry to `DOMAIN_PROFILES` in
`src/tau2/multilingual/domain_profiles.py`. The catalog is closed: everything
per-domain the multilingual layer knows lives here, and unknown domains fail
loud everywhere. Fields:

- `source_tasks_filename` — the English source set localized arms derive from.
  Curated domains (airline) point at `tasks.json`; generated-pool domains
  (telecom, 2285 machine-generated tasks) point at a reviewed seed file
  (`tasks_multilingual.json`, produced by `seed-tasks` below). This also
  decides which verb owns the domain's arm files, via the derived
  `curated_source` property: `tasks.json` → `tau2 factory arm-tasks`,
  anything else → `tau2 factory seed-tasks`. The two verbs refuse each
  other's domains, so no arm file has two owners.
- `caller_identity` — how the caller is anchored in a task.
  `USER_ID_HANDLE`: a prose handle like `mia_li_3668` (airline).
  `STRUCTURED_NAME_PHONE`: name + phone as structured `set_user_info`
  initialization-action args (telecom). A new anchoring style means a new
  enum member plus a dispatch branch in `invariants.identity_rename`,
  `localize_lib.derive_identity_tasks`, and
  `factory/entity_localization.build_identity_map`.
- `identity_swap_supported` — gates `tau2 factory localize-entities`.
- `tasks_translated` — `False` is the standard regime and what **both**
  shipped domains declare: the default English-prompt arm doesn't need
  translated prose, so the per-language sets carry the English source and the
  language lives entirely in the pack. While `False`, the
  script-presence/localized-digit invariants are skipped for the domain's
  sets and `arm-tasks`/`seed-tasks --lang` own the arm files.
- `translation_example_entities` — the domain's illustrative verbatim values
  (e.g. telecom's phone numbers / customer ids / GB amounts vs airline's
  reservation codes). Read only by the archived translator prompt now.
- `term_catalog` — see next section.

## 2. Term catalog + glossaries

Author ~20 `DomainTermDef` ids in `DOMAIN_TERM_CATALOGS`
(`tau2/multilingual/localization_catalog.py` — the profile's `term_catalog`
property is a read-only view of it) from the domain's policy/tool vocabulary
(telecom: `plan`, `data_allowance`, `esim`, `roaming`, `line_suspension`,
`speed_test`, …). Pack `domain_glossaries`
entries validate against this catalog, and guardrails require full catalog
coverage **for every domain a pack has an experiment for** — so adding the
catalog doesn't break existing packs until they add the domain's experiment.

Per-language glossaries are then drafted with the existing versioned prompt:

```
tau2 factory draft-localization --domain <d> --lang <l>
```

## 3. Per-language arm files

Every multilingual run of domain `<d>` in language `<l>` scores on task set
`<d>_<l>`, backed by `data/tau2/multilingual/<l>/<d>_tasks_<l>.json`. That
file is the domain's English SOURCE task set with `_<lang>`-suffixed ids and
**byte-identical English prose** — english-prompt mode is how every
multilingual run works, so the language lives in the pack, not the tasks.
Both producers go through the one emitter in `factory/task_arms.py`; which
one owns a domain is the profile's `curated_source`.

### 3a. Curated-source domains (airline)

```
tau2 factory arm-tasks --domain <d> --lang en --lang es ...
```

Nothing is materialized — the source IS the reviewed, checked-in
`tasks.json` the registry loads. The verb emits the requested arm files plus
the English caller-gender sidecar. Deterministic and idempotent.

### 3b. Generated-pool domains (telecom)

```
tau2 factory seed-tasks --domain <d> --split base
```

Materializes the reviewed split as `data/tau2/domains/<d>/tasks_multilingual.json`
(pool order — matching the domain's registry loader) plus a provenance
manifest (git sha, split, id list), and emits per-language arm files
(`--lang`, default `en`) with `_<lang>`-suffixed ids and byte-identical
English prose. Idempotent; the manifest keeps its sha when content is
unchanged. **Pause here for human review of the selected tasks.** Commit the
seed file, manifest, and arm files.

### Caller diversification + auth split (inside seed-tasks)

Generated pools hold the caller constant (telecom: every task is John Smith /
555-123-2002), so seed-tasks **rewrites each selected task onto one of the
domain's reviewed `caller_pool.yaml` identities** (25 for telecom; gender-mixed,
list order load-bearing — task *i* gets caller `i % 25`) with a deterministic
50/50 authentication split: even positions keep phone auth, odd positions
become **name+DOB auth** (`known_info` gives the name and a natural-English
date of birth — the agent must collect the name and convert the spoken date
to ISO for `get_customer_by_name`). Name-auth tasks also append a fixed
**line-identification clause** to `task_instructions`: the caller names the
line they are calling about when asked, but declines to be looked up by phone
number. That clause is load-bearing, not flavour — the canonical telecom
caller owns three lines, and with the number withheld outright nothing
identifies which one the call concerns, so the arm measured 39.8% wrong-line
actions against the phone arm's 9.1% before it existed. Keeping the number in
`task_instructions` rather than `known_info` is what stops it leaking back
into the lookup path (and keeps `task_auth_mode`'s known_info-digit
derivation exact). The pool file and `db.toml` are never edited:
each task ships a wholesale `initialization_data.agent_data.customers` patch
whose caller record carries the pool identity's `full_name`/`email`/
`date_of_birth` (bystanders byte-identical). Provenance lands in the manifest
(`caller_pool_sha256`, `caller_assignment`, `name_auth_task_ids`); there is
deliberately no stored auth-mode field — derive it with
`caller_diversity.task_auth_mode`. The English arm also gets a caller-gender
voice-routing sidecar (`multilingual/en/<d>_caller_gender_en.json`) that pins
the stock English persona choice to the caller's gender.

A seedable domain must therefore ship a reviewed
`data/tau2/domains/<d>/caller_pool.yaml` (validated by
`caller_diversity.CallerPool`: unique names/emails, valid adult DOBs, gender
consistent with the name catalog) **before** seed-tasks runs.

## 4. Identity swap

1. Extend `GIVEN_NAME_GENDERS` in `factory/name_genders.py` with every
   distinct caller given name in the seed set — for a seed-diversified domain
   that means every `caller_pool.yaml` given name (the pool validator and a
   coverage-guard test both fail on any missing name).
2. Make sure the locale has a reviewed `locale_corpus.yaml` (name pools by
   gender, email conventions, and a domestic `phone_number_format`) —
   see `data/tau2/multilingual/es/`.
3. ```
   tau2 factory localize-entities --lang <l> --domain <d>
   ```

What gets swapped is profile-driven. For `STRUCTURED_NAME_PHONE` domains the
caller's **name, email, and phone number** become locale material. The phone is
deterministically generated in the locale's national format; a compact alias
maps it to the canonical service line at run time. Customer
and line ids and amounts stay canonical, and the date of birth stays pinned to
the **English seed value** (the pool caller's DOB; dates never localize). The
identity map is keyed by the English caller **full name** (25
entries for telecom, one per pool caller; the shared customer id identifies
nobody), each rename **composes on top of the seed's diversification patch**
(the source task's effective customers list with just the caller's
name/email/phone swapped, merged into the existing `initialization_data`), and is
re-applied to the English prose, `set_user_info` args, ticket, and
`evaluation_criteria`. Outputs:
`<d>_tasks_<l>_identity.json` (auto-registered as `<d>_<l>_identity`),
`<d>_identity_map_<l>.json` (reviewable), `<d>_caller_gender_<l>.json`
(voice-routing sidecar).

Because the default run arm is English-prompt mode
(`english_prompts.english_user_task_variant` re-applies the rename onto the
English source prose), the identity arm needs **no translated tasks**.

## 5. Experiments + runs

```
tau2 factory add-experiment --lang <l> --domain <d> --smoke-task-stem <stem>
tau2 factory validate --lang <l>
tau2 run-preset --list          # the new preset appears
tau2 run-preset multilingual_v1_<arm>_<d> --stage smoke
```

Non-default domains get their own arm/preset, suffixed `_<d>` onto the
language's main arm name (`spanish` → `spanish_telecom` →
`multilingual_v1_spanish_telecom`); the default domain keeps the bare name so
shipped airline presets never rename.

`add-experiment` appends the domain's experiment block to the pack
idempotently and runs guardrails. Identity-variant sets replace the plain set
as the run arm by default (the `experiments` entry's
`include_identity_arm: false` opts out).

## 6. Translation (retired and deleted)

Task translation is gone. Both shipped domains declare
`tasks_translated: False` and ship English-prose arm files, and the whole
translate → verify → fix loop (plus the parity probe that verified it and the
translator CSV round-trip) was deleted on 2026-08-03.

`tasks_translated` survives as a profile flag because it still gates real
behaviour: while `False`, the script-presence/localized-digit invariants are
skipped for the domain's sets and the English-prose emitters own the arm
files. Flipping it to `True` blocks those emitters for non-`en` arms and
starts enforcing the script invariants — but nothing can produce a translated
set any more, so a domain would need its localized files supplied by hand or
the loop recovered from git history.

## Environment gotchas found adding telecom

- **Dual-DB domains**: `Environment.set_state` only re-aliases
  `user_tools.db`/`tools.db` after an `initialization_data` patch when they
  were the *same object* beforehand. Telecom's separate `db.toml` +
  `user_db.toml` must not be clobbered into one.
- **DB format**: `localize_lib.load_domain_db` probes `db.json` then
  `db.toml`. New formats extend that loader, not call sites.
- **List-valued DB collections** (telecom `customers`): addict-style dict
  patches replace lists wholesale — an identity patch must carry the full
  list, not just the renamed record.

## Tests you get for free / must add

Free (parametrized by discovery): the full localization-invariant suite in
`tests/test_multilingual/test_task_localization_invariants.py` runs over every
committed `<domain>_tasks_<suffix>.json`, and
`TestShippedIdentityArtifacts` in `test_entity_localization.py` covers shipped
identity sets.

Add: a given-name coverage guard over the whole task pool (see
`test_catalog_covers_every_telecom_caller`), derive-round-trip tests for a new
identity kind, and a seed-vs-registry sync test if the domain uses a seed file
(see `TestShippedSeedArtifacts`).
