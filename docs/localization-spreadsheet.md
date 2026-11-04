# Entity review spreadsheets

For entity-only review, export literal CSVs with seven columns:
Type, Difficulty, Original entity, Suggested localized entity, Approved / Disapproved,
Proposed entity, and Reason.

Type distinguishes Entity, Entity reference, Decoy, Pronunciation token,
Mispronunciation, Spoken digit, Date/time, and Identifier. It describes the
source role so reviewers can filter the existing rows without losing lineage.

```sh
tau2 factory export-localization-entities-csv \
  --map data/tau2/multilingual/hi/localization/banking_knowledge.json \
  --map data/tau2/multilingual/hi/localization/intake.json \
  --output review/hindi-entities
```

This writes `localization-hi.csv` and a source-bound `.manifest.json` sidecar.
Difficulty comes from the frozen measured intake selection: Easy, Hard, or Easy + Hard.
Incidental caller/context values say Context only; banking rows say Not applicable.
Identical banking reference pairs are consolidated across task scopes. Every original
binding remains in the manifest's typed reference_occurrences, so reviewers approve
one pair without losing source/task links. Different native or romanized mappings
remain separate.

Intake date rows show the caller-facing local date format using the same deterministic
projection as runtime (Hindi DD-MM-YYYY; Spanish/Brazilian Portuguese DD/MM/YYYY;
Chinese/Korean year-month-day). The manifest records the projection version and
format. Map/record/golden date values stay ISO; tool schemas already tell the agent
to submit YYYY-MM-DD. Relative phrases retain their meaning and get a local exact-date
anchor. Date decoys and omit-year behavior use that same local order.

Intake time rows show the initial caller expression. The shared
`intake-local-time-display-v1` formatter localizes complete clock notation and
supplies conventional quarter/half expressions where applicable: Hindi सवा,
पौने, डेढ़, ढाई and साढ़े; Spanish y cuarto/y media/menos cuarto; Brazilian
Portuguese e quinze/e meia/quinze para; Mandarin 一刻/半/三刻; Korean 반 and
15분 전. English runtime uses quarter past/half past/quarter to too.
These expressions initially omit the day period. The private caller lookup
includes the complete local time so the caller can supply it when asked.
The manifest's `caller_time_payload` retains that full anchor without putting
machine instructions in CSV cells. Non-fractional times use local clock notation.
Decoys retain a complete day period so an AM/PM-flip error stays audible.
Tool answers and frozen difficulty assignments do not change. Difficulty means
the original task's tier, not a new rating of each localized expression.

Both domains share the CSV. Full scripts, tool-argument JSON, and behavioral
instructions are excluded. Used decoys, correct pronunciation tokens, deliberate
mispronunciations, and spoken digit
names are individual pairs with their parent entry and value path retained in
the sidecar. Reference respellings are metadata and are excluded, so legacy
capitalized stress guides cannot appear as entity rows. Suggested pronunciation
values come from the native form, never its romanized review metadata.
CSV quoting preserves commas, newlines, and Unicode. All decisions
start blank. Export refuses to overwrite changed files or reviewer edits.

This is an entity-only subset, not approval of a complete map. Task prose,
behavioral instructions, speech stimuli in context, and publication still need
their separate checks. The full-map workbook importer below does not accept
entity-only CSV manifests as complete-map approval.

## Full-map workbook

The factory exports source-checked compact localization maps into an Excel workbook
with one tab per language. Banking and intake maps can share a tab. Install the
optional file-format dependency with `uv sync --extra localization-review`.

```sh
tau2 factory export-localization-spreadsheet \
  --map data/tau2/multilingual/es/localization/banking_knowledge.json \
  --map data/tau2/multilingual/hi/localization/banking_knowledge.json \
  --output review/entities.xlsx
```

The map paths above refer to completed draft artifacts produced by the authoring
workflow; an incomplete entity pilot cannot be exported as an approved map.
All maps must resolve against their pinned canonical source inventory.

The first five columns are Original entity, Suggested localized entity,
Approved / Disapproved, Proposed entity, and Reason. Reviewers edit the last
three. A blank decision remains pending. Disapproval or a proposed replacement
requires a reason. A replacement may accompany a pending or disapproved row;
it cannot accompany approval of the existing suggestion. Supporting columns show
domain, category, reused/new status, romanization, rationale and prior review
evidence. Protected hidden IDs keep decisions attached to entities when rows move.
Worksheet protection prevents accidental edits; it is not a security boundary.

Keep `entities.manifest.json` beside the workbook. It pins every displayed
candidate, canonical source and exact compact map. The importer checks those
fields against the current maps, refuses formulas, and rejects changed, missing,
duplicate or stale rows. It does not trust protection or hidden columns as proof
that an original value remained unchanged.

```sh
tau2 factory ingest-localization-spreadsheet \
  --map data/tau2/multilingual/es/localization/banking_knowledge.json \
  --map data/tau2/multilingual/hi/localization/banking_knowledge.json \
  --workbook review/entities.xlsx --reviewed-by 'Reviewer name' \
  --output review/decisions.json --approval-dir review/approvals
```

The decisions artifact retains pending rows, negative decisions and proposed
replacements. Only a map whose entire authored content is approved receives a
`ReviewApproval` in `<approval-dir>/<language>/<domain>.approval.json`. These
approvals bind the exact compact map, as required by the existing publication
command. Exporting a workbook or reusing a value never implies approval of its
new context. Reviewer attribution identifies the person who made the decisions;
do not use the importer to represent an automated review as native-speaker review.

Proposed replacements are revision feedback. Feed the decisions artifact into
the existing source-checked revision workflow, review the revised native/romanized
forms and dependent task changes, and then approve the new map. The importer does
not patch a native value in isolation or mark that patch approved.

To add another domain or revise candidates, export to a new path with `--previous
review/entities.xlsx`, supplying all previous maps plus the new maps. Unchanged
entries retain their decisions; changed candidates or source manifests reset them.
Removing an existing map or row is rejected in this refresh mode. An intentional
new selection starts a new workbook without `--previous`. Identical export reruns
leave the workbook, including reviewer edits, untouched.

Verify the workflow by exporting maps, editing one approval and one disapproval
with a replacement/reason, and ingesting the workbook. The rejected map must have
feedback and no approval file. Add a second domain with `--previous` and verify
existing decisions remain. Changing a suggested value directly in the sheet must
fail ingest; changes belong in a new source-checked map revision.

Run the contract tests with the review and dev extras:

```sh
uv run --extra localization-review --extra dev pytest tests/test_multilingual/test_localization_spreadsheet.py
```
