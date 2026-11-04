# Hindi / India regional localization

These maps are the published localization source for Hindi. The compiled
bundles in `../regional/` are built from these maps; the runtime loads those
bundles by default for Hindi calls. `--no-localize-db` selects the canonical
database and tasks instead, keeping the Hindi conversation.

## Review status

Reviewed by native-speaker reviewers; reviewer identities are not published.

## Data scope

Each domain includes its complete canonical database, its frozen 50 tasks,
policies, and an inventory of code-owned text handled by the regional runtime
rules.

| Map | Complete database coverage |
| --- | --- |
| `airline.json` | 500 users, 2,000 reservations, 300 flights and 20 distinct airport nodes |
| `retail.json` | 500 users, 1,000 orders and 50 catalog products with their variants |
| `telecom.json` | 4 customers, 9 lines, 9 devices, 5 plans, 6 bills and the user database |

Every authored value includes a native form (Devanagari where applicable), a
matching romanized form, and an English explanation. Romanization is stored
review data, not another runtime mode. Instruction, evaluation and policy
sentences stay English with regional entities.

## Settings and generation

- Region: India; language: Hindi; given-name-first ordering.
- Currency: INR at a fixed synthetic benchmark scale of 80, not a live exchange rate.
- Time zone: `Asia/Kolkata`; synthetic scenario dates and wall-clock values remain fixed.
- Authoring: `openai/gpt-5.5`; initial drafts used low reasoning with the
  `regional-database-localization-v1` prompt, and targeted airline/retail
  corrections used medium reasoning with `regional-database-revision-v1`. These
  are the versions recorded in the maps; the factory's current drafting prompt
  for new maps is `regional-database-localization-v2`. Exact settings, source
  descriptors, request digests and parent-map digests are recorded in the maps.
- An explicit offline source refresh (`regional-source-lines-v1`) updated only
  code-evidence line numbers and verified unchanged source values.

Airline and retail address corrections resolve cities before states and postal
values. Airline airport and dependent task prose corrections preserve graph nodes,
accepted/rejected alternatives and the source destination choice sets. Retail
checks preserve deliberate wrong-first/correct-later authentication outcomes.

## Contact identifiers

Emails use localized romanized names with the original numeric suffixes and
punctuation; source addresses without numbers receive a short numeric suffix.
Collision checks preserve distinct account lookups, including deliberate incorrect
email values in tasks. Customer/user IDs follow the localized romanized name and
retain their original numeric suffix. All linked references are updated and the
maps record canonical-to-localized aliases. Task and unrelated identifiers remain
unchanged. These offline rewrites preserve the original generation provenance.
The deterministic caller-name check corrected 8 airline instruction entries
whose explicit caller name differed from the structured customer record. Only
that name span changed; the remaining instruction text was retained.

## Verification

Both native and romanized forms pass source coverage, typed database/task,
reference, monetary scaling and authentication validation.

All 1,500 retail customer and order addresses were checked after separating
state mappings for different delivery cities. The 17 corrected state entries
leave all other values unchanged. Romanized `Keral` and `Kerala` both represent
`केरल`.

The source LGA node uses Belagavi (IXG) within a broader Goa–Konkan–Belagavi
catchment, preserving the source airport-choice structure.

Telecom tasks keep their per-task caller-gender and identity assignments.

Regenerate the English review with:

```sh
tau2 factory validate-database-localization --map data/tau2/multilingual/hi/localization/airline.json
tau2 factory validate-database-localization --map data/tau2/multilingual/hi/localization/retail.json
tau2 factory validate-database-localization --map data/tau2/multilingual/hi/localization/telecom.json
tau2 factory review-database-localization \
  --map data/tau2/multilingual/hi/localization/airline.json \
  --map data/tau2/multilingual/hi/localization/retail.json \
  --map data/tau2/multilingual/hi/localization/telecom.json \
  --output /tmp/hi-localization-review
```

## Banking name-gender evidence

The current `banking_knowledge.json` map embeds source-name evidence bound to
each exact structured name. Source-owner aliases, linked emails and task
references preserve their linked identities. Source gender assessments are not
native-speaker approval or evidence of equal task difficulty.

Banking and intake are separate optional extensions under
`multilingual_extensions/`, outside the released three-domain Voice V2 frame.
See [retained review evidence](../../../../../docs/language-review-evidence.md)
for the recorded scope and digests of final extension approvals. Historical
authoring checkpoints and stale review CSV exports are not publication inputs.
