# Korean regional localization

Country: South Korea (`KR`). Language: Korean (`ko`). Currency: Korean won
(`KRW`). Scenario time zone: `Asia/Seoul`.

These maps are the published localization source for Korean. The compiled
bundles in `../regional/` are built from these maps; the runtime loads those
bundles by default for Korean calls. `--no-localize-db` selects the canonical
database and tasks instead, keeping the Korean conversation.

## Review status

Reviewed by native-speaker reviewers; reviewer identities are not published.

## Representation

Each authored value includes native Korean, a romanization of the same value,
and an English explanation. Source document hashes pin the canonical inputs.
Preserved protocol/state values and monetary conversions are reconstructed by
versioned deterministic rules.

The fixed benchmark currency scale is **1,000 KRW per source monetary unit**.
It is a synthetic design choice, not a market exchange rate. All regional fees,
prices, budgets and refunds follow the same scale.

Initial drafting used `openai/gpt-5.5` with low reasoning and the
`regional-database-localization-v1` prompt; targeted airline/retail corrections
used medium reasoning with `regional-database-revision-v1`. These are the
versions recorded in the maps; the factory's current drafting prompt for new
maps is `regional-database-localization-v2`.

## Coverage and checks

- `telecom.json`: all base database records and the frozen 50 task references;
  583 reconstructed entries, including 131 authored entries.
- `retail.json`: all base database records and the frozen 50 task references;
  12,643 reconstructed entries, including 6,135 authored entries.
- `airline.json`: all base database records and the frozen 50 task references;
  32,647 reconstructed entries, including 8,542 authored entries.
- Factory validation passed for native and romanized forms, source coverage,
  domain schemas and reference constraints.
- All 1,500 retail customer and order addresses were checked for matching city,
  province and broad postal prefix. Different delivery cities keep separate
  province mappings; exact street-level deliverability is not asserted.
- An English HTML review covers every entry and source occurrence.
- Task and policy instructions remain English with localized entity values.
  Task and unrelated IDs, scenario dates and wall-clock values remain unchanged.
- Retail physical quantities and numeric size labels keep their original scales;
  the maps do not infer an unspecified shoe-sizing convention.

## Source and contact identifiers

An explicit offline source refresh (`regional-source-lines-v1`) updated only
code-evidence line numbers and verified unchanged source values.

Emails use localized romanized names with the original numeric suffixes and
punctuation; source addresses without numbers receive a short numeric suffix.
Collision checks preserve distinct account lookups, including deliberate incorrect
email values in tasks. Customer/user IDs follow the localized romanized name and
retain their original numeric suffix. All linked references are updated and the
maps record canonical-to-localized aliases. These offline rewrites preserve the
original generation provenance.
The deterministic caller-name check corrected 3 airline instruction entries
whose explicit caller name differed from the structured customer record. Only
that name span changed; the remaining instruction text was retained.

## Names and airline network

Caller gender assignments and persona/voice selection are unchanged.
Korean displayed names are family-first; given/family tool fields retain their
separate meanings. Names have role explanations, not invented etymologies.
All 50 retail callers' task instructions carry explicit given-name and
family-name facts from each effective localized customer record, with the
family-first spoken order.

The airline network uses 20 distinct airports: 15 in Korea and five nearby
international destinations. Routes and schedules are synthetic. The original
West Coast task selects exactly Narita, Haneda and Kansai; it does not describe
these airports as a Korean western coast. Wonju/Hoengseong is a wider-region
airport, not an airport in Seoul city.

Airline adaptation preserves distinct airports, alternative-airport choices,
connection constraints and task intent. Its international scope uses consistent
policy framing without adding new travel requirements.

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
