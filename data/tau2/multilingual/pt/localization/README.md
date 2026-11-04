# Brazilian Portuguese regional database localization

These maps are the published localization source for Brazilian Portuguese. They
adapt the complete airline, retail and telecom databases to Brazil, with matching
policies and each domain's frozen 50 tasks. The compiled bundles in
`../regional/` are built from these maps; the runtime loads those bundles by
default for Portuguese calls. `--no-localize-db` selects the canonical database
and tasks instead, keeping the Portuguese conversation.

The maps preserve task membership, physical quantities, reward axes and unrelated
record IDs. User IDs follow each localized name, retain the original numeric
suffix and are remapped consistently across database keys, task text and linked
references. The map records every canonical-to-localized ID pair. Explicit
caller-name statements are synchronized with the same structured customer
record. Task and policy instructions remain English with localized values
inserted.

Each authored mapping records the source value, native form, romanized form,
English explanation and review notes. Portuguese uses Latin script, so native and
romanized forms are identical and retain Portuguese diacritics.

| Domain | Expanded source entries | Authored mappings |
|---|---:|---:|
| Airline | 32,647 | 8,542 |
| Retail | 12,643 | 6,135 |
| Telecom | 583 | 131 |

## Review status

Reviewed by native-speaker reviewers; reviewer identities are not published.

## Generation and verification

- Region: Brazil (`BR`), currency `BRL`, time zone `America/Sao_Paulo`.
- Fixed synthetic price scale: 5 BRL per source monetary unit. This is a benchmark
  design choice, not an exchange-rate quote.
- Authoring: `openai/gpt-5.5`, initially low reasoning with the
  `regional-database-localization-v1` prompt; targeted airline and retail
  revisions use medium reasoning with `regional-database-revision-v1`. These are
  the versions recorded in the maps; the factory's current drafting prompt for
  new maps is `regional-database-localization-v2`. Maps record exact prompt
  versions, request digests and parent provenance.
- An explicit offline source refresh (`regional-source-lines-v1`) updated
  tool-evidence line numbers without changing the evidence text, databases,
  tasks or policies.
- Email addresses use short numeric suffixes with collision checks. Original
  four-digit numbers and deliberately incorrect email variants remain distinct.
  These deterministic email, caller-name and user-ID revisions make no LLM calls
  and record their own provenance separately from generation.
- Both representations pass complete compilation, source coverage, record/model,
  reference-graph, task-invariant and authentication-outcome checks.
- Additional checks cover native/romanized equality, English instruction prose,
  consistent localized user IDs in prose, coherent city/state pairs, wrong-first
  postcodes, airport-choice constraints and unchanged numeric product quantities.
  All 1,500 customer and order addresses were checked for city/state consistency;
  state entries are scoped to their address rather than shared across different
  cities. Unspecified shoe sizes retain their catalog labels; physical-unit words
  are translated while keeping the original quantities and unit scales.

The airline network uses GRU/CGH/VCP for the source metropolitan airport group
and SJK for the nearby alternative. The open-ended destination task names exactly
Fortaleza (FOR), Recife (REC) and Salvador (SSA), preserving its original candidate
set. Airports are real, but flight schedules remain synthetic. SJK's return to
passenger service in March 2024 is documented by the
[municipal government](https://www.sjc.sp.gov.br/noticias/2024/marco/22/ministro-faz-visita-tecnica-ao-aeroporto-de-sao-jose/).

Structural checks do not establish unchanged task difficulty across regions.
Synthetic street/postcode pairs are not claims about deliverable real-world
addresses.

## Validate and review

```sh
tau2 factory review-database-localization \
  --map data/tau2/multilingual/pt/localization/airline.json \
  --map data/tau2/multilingual/pt/localization/retail.json \
  --map data/tau2/multilingual/pt/localization/telecom.json \
  --output /tmp/pt-localization-review
```

The command validates all maps before rendering. Expanded inventories, checkpoints
and generated HTML are local authoring artifacts; these compact maps contain the
reproducible source descriptors and authored values.

## Database modes

With the regional database (the default), the agent's tool arguments use
localized free-text values; with `--no-localize-db`, canonical values. Tool
names, JSON keys, enums and ID formats follow each tool's schema in both modes.
Telecom tasks keep their task-specific caller gender and persona assignments.

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
