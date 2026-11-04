# Mandarin database localization: Mainland China

These maps are the published localization source for Mandarin. They cover the
complete airline, retail and telecom databases, each domain's frozen 50 tasks,
and policy references. The compiled bundles in `../regional/` are built from
these maps; the runtime loads those bundles by default for Mandarin calls.
`--no-localize-db` selects the canonical database and tasks instead, keeping the
Mandarin conversation.

## Review status

Reviewed by native-speaker reviewers; reviewer identities are not published.

## Included maps

| Domain | Reconstructed entries | Authored mappings | Coverage |
| --- | ---: | ---: | --- |
| Telecom | 583 | 131 | Database, user-device database, policies and 50 tasks |
| Airline | 32,647 | 8,542 | Database, policy and 50 tasks |
| Retail | 12,643 | 6,135 | Database, policy and 50 tasks |
| **Total** | **45,873** | **14,808** | All three domains |

Both representations pass offline structural and authentication validation. The
English HTML review includes every reconstructed entry across all three maps.

## Region and generation

- Country/language: Mainland China (`zh`). Scenario timezone: `Asia/Shanghai`.
- Currency: `CNY`, using a fixed synthetic scale of 7 per source
  monetary unit. This is a benchmark choice, not an exchange-rate quotation.
- Native values use Chinese script; romanized values represent the same entities
  in Latin script.
- Task instructions, evaluation assertions and policies remain English, with
  native or romanized entity substitutions. Customer IDs follow the localized
  romanized name and original numeric suffix; every linked reference changes
  consistently. Task IDs, other machine IDs, dates, scenario logic, reward axes
  and the frozen task sample remain unchanged.

Initial drafting used `openai/gpt-5.5`, `reasoning_effort: low`, the fixed
`regional-database-localization-v1` prompt, batch size 32 and concurrency 3.
Targeted corrections used the same model with `reasoning_effort: medium` and the
fixed `regional-database-revision-v1` prompt. These are the versions recorded in
the maps; the factory's current drafting prompt for new maps is
`regional-database-localization-v2`. Maps record the input digest, source
document digests, model settings and parent-map digest. An explicit offline
source refresh (`regional-source-lines-v1`) updated only shifted code-evidence
line numbers, verifying unchanged authored data and code-evidence text. Original
model-generation provenance remains recorded.

Subsequent deterministic revisions use `regional-email-v1`,
`regional-caller-names-v1` and `regional-user-ids-v1`; none makes model calls.
Emails preserve source format and numeric suffixes, using a stable four-digit
suffix where none existed. Canonical-to-localized customer ID aliases are
recorded explicitly and rejected if they collide; both representations use the
same ASCII IDs. Linked database keys, task initialization, expected tool
arguments and prose use those aliases. Explicit caller-name statements are
synchronized to that same structured customer record, preserving all other
instruction text. These steps preserve canonical source snapshots and record
their parent digests.

## Content checks

- Airline airport catchments, nearby-airport choices and destination sets align
  with task prose, including linked user IDs, policy fee scaling and scenario
  timezone labels.
- Retail addresses use the refined inventory, which separates shared state
  fields where one caller has addresses in different cities. All 1,500 compiled
  user/order addresses pass city/province and broad postal-region checks.
- Task 68 retains its intentional wrong-ZIP authentication failure.
- Product options retain original physical quantities and unit scales, with unit
  words translated. Unsupported shoe-size conversions and rounded conversions
  were removed. Exact equivalent numeric expressions remain permitted.
- Explicit caller-name statements are checked against the linked structured
  customer name, in native and romanized forms, before customer IDs are derived.
- Customer IDs cover 500 airline users, 500 retail users and 4 telecom customers.
  All dependent references compile in both representations, with unchanged
  authentication success/failure and frozen task IDs.
- All 50 retail callers' task instructions carry explicit given-name and
  family-name facts from their effective localized customer records, with the
  family-first spoken order. Display names remain family-first.

These checks establish structural and broad geographic consistency. They do not
establish deliverability of synthetic street addresses or equal task difficulty
across regions.

## Validate and review

```sh
tau2 factory validate-database-localization --map data/tau2/multilingual/zh/localization/telecom.json
tau2 factory validate-database-localization --map data/tau2/multilingual/zh/localization/airline.json
tau2 factory validate-database-localization --map data/tau2/multilingual/zh/localization/retail.json
tau2 factory review-database-localization \
  --map data/tau2/multilingual/zh/localization/telecom.json \
  --map data/tau2/multilingual/zh/localization/airline.json \
  --map data/tau2/multilingual/zh/localization/retail.json \
  --output /tmp/zh-localization-review
```

Inventories, expanded snapshots, checkpoints, rejected drafts and generated HTML
are local authoring artifacts and are not published.

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
