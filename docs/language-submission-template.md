# Language: <display name / language code / locale>

Copy this document to `README.md` beside the submitted `pack.yaml`. Replace the
placeholders with evidence from your submission. Follow
[language authoring](language-authoring.md) and
[judge validation](judge-validation.md); adjust these links relative to the pack.

## Scope

- Submission: **user-only** or **user-and-judge**.
- Pack revision:
- Regional variety, script and declared name order:
- Supported domains:
- Canonical task subsets and source inventory digests:
- Known limitations:

Adding a language does not change the official benchmark's frozen language or
task selection. Propose a benchmark revision separately if one is needed.

## User simulator

| Persona | Caller gender | Locale/register | Voice provider/model/ID |
| --- | --- | --- | --- |
| | | | |

Describe localization decisions that affect names, addresses, dates, numbers,
code-switching or forms of address. Explain how task identities and lookup
behavior were preserved. User instructions describe text; acoustic instructions
belong in the voice brief. Spell only on request, and use local anchors only when
anchors are requested. Authentication procedures belong to the domain policy.

## Content and voice review

- Native-language reviewer and relevant language/locale expertise:
- Exact content revision/digest reviewed:
- Prompt review: agent and user prompts, including tool/no-tool variants:
- Voice audition reviewer and reviewed artifact digest:
- Audition recordings or reproducible references:
- Issues found, corrections and remaining limitations:

Review attribution records who assessed the artifact; it is not proof of judge
accuracy. Record incomplete review honestly.

## User validation

| Check | Command / input revision | Result | Evidence |
| --- | --- | --- | --- |
| Pack and domain initialization | | | |
| Frozen task IDs and lookup behavior | | | |
| Voice assets and persona/gender mapping | | | |
| Live smoke (`tau2 run --audio-native --benchmark none --language <code> --domain <domain> --num-tasks 2`) | | | |

For the live smoke, record provider/model, reasoning setting where supported,
caller model/settings, language, domain, selected tasks, run revision and result
location. Explain failures. A smoke establishes execution, not judge accuracy or
a benchmark score. If not run, write **not run** and explain why.

## Agent judges (optional)

For a user-only submission, write **No judges included**. Otherwise list every
submitted judge separately; one language can contain a mixture of statuses.

| Judge ID | Purpose | New / adapted from | Exact version | Status | Applicability | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | unvalidated / validated | | |

For each judge, include:

- Model/settings, prompt/rubric version, evaluation unit and judging method.
- Language/locale and participant restrictions, including agent gender if relevant.
- Agent expectations, exceptions and natural/violating agent examples. Do not
  copy caller persona behavior into agent requirements.
- Candidate artifact and exact commands used to build and execute it.
- For **unvalidated**: what validation is missing, failed or inconclusive.
- For **validated**: frozen evidence, computed validation report, reviewer
  approval and production catalog change matching that exact contract/scope.

An author-written status does not enable production scoring. Unvalidated judges
run only when explicitly selected as candidates. Promotion follows the evidence
requirements in the judge-validation guide.

## Reproduction

Provide exact commands and relative paths to source requests, reviewed artifacts,
pack, voice recordings and judge evidence. Record dependencies and configuration
without credentials. Make evidence accessible to reviewers; do not require an
internal annotation application to inspect it.

## PR checklist

- [ ] README and required pack/task files included; staged scratch files excluded.
- [ ] User content and voices reviewed; automated checks and smoke status recorded.
- [ ] Domain goals, authentication policy and frozen subsets preserved.
- [ ] Optional judges have explicit individual status and scope.
- [ ] Validation claims reproduce against the submitted versions.
- [ ] User-only or unvalidated submissions do not enable new production diagnostics.
