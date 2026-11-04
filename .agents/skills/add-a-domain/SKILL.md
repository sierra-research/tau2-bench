---
name: add-a-domain
description: Localize another existing tau2 domain for a language pack through the factory, preserving reviewed personas and reusing voices. Use for adding airline, retail or telecom coverage to an existing language, not for implementing a new benchmark domain.
---

# Add a supported domain to a language

Read the extension workflow in [language authoring](../../../docs/language-authoring.md)
and the [submission template](../../../docs/language-submission-template.md).
The language's README must describe the new coverage and its verification.

Respect offline-only scope: reuse exact supplied artifacts, prepare requests and
run local checks. `add-domain` with `localization.mode="reuse"` is offline;
`mode="generate"` calls an LLM. Auditioning even existing voice IDs calls the
provider. Mark missing generation/review/live evidence in a draft PR rather
than running those steps without authorization or claiming completion.

## Extend the pack

1. Identify the existing pack, its exact task bundles, requested supported domain
   and a new revision. Preserve source pack files. Confirm that the new domain is
   not already bound; replacing reviewed data requires a rebuilt revision.
2. Run `tau2 factory stage-pack` using the exact published pack and domain bundles.
3. For the new domain, run `tau2 factory init-request --domains <domain>` with the
   language's reviewed author form (`--form`), `--reviewed-by`, an explicit
   `--model` and `--reasoning-effort`, and the pack's regional settings. Use its emitted
   `domains[0]` entry (frozen-50 tasks, caller genders, task IDs and `$artifact`
   references, kept relative to the new request file) as the `domain` of an
   `AddDomainRequest`. Do not hand-assemble task lists.
4. Run `add-domain` with that bundle and a new revision. Review
   `<output>.review/index.html` (content, mappings and before/after flags), and
   have the reviewer record `tau2 factory approve --gate content`.
5. When persona/voice content is unchanged, use `reuse-voices` with the original
   reviewed assets and their audio directory; `--output` is a new directory. The
   original listening approval still covers the reused voices. Otherwise run
   `prepare-voices`, `audition` and `tau2 factory approve --gate listening`.
   Imported packs without factory artifacts can audition their existing voice IDs.
6. Run `finalize` into a new directory and update the language's submission README.
   Install that local output in the contribution worktree and use `tau2 languages
   render-prompts --language <code> --domain <domain> --output <review-directory>`
   for full runtime prompt review before submission.
   Record task identity/lookup checks and the smoke gate `tau2 run --audio-native
   --benchmark none --language <code> --domain <domain> --num-tasks 2`.

Inspect command help and use the package's typed request models. Keep all source
requests and review digests. The stage/build commands own content generation and
validation; do not improvise translations or create ad hoc scripts to replace
them. Keep simulator prompts about text, spelling conditional on request and
authentication guidance in domain policy.

For family-first name conventions, set the pack/form `name_order` metadata.
The shared retail compiler supplies explicit given/family-name facts from the
structured caller identity; review that output instead of adding language-specific
manual prompt patches. Existing agent-clause assembly stays in the runtime.

## Judge coverage

An existing language judge is not automatically validated for newly claimed
conditions. Record whether the domain extension falls within its evidence scope.
If extending or adding a judge, follow [judge validation](../../../docs/judge-validation.md)
and keep candidate results separate until validated. User-only coverage can ship
without a new judge. Do not copy caller behavior into agent judging rules.

## PR

Include the reviewed pack/task change, updated README, reproducible checks and
optional judge evidence. Reuse voices when possible; do not recreate them just
because domain tasks changed. Do not change the benchmark's frozen language/task
selection or run a full benchmark as part of routine domain localization. Report
missing review or live evidence accurately; never manufacture approvals.
