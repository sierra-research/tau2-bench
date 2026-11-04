---
name: add-a-language
description: Add a language to the tau2 voice benchmark through the language factory, including reviewed user content, localized tasks, voices and a contribution PR. Optionally author separate agent judges. For extending an existing pack to another supported domain, use add-a-domain.
---

# Add a language

Read [language authoring](../../../docs/language-authoring.md) and use the
[submission template](../../../docs/language-submission-template.md) for the
README beside the new `pack.yaml`. Those documents own the submission format and
checks; keep the skill focused on executing them.

## Establish the submission

Determine the language code, locale/variety, supported domains and whether the
submission is user-only or user-and-judge. Use existing session choices. Supported
factory domains are airline, retail and telecom; adding a new domain implementation
is separate work. Adding a pack does not change the frozen official benchmark. The ordinary
`tau2 run --audio-native` command selects only the released manifest; use
`--benchmark none --language <language> --domain <domain>` for candidate runs.
Including a contributed language in the overall score requires an explicit new
benchmark release/revision and results on the expanded population.

Use a working branch and a new staging directory. Inspect `tau2 factory --help`
and each selected command's help. Requests are typed JSON: obtain fields from the
stage's Pydantic model/schema, rather than inventing parameters or pack fields.
Choose generation settings explicitly and retain requests with their artifacts.

For an offline-only request, prepare inputs, reuse exact supplied artifacts and run
local validation. `autoform`, `draft` and `prepare-voices` call providers (see the
credentials table in language authoring); `prepare-voices` calls ElevenLabs even
with existing voice IDs. Skip those calls and the live smoke unless authorized. An
offline draft PR may record missing generation, review or live evidence explicitly;
it must not claim completed onboarding.

## Build and review

Follow "Create a language" in the authoring guide. Each stage prints its output
path as JSON, plus digests where a later stage needs them; pass digests forward
instead of recomputing them.

1. Run `autoform` and review the locale/register and persona sketches.
2. Run `tau2 factory init-request` with the reviewed form, explicit model and
   reasoning effort, and the regional settings. It writes `draft-request.json` with the frozen-50 tasks for each
   domain, English caller genders, the task-ID convention and `$artifact`
   references to the inventories. Do not hand-assemble task lists or use a
   task-file prefix. Add hand-authored `voice_guidelines` only when needed.
3. Run `draft`, then open `draft.review/index.html`. Review the staged agent/user
   content, entity mappings and the before/after flag table. Preserve source task
   goals, normal authentication and lookup behavior. Keep acoustic instructions in
   the separate voice brief. Spell only when asked; local word anchors require an
   explicit request for anchors. The human reviewer records approval with
   `tau2 factory approve --gate content`.
4. Run `prepare-voices` with either existing voices or voice-design inputs, export
   `audition`, and obtain listening review, recorded with
   `tau2 factory approve --gate listening` against the printed `listening_sha256`.
   Confirm persona gender and voice mapping.
5. Run `finalize` with a request referencing the draft, assets and listening
   approval. Publish into a new output directory, install it under
   `data/tau2/multilingual/<code>/`, run `tau2 languages validate <code>` and
   `tau2 languages render-prompts --language <code> --output <review-directory>`,
   then the smoke gate `tau2 run --audio-native --benchmark none --language <code>
   --domain <domain> --num-tasks 2`. Complete the README alongside the pack and
   link exact artifacts, checks, review and smoke evidence. Finalizing locally does
   not release the pack.

Generate linguistic material through the fixed factory prompts, not by writing
unrecorded model content directly into packs. Human reviewers may edit staged
content, then approve its new digest with `tau2 factory approve`; never write
approval JSON by hand. Retain already supplied approvals when they still match the
exact artifact. A coding agent must not fabricate native-speaker
review, listening review or a reviewer's identity.

On retries, reuse unchanged requests and artifacts. For uncertain remote voice
creation, inspect the provider account and use `record-created-voice` only with a
confirmed ID matching the recorded request. Do not retry creation blindly. Keep
content-addressed audio files with their manifests when moving artifacts.

For family-first name conventions, set the pack/form `name_order` metadata.
The shared retail compiler supplies explicit given/family-name facts from the
structured caller identity; review that output instead of adding language-specific
manual prompt patches. Existing agent-clause assembly stays in the runtime.

## Optional agent judges

Read [judge validation](../../../docs/judge-validation.md). Agent criteria are
independent of caller persona behavior. With the judge-authoring commands installed,
use `draft-judge` for a naturalness extension, an adapted catalog diagnostic or a
new diagnostic, then review and `compile-judge`. Use `judges candidate` for explicit
experimental execution. Every judge starts unvalidated.
To contribute a reusable authoring source, run `factory register-judge` into
`data/tau2/judge_catalog/`. `factory list-judges` and future adaptations discover
reviewed entries there; registration does not enable production scoring.

Validation requires frozen held-out human evidence and a computed report. Only
`judges promote` with exact review/evidence can produce a production registry
change. A user-only submission or an unvalidated judge submission is valid; do
not invent evidence to complete the workflow. Keep status and scope per judge in
the README. Do not import annotation tooling.

## Verify and submit

Run the documented offline checks and a bounded live smoke using explicitly
recorded provider/model settings. Inspect the actual prompts, identity/tool
behavior, voice mapping and results. Report unavailable live checks honestly;
offline tests do not prove native quality. Do not launch a full benchmark unless
requested. Preserve frozen paper runs.

Submit the pack/task data, completed README, relevant judge definitions/evidence
and tests through a PR. Exclude credentials, transient staging files and internal
design notes. Report user validation separately from judge validation and keep
production judge activation explicit in the PR.
