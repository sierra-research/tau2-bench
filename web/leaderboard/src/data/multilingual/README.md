# τ-i18n Voice leaderboard

τ-i18n is the current name for τ-Multilingual (the legacy “multilingual”
benchmark); τ-Entity was formerly “elicitation”. Internal identifiers and
historical provenance keep their original names.

The Voice leaderboard's τ-i18n view is built only from submitted 2.0 packages
(`voice_v2_submissions` and `voice_v2_domain_submissions` in
`public/submissions/manifest.json`, loaded by `src/data/voiceV2Packages.js`).
The English-only leaderboard and progress view keep the 1.x submission-backed
behavior.

- **Task completion** comes from `benchmark-submission.json`: pass@1 with each
  language–domain cell weighted equally. Overall covers Airline, Retail and
  Telecom; banking and entity packages fill Banking and Intake on the same
  agent's row. Intake is the exception: it shows crossed Pass^3 from the entity
  package's `benchmark-pass3.json` (tasks passed in all three voice
  conditions), labelled pass^3, and “—” for a package without the file.
- **Interaction** and **Generation quality** come from the optional
  `benchmark-quality.json` (`voice-v2-interaction-call-v1` and
  `voice-v2-experience-call-v2`): passed / (passed + failed) calls, pooled over
  the scope shown. Unscored and excluded calls are left out. A package without
  the file shows “—” for both.
- The 3D frontier compares the three scores, higher being better; missing
  measurements cannot compete.

`metrics.js` holds shared formatting, Pareto and heatmap helpers. Run
`npm run build && npm run prerender && npm run test:e2e` from `web/leaderboard`.
