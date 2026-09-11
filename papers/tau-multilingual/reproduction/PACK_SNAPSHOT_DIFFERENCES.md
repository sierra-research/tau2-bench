# Language-pack differences from the frozen runs

The prompt archive preserves the exact run-time language-pack values for every
reported simulation. Comparing only fields that could affect each run reveals
the following differences from the current checked-in packs.

| Language | Affected runs | Difference |
|---|---|---|
| English | The four repeated telecom voice arms | Lisa's backchannel list changed from seven phrases to `uh-huh`; Matt's changed from eight phrases to `mm-hm`. Their three background remarks were shortened. |
| Hindi | All reported Hindi cells | The greeting changed from `नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?` to `नमस्ते! बताइए, मैं आपकी कैसे मदद करूँ?`. |
| Korean | The four repeated telecom voice arms | Sixteen telecom glossary notes were translated from Korean explanatory prose to English. The term identifiers and spoken Korean terms are unchanged. |
| Mandarin | The four repeated telecom voice arms | Sixteen telecom glossary notes were translated from Chinese or mixed-language explanatory prose to English. The term identifiers and spoken Mandarin terms are unchanged. |

Spanish and Portuguese match their frozen run-time snapshots. No task-facing
retail additions are missing: the retail preset and retail glossary were
already present in every retail run snapshot. The English pack correctly has
no localized-entity glossary.

These differences do not alter the archived reproduction: every scored
simulation maps to its exact historical agent and user prompts under
`prompts/`. The machine-readable list of affected run profiles and fields is
in `audit.json` under the `pack-run-drift` findings.
