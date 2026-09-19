# Reviewer-release verification

This release candidate was rechecked on 2026-09-18 after promoting the
corrected Korean/Mandarin retail cohort and corrected Mandarin ablations. All
checks below are read-only with respect to the frozen result bundle unless an
output directory is explicitly supplied.

## Completed checks

| Surface | Result |
|---|---|
| Frozen cohort audit | 141 pass, 8 documented warnings, 0 failures |
| Core result tables | Task success, text success, text--voice gap, and trial stability regenerated from the corrected primary/text cohort; retail ablations regenerated from the corrected four-cell Mandarin replacement plus unchanged Hindi cells |
| Prompt archive | 2,813 objects verified; all 134 result cells and 10,140 simulations matched their frozen source inputs and historical language-pack values |
| Human-validation archive | 3,601 row-level human/judge comparisons (343 call-level and 3,258 utterance-level), 80 metric rows, and 104 file hashes verified |
| Combined-naturalness validation | Five language leaves and 311 utterances verified; F1/kappa span HI 0.814/0.634 to ES 0.979/0.965 |
| First-critical-error review | 180 final call reviews verified, 30 for each of EN, ES, PT, HI, KO, and ZH |
| Supporting language review | 136 native-language rows across all five languages, plus 58 supplemental call-experience findings |
| Fixed task frames | Six shipped subsets verified with zero problems |
| Historical judge bootstrap | The archived pre-correction replay remains byte-identical: 3,750 non-English calls, 59,680 utterances, and `experience.json` SHA-256 `2087ea23bcb9c14cd73cb945e70f53ffd0685d2c36c5492d3c7a325b44f1369f` |
| Active naturalness replay | 3,750 calls and 60,417 utterances; manifest `8f3a94f72f1a740df264562fb363845cf35d9c443726b22b1acd36067e1cb054`, identity `b3cde4227d98175c7c5ddcd3be53292f24dcc7f858f0289c1cb5efd64d0db849`, work fingerprint `36f5d814b9e9372ebeec77f5eb35fa7e7ee21c0c03e1f720853038f4fa1ebc51`, rebind receipt `08360a565673afe75255e39ddf063b2e5147be669ed791f00c0d74cf6efcc493` |
| Frozen validation projection | The versioned public projection has manifest `4f6bd97c3d9a1d8ae4179ae5a1b3990b13a055f3922e86c3957fd7da4b4c86a5`; its public rows reproduce the private source after removing only provenance and free-text fields |
| Active Experience reproduction | The corrected 4,500-call analysis has SHA-256 `2b1af33f3dfbb75886581c50e2bf8cad82d16a2c222fd162b794545c2bceab05`; its final join contains 60,265 eligible utterances |
| Final-window exclusions | The canonical 7,462-row CSV has SHA-256 `2813e0dc25a80fd3a60775dd1533326523fe0b569220f1010bdf78c91dd52841`; its fail-closed manifest has SHA-256 `d4f4abf5bfc9f56895678d194903d62193f1871aa82efa9a567dcba6f488d48e` |
| Active latency reproduction | Typed replay over 90 cells and 4,500 calls verifies 80 unchanged inputs and ten corrected Korean/Mandarin retail cells; SHA-256 `eb56617f3b216566a9147e49699500d30b9ed9b311af3586bda6b85d19c3e01f` |
| Retail-ablation transcripts | 360 calls, 8,675 delivered turns, 12 source-result hashes, and 360 source-simulation hashes verified; all four ablation rows reproduce |
| Offline tests and style | 67 focused replacement, naturalness, final-window, Experience, and latency tests passed; Ruff lint and format checks passed over 663 files |
| Manuscript | The targeted Mandarin-ablation table/prose update renders cleanly as a five-page submission PDF |
| Executable voice smoke | One Spanish airline call completed normally through the real OpenAI streaming path; 2/2 action checks and DB match passed, with no agent, caller, or infrastructure error. The communication criterion failed, so this one stochastic smoke received reward 0; it is a pipeline check, not a reported result. |

## Documented audit warnings

The strict frozen-evidence audit has no failures. Its 8 warnings remain visible
rather than being rewritten away:

- four early English telecom result headers omit `task_subset`; their exact task
  ids still equal the frozen `telecom_50` frame;
- the current effective EN, HI, KO, and ZH packs differ from values recorded by
  some paper runs. The exact run-time values are preserved in the prompt archive,
  and `PACK_SNAPSHOT_DIFFERENCES.md` enumerates every difference.

## Release scope

Validation and FCE cover English and the five localized languages; supporting
language-review rows remain scoped to the five localized languages. Korean and
Mandarin validation remains on the archived pre-correction retail cohort, in
which caller name roles were underspecified; active reported metrics use the
corrected calls. A separate retail-localization review export is outside this
release's scope; the compact ablation transcripts and their source hashes
provide the evidence for the reported localization ablation. The deterministic
email-symbol check in ES, PT, KO, and ZH is verified directly in code and does
not require human-labeled rows.
