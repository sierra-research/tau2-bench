# Human validation

Final human labels and row-aligned stored judge outputs for retained paper measures.

- Rows are partitioned once as `<evaluation-level>_level/<measure>/<language>/validation.csv`.
- Call-level leaves contain one whole-call comparison per row; utterance-level leaves contain one stable utterance comparison per row, aligned either to an agent turn or an exact audio segment.
- `utterance_alignment` identifies that authority; audio segments carry their exact start/end window and never claim a single agent-turn id.
- Speech-fidelity rows preserve the cleaned delivered reference in `agent_utterance` and v5's exact, at-most-200-character metadata preview in `stored_reference_preview`; `stored_reference_alignment` is verifier-enforced as `exact` or `stored_prefix` after whitespace normalization. The full reference, not only this preview, was prompted.
- Speech-fidelity validation tests the underlying detector at any retained severity after the final-second filter; the paper's downstream Experience and standalone summaries separately require severity 2 or higher.
- `precision` and `recall` identify sampling strata in `source_sets`; they are not directory levels.
- `index.csv` selects one paper-facing metric per language and measure. Fixed packages report their recorded `held_out` or `fixed_curated` set; precision/recall corpora report `unified`.
- The archive contains only final rows for the paper-facing measures.
- `stored_prediction_prompts` is the sole learned-judge prompt inventory. It preserves only the exact contracts used for the frozen verdicts. A multi-factor contract can mention criteria outside the paper-facing measure set when those criteria were present in the original batched judge call; their unused outputs are not shipped as measures.
- `gate_status` applies the frozen evidence rule: precision and recall must each exceed 0.75, F1 must exceed 0.80, at least ten human violations must be present, and Cohen's kappa must exceed 0.60 when there are at least five human negatives.
- Runtime factors and tool-use subtypes remain in `source_factor_ids`; acceptance gates apply only to `measure_id`.
- Every `metrics.csv` is derived from its sibling `validation.csv`; leaf manifests bind both by SHA-256, and the root manifest binds every leaf.
- Portuguese `gender_agreement` validates the frozen v15/v19 female-agent frame recorded in `prompts.json`; it does not validate a later runtime contract.

| Language | Measure | Unit | Design | Pos/neg | Gate status | N | Precision | Recall | F1 | κ |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| en | speech_fidelity | utterance | fixed_curated | 10/50 | passes | 60 | 0.769 | 1.000 | 0.870 | 0.839 |
| en | tool_use | call | unified_precision_recall | 29/13 | passes | 42 | 0.967 | 1.000 | 0.983 | 0.943 |
| es | naturalness | utterance | fixed_curated | 24/36 | passes | 60 | 1.000 | 0.958 | 0.979 | 0.965 |
| es | speech_fidelity | utterance | fixed_curated | 15/45 | passes | 60 | 1.000 | 0.933 | 0.966 | 0.955 |
| es | tool_use | call | fixed_curated | 35/8 | passes | 43 | 0.946 | 1.000 | 0.972 | 0.830 |
| es | counting_units | utterance | fixed_curated | 20/262 | passes | 282 | 0.952 | 1.000 | 0.976 | 0.974 |
| pt | naturalness | utterance | fixed_curated | 33/30 | passes | 63 | 0.938 | 0.909 | 0.923 | 0.841 |
| pt | speech_fidelity | utterance | fixed_curated | 11/49 | passes | 60 | 0.917 | 1.000 | 0.957 | 0.946 |
| pt | tool_use | call | fixed_curated | 24/15 | passes | 39 | 1.000 | 1.000 | 1.000 | 1.000 |
| pt | register_formality | call | fixed_curated | 14/19 | passes | 33 | 0.933 | 1.000 | 0.966 | 0.939 |
| pt | gender_agreement | utterance | fixed_curated | 13/295 | passes | 308 | 0.867 | 1.000 | 0.929 | 0.925 |
| pt | counting_units | utterance | fixed_curated | 10/206 | passes | 216 | 1.000 | 1.000 | 1.000 | 1.000 |
| pt | regional_consistency | call | fixed_curated | 5/25 | insufficient_positive_support | 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| hi | naturalness | utterance | fixed_curated | 28/32 | passes | 60 | 0.774 | 0.857 | 0.814 | 0.634 |
| hi | speech_fidelity | utterance | fixed_curated | 21/39 | passes | 60 | 0.941 | 0.762 | 0.842 | 0.770 |
| hi | tool_use | call | fixed_curated | 32/12 | passes | 44 | 0.912 | 0.969 | 0.939 | 0.758 |
| hi | gender_agreement | utterance | fixed_curated | 30/30 | passes | 60 | 0.964 | 0.900 | 0.931 | 0.867 |
| hi | name_address_conventions | utterance | fixed_curated | 21/379 | passes | 400 | 0.955 | 1.000 | 0.977 | 0.975 |
| ko | naturalness | utterance | fixed_curated | 32/30 | passes | 62 | 0.811 | 0.938 | 0.870 | 0.708 |
| ko | speech_fidelity | utterance | fixed_curated | 16/44 | passes | 60 | 0.778 | 0.875 | 0.824 | 0.754 |
| ko | tool_use | call | fixed_curated | 31/5 | passes | 36 | 0.969 | 1.000 | 0.984 | 0.873 |
| ko | honorific_agreement | utterance | fixed_curated | 18/207 | passes | 225 | 0.900 | 1.000 | 0.947 | 0.943 |
| ko | name_address_conventions | utterance | fixed_curated | 4/199 | insufficient_positive_support | 203 | 0.800 | 1.000 | 0.889 | 0.886 |
| zh | naturalness | utterance | fixed_curated | 36/30 | passes | 66 | 0.914 | 0.889 | 0.901 | 0.787 |
| zh | speech_fidelity | utterance | fixed_curated | 23/36 | passes | 59 | 0.957 | 0.957 | 0.957 | 0.929 |
| zh | tool_use | call | fixed_curated | 39/3 | passes | 42 | 1.000 | 0.974 | 0.987 | 0.844 |
| zh | counting_units | utterance | fixed_curated | 19/434 | passes | 453 | 0.950 | 1.000 | 0.974 | 0.973 |
| zh | modal_particles | call | fixed_curated | 25/9 | passes | 34 | 1.000 | 0.920 | 0.958 | 0.859 |
| zh | name_address_conventions | utterance | fixed_curated | 3/437 | insufficient_positive_support | 440 | 1.000 | 1.000 | 1.000 | 1.000 |
