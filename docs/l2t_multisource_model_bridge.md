# L2T Multisource Model Bridge

This bridge exports tau2-owned tool-calling datasets into the pickle contract
used by Minxing Zheng's `physics_informed_testing/share_code/experiment/run_baseline.py`.
The implementation is designed to match the split and external-data logic
observed in that entry point; it is not a verified statement that every
Minxing-side semantic assumption matches these tool-calling labels.

The Minxing loader expects one pickle containing:

- `X`: `float32` array, shape `(N, d_x)`.
- `y`: binary array, shape `(N,)`.
- `traj["s"]`: `float32` array, shape `(N, T)`.

The generated BFCL and API-Bank pickles are local ignored derived artifacts.
Regenerate them with the converter scripts below. Track the compact manifests,
diagnostic JSON/CSV outputs, docs, scripts, and tests instead of committing the
binary pickle files. The earlier tau2 compatibility pickle,
`data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl`, is
also a local ignored prerequisite unless regenerated through the existing tau2
pipeline.

The loader casts these arrays to `float32`, builds one-step sequence pairs from
`traj["s"]`, and performs a deterministic 80/20 train/validation split with
`numpy.random.RandomState(seed).permutation`.

## Dataset Separation

tau2, BFCL, and API-Bank are exported separately. They are not pooled as IID
samples because their label scopes differ:

- tau2: `Y=1` means task-level benchmark success.
- BFCL: `Y=1` means BFCL evaluator correctness for a test case.
- API-Bank: `Y=1` means a correct API call under the current pilot definition.

`Y=0` labels are preserved. The bridge does not replace labels with all ones.

## BFCL Mapping

Script:

```bash
python scripts/convert_bfcl_to_l2t_pkl.py
```

Default output:

- `data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t.pkl`
- `data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t_manifest.json`

`X` is a 17-dimensional structural vector derived from BFCL prompt/tool-schema
context only:

- category one-hot indicators
- question message count
- log-scaled question text length
- function count
- log-scaled function name and description lengths
- parameter and required-parameter counts
- category structure flags

`traj["s"]` is a 32-step event sequence derived from `s_raw.model_result`:

- `0`: padding
- `1`: candidate function call
- `2`: arguments JSON parse success
- `3`: arguments JSON parse error
- `4`: scalar argument
- `5`: list or object argument
- `6`: null argument
- `7`: end

Function-call names and argument keys are sorted canonically during sequence
encoding for deterministic output. This trades away original JSON key order.

Excluded from model-facing arrays: evaluator error fields, label metadata,
synthetic flags, and `y`.

BFCL irrelevance rows are not removed or relabeled. In this category, correct
abstention is part of the evaluator definition, so call-vs-no-call behavior can
be a valid category-specific signal in `S`. Supervised diagnostics therefore
report all categories, non-irrelevance only, and irrelevance only, and should
not describe every BFCL `S` signal as broadly general tool-call competence.

## API-Bank Mapping

Script:

```bash
python scripts/convert_apibank_to_l2t_pkl.py
```

Default output:

- `data/processed/l2t/apibank/apibank_full_l2t.pkl`
- `data/processed/l2t/apibank/apibank_full_l2t_manifest.json`

API-Bank reuses the existing tau2-bench numerical artifact:

- `X = toolcalling_numerical_full.npz["X"][source_dataset == "api_bank"]`
- `traj["s"] = toolcalling_numerical_full.npz["S"][source_dataset == "api_bank"]`
- `y = toolcalling_numerical_full.npz["y"][source_dataset == "api_bank"]`

Minxing's current loader has no separate fixed-vector `S` input. To avoid model
code changes, each existing API-Bank `S` row is supplied as `traj["s"]`, letting
the loader form one-step sequence pairs. The manifest records this compatibility
mapping explicitly.

Excluded from model-facing arrays follow the original numerical artifact's
leakage audit: label scope, label origin, synthetic flag, corruption type,
validation status/error, and `y`.

API-Bank rows are paired: each reference positive has a synthetic corrupted
negative with the same pre-call context. The converter preserves `sample_ids`
and stores `pair_id` as non-model metadata in `metadata[].pair_id` and
`group_ids`; neither is placed in `X` or `traj["s"]`. Supervised diagnostics use
these group IDs for the primary API-Bank split. The synthetic negatives are not
natural LLM failures, and API-Bank diagnostics should not be presented as
natural deployment success-rate evidence.

Many API-Bank structural `S` slots are constant or inert in the current pilot,
and the corrected diagnostic audit reports those columns explicitly. The
presence of structural slots in the fixed-width representation does not imply
that every slot is informative.

## Validation

Both converters validate:

- duplicate sample IDs
- shape consistency across `X`, `y`, and `traj["s"]`
- binary labels
- NaN values
- infinite values
- deterministic train/validation split summaries
- class distributions

The converter manifests retain the Minxing-style row split summary for
compatibility, but API-Bank supervised diagnostics must split by pair/group to
avoid pair leakage.

Run targeted tests:

```bash
pytest tests/test_l2t_model_bridge_converters.py
```

Run lint on bridge files:

```bash
ruff check scripts/l2t_model_bridge.py scripts/convert_bfcl_to_l2t_pkl.py scripts/convert_apibank_to_l2t_pkl.py tests/test_l2t_model_bridge_converters.py
```
