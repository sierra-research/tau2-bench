# L2T Supervised Diagnostic Baselines

This diagnostic evaluates whether the BFCL and API-Bank L2T bridge artifacts
contain predictive signal for `Y` under ordinary supervised classifiers. It is
not a replacement for Minxing's L2T objective; it is a controlled sanity check
for the bridge representations.

Artifacts:

- BFCL: `data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t.pkl`
- API-Bank: `data/processed/l2t/apibank/apibank_full_l2t.pkl`

These pickles are deterministic local derived artifacts. They are ignored by
Git and regenerated with `scripts/convert_bfcl_to_l2t_pkl.py` and
`scripts/convert_apibank_to_l2t_pkl.py`; the compact manifests and diagnostic
JSON/CSV outputs are the trackable artifacts.

Outputs:

- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_summary.json`
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_results.csv`
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_best_by_view.csv`
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_permutation_summary.csv`

## Split

BFCL supervised diagnostics use the deterministic row split implemented to
match the split logic observed in Minxing's referenced `run_baseline.py` entry
point:

```python
perm = np.random.RandomState(seed).permutation(N)
n_train = int(0.8 * N)
train_idx = perm[:n_train]
val_idx = perm[n_train:]
```

API-Bank supervised diagnostics use a deterministic grouped split by `pair_id`.
The artifact contains 508 positive/synthetic-negative pairs. A row-level random
split placed 160/508 pairs across train and validation, so it is retained only
as a leakage audit number, not as the primary clean diagnostic split.

The grouped split uses `seed=1`, permutes sorted pair IDs, assigns 406 pairs to
training and 102 pairs to validation, and keeps all rows from a pair in the same
split. The resulting row counts are 812 train rows and 204 validation rows,
with class counts 406/406 and 102/102. Cross-split pair count is asserted to be
zero.

## Feature Views

Each dataset is evaluated separately, without pooling:

- `X-only`
- `S-only`
- `X+S`

For the Minxing-compatible pickle, `S` means `traj["s"]`.

## Baselines

The script evaluates:

- majority class
- stratified random
- logistic regression with train-only standardized features
- class-weighted logistic regression with train-only standardized features
- random forest
- histogram gradient boosting
- deterministic small MLP with train-only standardized features

Preprocessing is fitted on training data only through sklearn pipelines.

## Permutation Controls

Every dataset/subset, feature view, and model is also run with 10 deterministic
training-label permutations. Validation labels are not shuffled. The primary
null artifact reports `n_trials`, mean, standard deviation, minimum, and maximum
for accuracy, balanced accuracy, and macro-F1. API-Bank permutation controls use
the same grouped split as the corresponding non-permuted diagnostic.

## Results Summary

Best non-permuted model per dataset/view:

| Dataset | Subset | View | Best model | Accuracy | Balanced accuracy | Macro-F1 | Confusion matrix |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BFCL | all categories | X-only | class-weighted logistic regression | 0.5444 | 0.5406 | 0.4448 | `[[15, 13], [100, 120]]` |
| BFCL | all categories | S-only | class-weighted logistic regression | 0.5887 | 0.5812 | 0.4785 | `[[16, 12], [90, 130]]` |
| BFCL | all categories | X+S | small MLP | 0.9032 | 0.6338 | 0.6737 | `[[8, 20], [4, 216]]` |
| BFCL | non-irrelevance | X-only | class-weighted logistic regression | 0.6500 | 0.5644 | 0.4982 | `[[10, 12], [58, 120]]` |
| BFCL | non-irrelevance | S-only | class-weighted logistic regression | 0.6600 | 0.6895 | 0.5467 | `[[16, 6], [62, 116]]` |
| BFCL | non-irrelevance | X+S | class-weighted logistic regression | 0.7150 | 0.6606 | 0.5667 | `[[13, 9], [48, 130]]` |
| BFCL | irrelevance only | X-only | class-weighted logistic regression | 0.5833 | 0.6000 | 0.5152 | `[[5, 3], [17, 23]]` |
| BFCL | irrelevance only | S-only | logistic regression | 1.0000 | 1.0000 | 1.0000 | `[[8, 0], [0, 40]]` |
| BFCL | irrelevance only | X+S | logistic regression | 1.0000 | 1.0000 | 1.0000 | `[[8, 0], [0, 40]]` |
| API-Bank | all pairs | X-only | small MLP | 0.5000 | 0.5000 | 0.4988 | `[[46, 56], [46, 56]]` |
| API-Bank | all pairs | S-only | random forest | 0.9461 | 0.9461 | 0.9460 | `[[93, 9], [2, 100]]` |
| API-Bank | all pairs | X+S | small MLP | 0.9510 | 0.9510 | 0.9509 | `[[93, 9], [1, 101]]` |

Selected repeated shuffled-label controls:

| Dataset | Subset | View/model | Trials | Balanced accuracy mean/std/min/max | Macro-F1 mean |
| --- | --- | --- | ---: | ---: | ---: |
| BFCL | all categories | X+S small MLP | 10 | 0.5087 / 0.0108 / 0.4886 / 0.5286 | 0.4975 |
| BFCL | irrelevance only | S-only logistic regression | 10 | 0.5062 / 0.0187 / 0.5000 / 0.5625 | 0.4662 |
| API-Bank | all pairs | S-only random forest | 10 | 0.4966 / 0.0425 / 0.4265 / 0.5637 | 0.4954 |
| API-Bank | all pairs | X+S small MLP | 10 | 0.5088 / 0.0255 / 0.4608 / 0.5441 | 0.5082 |

## Interpretation

BFCL contains supervised signal, but it is not one uniform signal. The
all-category `X+S` result is partly influenced by the irrelevance category,
where correct abstention is part of the evaluator definition. In irrelevance
rows, correct abstention is usually free text/no function call and incorrect
behavior is typically a function call. That call-vs-no-call distinction is a
category-definition-specific abstention signal, not general tool-call
competence. The non-irrelevance rows are therefore reported separately.

API-Bank has strong signal in `S`, not in `X`, even after pair-grouped splitting.
This is consistent with the construction: positive and synthetic-negative pair
members share pre-call `X`, while candidate-call `S` differs. The negatives are
synthetic corruptions, not natural LLM failures, so these scores remain
representation/evaluator diagnostics rather than deployment-shift evidence.

The corrected leakage audit reports no exact `y` or `1-y` columns in API-Bank
`X` or `S`. It does report many constant or inert structural slots in the
current pilot: API-Bank `X` has 12 constant columns and API-Bank `S` has 15
constant columns. Do not interpret every structural slot as informative.

## Scientific Diagnosis

The supervised baselines show that the bridge can expose `Y` signal, especially
API-Bank `S` and BFCL `X+S`. The Minxing `proposed_only` collapse is therefore
more consistent with objective mismatch than with a completely signal-free data
bridge.

However, the API-Bank result is still synthetic-negative evidence and the BFCL
irrelevance result is partly abstention-definition-specific. Neither should be
overclaimed as semantic compatibility with Minxing's reconstruction objective or
as deployment-shift evidence. The scientifically safer next step remains to
clarify how `Y` is intended to enter training and how tool-calling success
should map to Minxing's reconstructed-trajectory score.

## Command

```bash
python scripts/evaluate_l2t_supervised_baselines.py
```
