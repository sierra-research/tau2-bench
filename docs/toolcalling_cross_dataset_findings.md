# Cross-Dataset Tool-Calling Shift Findings

This note compares the current tau2, API-Bank, and BFCL studies as complementary evidence. They use different label scopes and should not be pooled as IID rows.

## tau2

- 105 real task-level outcomes after Stage 1.
- `tau2_zero_or_one_write_to_two_plus_writes` remains `candidate_harmful` at `d=0.05` and `d=0.10`.
- The same multiple-write shift is `inconclusive` at `d=0.15`.
- All other tau2 shifts are inconclusive under the full-CI rule.

## API-Bank

- 1,016 API-call-level correctness records.
- 508 reference positives and 508 synthetic negatives.
- Useful for representation, schema, and evaluator-pipeline development.
- Not valid for estimating natural deployment success-rate shifts because the negative labels are synthetic corruptions and labels are balanced by construction.

## BFCL

- 1,240 real evaluated test-case-level outcomes.
- All BFCL shifts are `inconclusive` at `d=0.05`.
- At `d=0.10`, `bfcl_simple_python_to_multiple` and `bfcl_simple_python_to_parallel` are `candidate_harmless`; the other four BFCL shifts are `inconclusive`.
- At `d=0.15`, five BFCL shifts are `candidate_harmless`; `bfcl_multiple_to_parallel_multiple` remains `inconclusive`.
- `bfcl_simple_python_to_irrelevance` is explicitly a behavioral/abstention contrast, not a primary complexity shift.
- No BFCL shift is `candidate_harmful` or `candidate_beneficial` under the full-CI rule.

## Interpretation

These are complementary studies with different label scopes: tau2 is task-level, API-Bank is API-call-level with synthetic negatives, and BFCL is evaluated test-case-level. Cross-dataset summaries are bookkeeping comparisons only, not pooled IID estimates.
