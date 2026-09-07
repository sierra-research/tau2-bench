# Source gaps

## Missing frozen source artifacts

All locally recoverable results pass the executable audit. The following older ablation sources are still required for complete paper-claim coverage:

- **one-field composition row**: The exact 210-task x three-trial source root is not present in the frozen local evidence.
- **one-field-at-a-time, no-retry task score**: The separate no-retry run reported as 0.526 is not present. The retained verify/retry calls reproduce the reported 0.698 field score under a first-attempt counterfactual, but give 0.519 task success.

## Missing re-execution tools

The compact rows and reported outputs remain available, but the exact programs used for the following API/statistical passes were not present in the frozen local evidence:

- **Caller-realism causal contrasts**: the exact known-propensity Hajek estimator and task-clustered interval driver behind Figure 4. The assigned-realism ledger and plotted estimates are retained; the executable audit checks the ledger, not the causal estimates.
- **Caller-voice omnibus tests**: the exact Monte Carlo Fisher--Freeman--Halton driver behind the reported voice p-values. The compact call rows and the fully specified test settings remain available.
- **Speech-judge API runner**: all 6,422 utterance judgments, the exact v6 prompt/configuration, audio hashes, and final human-validation map are retained. This main-based release does not include the later API-backed convenience runner for regenerating those judgments from the detached audio corpus.
