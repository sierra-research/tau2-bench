# Frozen task subsets

Voice V2 uses the checked-in `airline_50`, `retail_50` and `telecom_50` artifacts.
Each contains canonical IDs in source-frame order, a source split and ID digest,
stratum counts, seed and creation provenance. Published IDs never change in place.
The digest pins IDs/order; runtime pack/task hashes separately pin task content.

## Commands

```sh
tau2 tasks list
tau2 tasks show telecom_50 --ids
tau2 tasks verify
tau2 tasks verify --name retail_50
tau2 tasks subset --domain retail --name retail_30_review --size 30 --seed 17 --dry-run
tau2 tasks subset --domain retail --name retail_30_review --size 30 --seed 17
```

`verify` rejects modified official files, canonical-frame drift, changed strata
and IDs that no longer reproduce from their recorded draw. For official subsets,
it additionally checks every installed pack's declared domain binding through
explicit canonical metadata. An installation without language packs reports an
empty language list; it does not claim multilingual coverage. New exploratory
samples are checked against their source frame, not the packs containing only
the official 50 tasks.

`subset` requires a new name. It uses the existing domain stratifier and seeded
largest-remainder sampling, then returns selected IDs in frame order. Requests
larger than the population produce a census. An identical rerun keeps the original
artifact/provenance; changed content at the same name fails. There is no overwrite
flag, prefix shortcut, suffix inference, or fallback from missing data to all tasks.
Authoring another subset does not alter Voice V2's official denominator.

## Python API

```python
from tau2.task_subsets.store import load_subset, verify_subset

subset = load_subset("telecom_50")
verify_subset(subset)
canonical_ids = subset.task_ids
```

Draw defaults come from `DEFAULT_TASK_SUBSET_SIZE=50` and
`DEFAULT_TASK_SUBSET_SEED=42` in `config.py`. The root CLI passes them to the
package-owned `add_tasks_args()` registration function. Benchmark consumers use
the same loader and verification functions instead of drawing another sample.

The subset commands manage task selection artifacts; they do not change ordinary
V1 run selection.
