# Factory fixtures

The telecom factory fixture loads ordered task IDs directly from the canonical
`data/tau2/task_subsets/telecom_50.json` through the typed `load_subset` API,
which verifies the shipped artifact's digest.

The canonical subset freezes the existing sample, not a changed authentication
condition. Tests load task content and the database from this checkout's
canonical data.

The five locale directories hold the reviewed telecom identity maps and retail
tasks 39/63/68 extracted unchanged from an unpublished research snapshot, plus
the reviewed correction of task 63 (`retail_63_reviewed.json`). These are
regression inputs, including known defects; tests require the compiler to fix
only the inconsistent identity references and preserve the intentional failed
lookup and address-move conditions. Autoform tests use clearly labeled
fixture-only content built through AuthorForm; they do not author or publish any
real language content.
