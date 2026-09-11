# Source gaps

## Missing frozen source artifacts

None. Both workflows in the release comparison have frozen sources.

## Intentionally excluded annotation inputs

Intermediate annotation material and free-text notes are intentionally excluded from the reviewer archive. The two release-safe validation bundles retain final structured labels and derived metrics only; this minimization is not a missing paper-claim artifact.

## Missing re-execution tools

None. The paper's statistical analyses, observed realism-event counts, mispronunciation repair-cost diagnostic, and caller-voice comparisons are re-executable from the compact reviewer archive.

## Detached source corpus

The approximately 41 GB audio/tick source corpus is available from [Google Drive](https://drive.google.com/drive/folders/1GAuTs3Naog5irE4J4MwILMTpFyJz-2dm?usp=sharing). Its `main_runs/`, `ablations/`, and `text_channel/` roots can be checked against the compact archive's recorded SHA-256 values with `tau2 paper elicitation-verify --evidence-root`.
