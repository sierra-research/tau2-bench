# Released elicitation bank provenance

The checked-in banks are the reviewed snapshot of September 16, 2026 from the
unpublished research repository. The task manifest
records every bank's SHA-256. Task reproduction begins with these banks and
does not require rebuilding their external source datasets.

Bank headers are preserved byte-for-byte because the manifest hashes the source
snapshot. References there to research commands such as `tau2 intake-names
build-foreign` and source design documents describe that historical pipeline;
those commands and documents are not shipped here. The supported reproduction
workflow is `tau2 intake-tasks freeze` / `verify`, described in
`docs/elicitation.md`.

Each bank contains 40 easy and 40 hard entries. The released task generator
selects ten entries from each tier of the ten active banks. Five additional
banks remain validated inputs to preserve the original provenance contract;
they are not part of the scored task selection.

| Bank | Source and validation |
|---|---|
| Person names | SSA given names and Census 2010 surnames; hard-name pronunciations from pinned CMUdict and WikiPron lexicons, with reviewed distortion metadata. |
| Medications | Curated generic names and strengths; pronunciation metadata from WikiPron and cited manual sources. The source snapshot includes an RxNorm finding for `Prilocaine 2% cream`; the frozen benchmark retains that entity verbatim. |
| Codes | VIN check digits and member-ID mod-97 checks enforced when loading banks and generating tasks. Submissions are normalized and scored against the expected record. |
| Phones | Values and decoys validated against fictional-number ranges for the US, UK, Australia, and France. |
| Dates and times | Frozen values use `YYYY-MM-DD` and `H:MM AM/PM`. The agent must submit the required format; scoring strips surrounding whitespace without rewriting dates or times. |
| Addresses | Synthetic street and postcode forms. |
| Properties | Invented property names with reviewed foreign-token pronunciation metadata. |
| Emails | Owner-linked patterns using coined domains; punctuation distinguishes decoys. |
| Coined names | Invented employer/company/brand names. |
| Amounts, insurance plans, vehicles, rate plans, shops | Retained source inputs; not selected for the released benchmark. |

Source extraction was recorded on August 25–26, 2026. The source pipeline used
[SSA national baby names](https://www.ssa.gov/oact/babynames/names.zip),
[Census 2010 surnames](https://www2.census.gov/topics/genealogy/2010surnames/names.zip),
[CMUdict at `0f8072f`](https://github.com/cmusphinx/cmudict/tree/0f8072f814306c5ee4fbf992ed853601b12c01f9),
[WikiPron at `d282e84`](https://github.com/CUNY-CL/wikipron/tree/d282e848a211ea31cfd730f0ced8bc8cdab9e83d),
and the [FreeBSD word list at `72f0bc8`](https://github.com/freebsd/freebsd-src/blob/72f0bc868bf00586cba1e50057d8f1998b4abe80/share/dict/web2).
These links document the frozen source lineage, not fresh registry checks.

Bank pronunciation columns retain phonemes, source labels, source-language
labels, citations where required, and reviewed mispronounced variants. Normal
speech uses the gold spelling. Only an armed elicitation pronunciation
complication substitutes its selected token; correct respellings are reference
metadata. Raw corpora and the bank-authoring pipeline are not required to
reproduce or run this task set.
