# τ-Multilingual v3 — ICASSP 2027

This directory is the full LaTeX manuscript workspace for the ICASSP 2027
submission.

## Conference constraints

Checked against the official ICASSP 2027 paper kit on 2026-09-08:

- Full-paper deadline: **2026-09-16**. The public call currently gives a date
  but not a time or time zone; verify the CMS portal before submission.
- Conference: **2027-05-16 through 2027-05-21**, Toronto, Canada.
- Regular-paper limit: **four pages of technical content**, with an optional
  fifth page containing only references, funding acknowledgements, and a
  Compliance with Ethical Standards statement. The paper kit is the final word
  when shorter conference pages disagree.
- Acceptance notification: **2027-01-13**.
- Final paper: **2027-01-27**.
- Author registration: **2027-02-10**.

Official sources:

- <https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php>
- <https://2027.ieeeicassp.org/call-for-papers/>
- <https://2027.ieeeicassp.org/paper-submission-instructions/>
- <https://2027.ieeeicassp.org/author-guidelines/>
- <https://2027.ieeeicassp.org/about/editorial-policies/>

## Template and companion-paper provenance

This draft follows the official ICASSP 2027 author kit and single-anonymous
review convention. Development or test submission pages are not treated as
conference guidance. Its compact section and float spacing is aligned with the
companion tau-Elicitation ICASSP 2027 manuscript; text size, line spacing,
margins, and column geometry remain those of the author kit.

## Build

Run:

```bash
uv sync --extra experiments
make
```

The build first regenerates the language-system figure through the locked
project environment, then uses `pdflatex` and BibTeX to write the submission
artifact `ray.pdf` in this directory. Use `make clean` to remove generated
LaTeX intermediates.
