# Intake ICASSP paper draft

This directory contains the standalone paper on compositional interactive
entity capture. It follows the official ICASSP 2027 paper kit, submission
format, and single-anonymous review convention. Development or test submission
pages are not treated as conference guidance.

Build with:

```bash
make
```

The submission PDF is written to `ray.pdf`.

Figure 1 can be regenerated with:

```bash
node figures/render_task_generator_pipeline.mjs
```

This requires Chrome or Chromium and Ghostscript. The renderer flattens the
browser-generated transparency so the figure displays consistently in Quartz
and other PDF viewers.

`RESULT_SOURCES.md` separates completed formative results from the planned
confirmatory evaluation. Do not convert planned cells into reported findings
until provenance-bearing analysis artifacts exist.
