.PHONY: all clean figure repro

# The figure renderer writes all three PDFs in one invocation.
.NOTPARALLEL:

REPO_ROOT := $(abspath ../../..)
OUTPUT := ray.pdf
LATEX_JOB := ray
FIGURE_SCRIPT := figures/render_language_system_metrics.py
FIGURES := figures/model_summary.pdf figures/model_metric_heatmap.pdf \
	figures/language_system_heatmaps.pdf
PYTHON ?= uv run --frozen --extra experiments python
TASK_SUCCESS_ARTIFACT ?= $(REPO_ROOT)/data/analysis/tau_multilingual_task_success_significance_2026-09-18.json
EXPERIENCE_ARTIFACT ?= $(REPO_ROOT)/data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json
LATENCY_ARTIFACT ?= $(REPO_ROOT)/papers/tau-multilingual/reproduction/latency.json

all: $(OUTPUT)

define render_figures
	cd $(REPO_ROOT) && $(PYTHON) \
		papers/tau-multilingual/v3/$(FIGURE_SCRIPT) \
		--task-success-artifact $(TASK_SUCCESS_ARTIFACT) \
		--experience-artifact $(EXPERIENCE_ARTIFACT) \
		--latency-artifact $(LATENCY_ARTIFACT)
endef

# Explicit reproduction always reruns the renderer, even when PDFs are tracked.
figure:
	$(render_figures)

repro: figure
	$(MAKE) --no-print-directory $(OUTPUT)

$(FIGURES): $(FIGURE_SCRIPT) $(TASK_SUCCESS_ARTIFACT) $(EXPERIENCE_ARTIFACT) $(LATENCY_ARTIFACT) $(REPO_ROOT)/pyproject.toml $(REPO_ROOT)/uv.lock
	$(render_figures)

$(OUTPUT): main.tex references.bib spconf.sty IEEEbib.bst $(FIGURES)
	pdflatex -jobname=$(LATEX_JOB) -interaction=nonstopmode -halt-on-error main.tex
	bibtex $(LATEX_JOB)
	pdflatex -jobname=$(LATEX_JOB) -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -jobname=$(LATEX_JOB) -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
	$(RM) ray.aux ray.bbl ray.blg ray.log ray.out ray.pdf
