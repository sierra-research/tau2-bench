.PHONY: all clean figure

# The figure renderer writes all three PDFs in one invocation.
.NOTPARALLEL:

REPO_ROOT := ../../..
OUTPUT := ray.pdf
LATEX_JOB := ray
FIGURE_SCRIPT := figures/render_language_system_metrics.py
FIGURES := figures/model_summary.pdf figures/model_metric_heatmap.pdf \
	figures/language_system_heatmaps.pdf
EXPERIENCE_ARTIFACT := $(REPO_ROOT)/papers/tau-multilingual/reproduction/experience.json
SOURCE_ANALYSIS_ARTIFACT := $(REPO_ROOT)/data/analysis/tau_multilingual_experience_without_fluency_2026-09-03.json

all: $(OUTPUT)

figure: $(FIGURES)

$(FIGURES): $(FIGURE_SCRIPT) $(EXPERIENCE_ARTIFACT) $(SOURCE_ANALYSIS_ARTIFACT) $(REPO_ROOT)/pyproject.toml $(REPO_ROOT)/uv.lock
	cd $(REPO_ROOT) && uv run --frozen --extra experiments python \
		papers/tau-multilingual/v3/$(FIGURE_SCRIPT)

$(OUTPUT): main.tex references.bib spconf.sty IEEEbib.bst $(FIGURES)
	pdflatex -jobname=$(LATEX_JOB) -interaction=nonstopmode -halt-on-error main.tex
	bibtex $(LATEX_JOB)
	pdflatex -jobname=$(LATEX_JOB) -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -jobname=$(LATEX_JOB) -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
	$(RM) ray.aux ray.bbl ray.blg ray.log ray.out ray.pdf
