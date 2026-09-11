.PHONY: all clean figure4

FIGURES = figures/task_generator_pipeline.pdf \
	figures/accuracy_effort_dumbbell.pdf \
	figures/adaptivity_effort.pdf \
	figures/realism_assignment_effects.pdf \
	figures/composition_results.pdf

all: ray.pdf

ray.pdf: main.tex references.bib spconf.sty IEEEbib.bst $(FIGURES)
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=ray main.tex
	bibtex ray
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=ray main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=ray main.tex

figure4: figures/realism_assignment_effects.pdf

figures/realism_assignment_effects.pdf: \
		figures/render_realism_assignment_effects.mjs \
		analysis/intake_realism_effects_agent_directed_2026-09-11.json \
		analysis/intake_realism_effects_scaffolded_2026-09-11.json
	node figures/render_realism_assignment_effects.mjs

clean:
	$(RM) ray.aux ray.bbl ray.blg ray.log ray.out ray.pdf \
		main.aux main.bbl main.blg main.log main.out main.pdf
