.PHONY: all clean

all: ray.pdf

ray.pdf: main.tex references.bib spconf.sty IEEEbib.bst
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=ray main.tex
	bibtex ray
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=ray main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=ray main.tex

clean:
	$(RM) ray.aux ray.bbl ray.blg ray.log ray.out ray.pdf \
		main.aux main.bbl main.blg main.log main.out main.pdf
