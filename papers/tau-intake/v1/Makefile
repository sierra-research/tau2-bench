.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib spconf.sty IEEEbib.bst
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
