# tau-Voice Presentation Makefile
# Usage: make build

LATEX = xelatex
LATEXFLAGS = -interaction=nonstopmode
MAIN = tau-voice-slides

.PHONY: build clean watch open help

help:
	@echo "tau-Voice Presentation - Build Commands"
	@echo ""
	@echo "  make build   - Compile the presentation to PDF"
	@echo "  make clean   - Remove auxiliary files"
	@echo "  make open    - Open the PDF (macOS)"
	@echo "  make watch   - Watch for changes and rebuild (requires fswatch)"
	@echo ""

build: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex *.sty
	@echo "Building presentation..."
	@$(LATEX) $(LATEXFLAGS) $(MAIN).tex > /dev/null
	@$(LATEX) $(LATEXFLAGS) $(MAIN).tex > /dev/null
	@echo "Done! Output: $(MAIN).pdf"

clean:
	@rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz
	@echo "Cleaned auxiliary files."

open: build
	@open $(MAIN).pdf 2>/dev/null || xdg-open $(MAIN).pdf 2>/dev/null || echo "Please open $(MAIN).pdf manually"

watch:
	@echo "Watching for changes... (Ctrl+C to stop)"
	@fswatch -o $(MAIN).tex *.sty | xargs -n1 -I{} make build
