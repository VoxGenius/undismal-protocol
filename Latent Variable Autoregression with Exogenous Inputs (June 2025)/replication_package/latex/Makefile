# Makefile for LaTeX compilation

# Default target
all: response_paper_clean.pdf

# Pattern rule for compiling any .tex file to .pdf
%.pdf: %.tex
	pdflatex -interaction=nonstopmode $<
	pdflatex -interaction=nonstopmode $<  # Run twice for cross-references

# Specific target for response_paper_clean
response_paper_clean.pdf: response_paper_clean.tex
	pdflatex -interaction=nonstopmode response_paper_clean.tex
	pdflatex -interaction=nonstopmode response_paper_clean.tex  # Run twice for cross-references

# Draft mode compilation (ignores missing figures)
draft: response_paper_clean.tex
	pdflatex -interaction=nonstopmode "\def\missing{}\input{response_paper_clean.tex}"
	pdflatex -interaction=nonstopmode "\def\missing{}\input{response_paper_clean.tex}"

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz

# Clean everything including PDFs
distclean: clean
	rm -f *.pdf

.PHONY: all clean distclean