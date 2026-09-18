LATEXMK = latexmk
LATEXFLAGS = -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all manuscript supplement clean

all: manuscript supplement

manuscript: manuscript.pdf

manuscript.pdf: main.tex sections/introduction.tex sections/methods.tex sections/results.tex sections/discussion.tex sections/conclusion.tex bib/refs.bib
	$(LATEXMK) $(LATEXFLAGS) main.tex
	cp main.pdf manuscript.pdf

supplement: supplementary/supplement.pdf

supplementary/supplement.pdf: supplementary/supplement.tex
	cd supplementary && $(LATEXMK) $(LATEXFLAGS) supplement.tex

clean:
	$(LATEXMK) -C main.tex
	cd supplementary && $(LATEXMK) -C supplement.tex
	rm -f manuscript.pdf
