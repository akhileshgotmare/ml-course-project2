DOC=doc
AUX=aux
_AUXILARY=groupNN-submission.aux groupNN-submission.log groupNN-submission.out groupNN-submission.toc
AUXILARY=$(patsubst %,$(AUX)/%,$(_AUXILARY))

all: tex

tex: $(AUXILARY)

$(AUXILARY): $(DOC)/groupNN-submission.tex
	if [ ! -d $(AUX) ]; then mkdir $(AUX); fi;\
	cd $(DOC);\
	pdflatex -draftmode -output-directory=../$(AUX) groupNN-submission.tex;\
	bibtex groupNN-literature;\
	pdflatex -draftmode -output-directory=../$(AUX) groupNN-submission.tex;\
	pdflatex -output-directory=../$(AUX) groupNN-submission.tex;\
	cd ..;\
	mv $(AUX)/groupNN-submission.pdf $(DOC);\

aspell:
	cat $(DOC)/groupNN-submission.tex | aspell -l en -t list | sort -u

clean:
	find . -name "*~" -type f -exec rm '{}' \;
	- rm $(AUX)/*
	- rmdir $(AUX)

distclean: clean
	- rm $(DOC)/groupNN-submission.pdf

.PHONY: aspell clean distclean

