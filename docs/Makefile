SPHINXOPTS    =
SPHINXBUILD   = python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build
HOST          = 0.0.0.0
PORT          = 8010

.PHONY: help clean html serve

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@test "$(BUILDDIR)" = "_build" || (echo "Refusing unsafe BUILDDIR=$(BUILDDIR)"; exit 1)
	rm -rf "_build"
	rm -rf "api/generated"

html:
	PYTHONPATH=../src $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

serve:
	python -m http.server --bind "$(HOST)" "$(PORT)" -d "$(BUILDDIR)/html"
