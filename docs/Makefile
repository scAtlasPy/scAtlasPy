SPHINXOPTS    =
SPHINXBUILD   = python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@test "$(BUILDDIR)" = "_build" || (echo "Refusing unsafe BUILDDIR=$(BUILDDIR)"; exit 1)
	rm -rf "_build"
	rm -rf "api/generated"

html:
	PYTHONPATH=../src $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
