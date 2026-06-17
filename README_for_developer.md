# Developer README

This guide is for team members who need to develop, test, or build the
documentation for scAtlasPy from a local checkout.

## 1. Clone the Repository

```bash
git clone https://github.com/scAtlasAnalysis/scAtlaspy.git
cd scAtlaspy
```

If you are already working in an existing checkout, run all commands from the
repository root unless a section says otherwise.

## 2. Create a Python Environment

Use a clean virtual environment or conda environment. Python 3.11 or newer is a
good default for local development.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Using conda:

```bash
conda create -n scatlaspy-dev python=3.11
conda activate scatlaspy-dev
python -m pip install --upgrade pip
```

## 3. Install the Package

For normal development, install the package in editable mode:

```bash
pip install -e .
```

Editable mode means changes under `src/scatlaspy/` are picked up without
reinstalling the package.

Verify the install:

```bash
python -c "import scatlaspy as sap; print(sap.Atlas)"
```

## 4. Optional Dependency Groups

Optional dependencies are defined in `pyproject.toml` under
`[project.optional-dependencies]`.

Install with test dependencies:

```bash
pip install -e ".[test]"
```

Install with documentation dependencies:

```bash
pip install -e ".[docs]"
```

Install with machine-learning dependencies:

```bash
pip install -e ".[ml]"
```

Install everything commonly needed for development:

```bash
pip install -e ".[test,docs,ml]"
```

## 5. Run Tests

Run the default test suite:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_l0_storage_exact.py
```

Run tests by marker:

```bash
pytest -m l0
pytest -m l1
pytest -m l2
```

Some tests are marked as slow, stress, realdata, or import_benchmark. These may
need large datasets or more memory and should not be part of a quick local check
unless you know the required inputs are available.

Examples:

```bash
pytest -m "not slow and not stress and not realdata"
pytest -m realdata
pytest -m import_benchmark
```

Run tests with coverage:

```bash
pytest --cov=scatlaspy
```

## 6. Build Documentation

Install the documentation dependencies first:

```bash
pip install -e ".[docs]"
```

Build the HTML documentation:

```bash
cd docs
make html
```

The generated site is written to:

```text
docs/_build/html/
```

Preview it locally:

```bash
python -m http.server --bind 0.0.0.0 8010 -d _build/html
```

Then open:

```text
http://localhost:8010
```

If you are working on a remote server, replace `localhost` with the server IP or
hostname.

## 7. Clean Documentation Outputs

To remove generated documentation files:

```bash
cd docs
make clean
```

This removes only the generated documentation directories:

```text
docs/_build/
docs/api/generated/
```

Do not manually edit files under `docs/_build/` or `docs/api/generated/`.
Regenerate them with `make html` instead.

## 8. What to Commit

Commit source code, tests, documentation source files, and configuration:

```text
src/
tests/
docs/*.md
docs/tutorials/
docs/how-to/
docs/developer/
docs/api/*.md
docs/conf.py
docs/Makefile
docs/_static/
docs/_templates/
pyproject.toml
README.md
README_for_developer.md
```

Do not commit generated outputs, caches, logs, local environments, or large data:

```text
docs/_build/
docs/api/generated/
__pycache__/
.pytest_cache/
.coverage
*.log
*.h5ad
*.h5
*.sasql
*.duckdb
data/
.venv/
```

Check the current Git state before committing:

```bash
git status --short --ignored
```

## 9. Common Workflows

After changing Python code:

```bash
pip install -e ".[test]"
pytest
```

After changing documentation:

```bash
pip install -e ".[docs]"
cd docs
make html
python -m http.server --bind 0.0.0.0 8010 -d _build/html
```

After changing public APIs:

```bash
pip install -e ".[test,docs]"
pytest
cd docs
make html
```

Public API changes should usually be reflected in `docs/api/*.md` and in the
relevant docstrings under `src/scatlaspy/`.
