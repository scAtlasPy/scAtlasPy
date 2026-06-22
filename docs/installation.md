# Installation

scAtlasPy requires Python 3.10 or later. It can be added to an existing
single-cell analysis environment or installed in a new environment created for
an analysis project.

## Install scAtlasPy

Activate the Python environment used for your project, then install the latest
stable release from PyPI:

```bash
pip install scatlaspy
```

Required runtime dependencies are installed automatically.

To upgrade an existing installation:

```bash
pip install --upgrade scatlaspy
```

```{note}
Before the first public PyPI release, install the current version from GitHub as
described in {ref}`install-from-source`.
```

## Verify the Installation

Start Python and import scAtlasPy:

```python
import scatlaspy as sap

print(sap.__version__)
```

The installed version should be printed without an import error.

## Create a New Project Environment

Skip this section when you already have a suitable Python environment for your
analysis project.

A project environment provides control over package versions while allowing
scAtlasPy and the other analysis, visualization, and machine-learning tools used
in the same workflow to be installed together.

For conda or mamba:

```bash
conda create -n atlas-project python=3.11 pip
conda activate atlas-project
pip install --upgrade pip
pip install scatlaspy
```

Replace `conda` with `mamba` when using Mamba.

For Python's built-in `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install scatlaspy
```

## Optional Dependencies

The standard installation includes the dependencies required for core
scAtlasPy workflows. Some computational methods may require additional
frameworks that are not installed automatically.

For example, machine-learning workflows may require PyTorch. Install such
frameworks according to the requirements of the corresponding workflow and your
computing environment. GPU-enabled packages should be selected to match the
available hardware and CUDA configuration.

The documentation for each optional workflow identifies any additional
dependencies that are required.

(install-from-source)=
## Install from Source

Install the current development version directly from GitHub to use features
that have not yet been included in a stable release:

```bash
pip install "scatlaspy @ git+https://github.com/GaoLab-XDU/scAtlasPy.git"
```

To modify scAtlasPy itself, clone the repository and install it in editable
mode:

```bash
git clone https://github.com/GaoLab-XDU/scAtlasPy.git
cd scAtlasPy
pip install -e .
```

Editable mode makes the active environment import scAtlasPy directly from the
source tree. Changes to Python source files are therefore available without
reinstalling the package.

Install the additional dependencies required for testing or documentation
development with:

```bash
pip install -e ".[test]"
pip install -e ".[docs]"
```

The dependency groups can also be installed together:

```bash
pip install -e ".[test,docs]"
```

Instructions for running the test suite and building the documentation are
provided in the contributor documentation.

## Troubleshooting

### `scatlaspy` cannot be imported

Confirm that scAtlasPy is installed in the environment used to run the
analysis:

```bash
pip show scatlaspy
python -c "import sys; print(sys.executable)"
```

When multiple Python installations are available, invoke `pip` through the
intended interpreter:

```bash
python -m pip install scatlaspy
```

This ensures that scAtlasPy is installed for that Python interpreter.

### No matching distribution is available

Check the Python version:

```bash
python --version
```

scAtlasPy requires Python 3.10 or later. Updating `pip` may also resolve package
discovery or compatibility problems:

```bash
pip install --upgrade pip
pip install scatlaspy
```

Before the first public PyPI release, install scAtlasPy directly from GitHub:

```bash
pip install "scatlaspy @ git+https://github.com/GaoLab-XDU/scAtlasPy.git"
```

## Next Steps

After installing scAtlasPy:

- Follow the {doc}`tutorials/index` to run a complete analysis workflow.
- Prepare an atlas from your own data with
  {doc}`tutorials/basic/import-data-from-multiple-formats`.
- Learn how scAtlasPy organizes and accesses atlas-scale data in
  {doc}`architecture-and-development/data-model`.
- Explore the {doc}`api/index` for available classes and analysis functions.
