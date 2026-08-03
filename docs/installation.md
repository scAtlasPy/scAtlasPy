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

The standard installation includes the dependencies required to create an
Atlas, import and export data, run database-backed preprocessing, build read
indexes, compute streaming PCA, read minibatches, and use plotting functions.

PyTorch is required for the distilled UMAP and distilled Louvain backends used
by:

- `sap.tl.umap(...)`;
- `sap.tl.graph_clustering(...)`.

PyTorch is kept optional because CPU, CUDA, and Apple Silicon MPS builds are
hardware-specific. For a generic CPU installation, use the scAtlasPy optional
extra:

```bash
pip install "scatlaspy[parametric]"
```

For GPU acceleration, install the PyTorch build recommended for your hardware
and driver stack, then install scAtlasPy in the same environment:

```bash
pip install scatlaspy
```

After PyTorch is installed, scAtlasPy can use devices such as `cuda:0`, `mps`,
or `cpu` through the `device` parameter of the relevant tool.

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

## Next Steps

After installing scAtlasPy:

- Follow the {doc}`tutorials/index` to run a complete analysis workflow.
- Prepare an atlas from your own data with
  {doc}`tutorials/basic/import-data-from-multiple-formats`.
- Learn how scAtlasPy organizes and accesses atlas-scale data in
  {doc}`architecture-and-development/data-model`.
- Explore the {doc}`api/index` for available classes and analysis functions.
