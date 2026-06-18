from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent / "src"
sys.path.insert(0, str(PACKAGE_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

project = "scAtlasPy"
author = "scAtlasPy development team"
copyright = f"{datetime.now():%Y}, scAtlasPy development team"
release = "0.1.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

try:
    import scatlaspy as _sap

    sys.modules.setdefault("pp", _sap.pp)
    sys.modules.setdefault("tl", _sap.tl)
    sys.modules.setdefault("pl", _sap.pl)
except Exception:
    pass

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "index_structure.md"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

html_theme = "sphinx_book_theme"
html_title = "A scalable Python platform for atlas-scale single-cell omics analysis beyond in-memory limits"
html_logo = "_static/img/scAtlas_full_paths.svg"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/scAtlasAnalysis/scAtlas",
    "use_repository_button": True,
    "use_issues_button": True,
    "show_navbar_depth": 2,
    "logo": {
        "image_light": "_static/img/scAtlas_full_paths.svg",
        "image_dark": "_static/img/scAtlas_logo_BrightFG.svg",
        "alt_text": "scAtlasPy",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

autoclass_content = "class"


def setup(app):
    pass
