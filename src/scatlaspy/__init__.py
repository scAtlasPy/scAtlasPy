from importlib.metadata import PackageNotFoundError, version

from . import data
from . import preprocessing as pp
from . import tools as tl
from . import plots as pl

from .data import Atlas, set_verbosity
from .io import set_progress

try:
    __version__ = version("scatlaspy")
except PackageNotFoundError:
    __version__ = "unknown"
