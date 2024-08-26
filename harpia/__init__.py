# from .__version__ import __version__
from . import _version
from .filters import *
from .morphology import *
from .quantification import *
from .threshold import *

__version__ = _version.get_versions()["version"]
