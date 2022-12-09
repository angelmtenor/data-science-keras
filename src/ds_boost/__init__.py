""" Read version from installed package and import all modules """
from importlib.metadata import version

__version__ = version(__name__)

from .ds_boost import *  # noqa
