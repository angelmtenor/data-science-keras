""" Python Package init file """
# read version from installed package
from importlib.metadata import version

__version__ = version("ds_boost")

from .ds_boost import *  # noqa
