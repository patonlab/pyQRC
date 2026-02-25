"""
pyQRC - Quick Reaction Coordinate calculations.

A Python implementation of Silva and Goodman's QRC approach for
displacing molecular structures along normal modes.
"""

from pyqrc.pyQRC import (
    QRCGenerator,
    QRCParseError,
    OutputData,
    Logger,
    mwdist,
    check_overlap,
    element_id,
    PERIODIC_TABLE,
    ATOMIC_MASSES,
    COVALENT_RADII,
)

__version__ = "2.1.0"
__author__ = "Robert Paton"
__email__ = "robert.paton@colostate.edu"

__all__ = [
    "QRCGenerator",
    "QRCParseError",
    "OutputData",
    "Logger",
    "mwdist",
    "check_overlap",
    "element_id",
    "PERIODIC_TABLE",
    "ATOMIC_MASSES",
    "COVALENT_RADII",
    "__version__",
]
