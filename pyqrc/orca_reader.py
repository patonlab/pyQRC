"""
Minimal reader for ORCA frequency outputs.

pyQRC reads outputs with cclib. Released cclib versions (up to 1.8.1) fail on
ORCA 6 outputs, so this reader is used as a fallback for ORCA files. It reads
only what pyQRC needs: the final geometry, charge, multiplicity, harmonic
frequencies and normal modes.
"""

import re
from types import SimpleNamespace
from typing import Optional

import numpy as np
from cclib.parser.utils import PeriodicTable

ORCA_BANNER = '* O   R   C   A *'

_ATOMIC_NUMBERS = {symbol.lower(): z for symbol, z in PeriodicTable().number.items() if z > 0}


class ORCAReadError(Exception):
    """Raised when an ORCA output lacks the data pyQRC needs."""


def is_orca_output(file: str) -> bool:
    """Return True if the file looks like an ORCA output.

    Args:
        file: Path to the output file.
    """
    with open(file, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if ORCA_BANNER in line:
                return True
            if i > 200:
                return False
    return False


def _last_index(lines: list[str], text: str, before: Optional[int] = None) -> Optional[int]:
    """Index of the last line containing text (optionally before a line index)."""
    end = len(lines) if before is None else before
    for i in range(end - 1, -1, -1):
        if text in lines[i]:
            return i
    return None


def _read_geometry(lines: list[str], before: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the last Cartesian geometry (Angstrom) printed before a line index."""
    start = _last_index(lines, 'CARTESIAN COORDINATES (ANGSTROEM)', before)
    if start is None:
        raise ORCAReadError('no Cartesian coordinates found')
    atomnos, coords = [], []
    for line in lines[start + 2:]:
        fields = line.split()
        if len(fields) != 4:
            break
        symbol = re.match(r'[A-Za-z]+', fields[0])
        if symbol is None or symbol[0].lower() not in _ATOMIC_NUMBERS:
            raise ORCAReadError(f'unknown element in geometry: {fields[0]}')
        atomnos.append(_ATOMIC_NUMBERS[symbol[0].lower()])
        coords.append([float(x) for x in fields[1:]])
    if not atomnos:
        raise ORCAReadError('empty Cartesian coordinate block')
    return np.array(atomnos), np.array(coords)


def _read_charge_mult(lines: list[str]) -> tuple[int, int]:
    """Read the last total charge and multiplicity."""
    charge = mult = None
    for line in lines:
        match = re.match(r'\s*Total Charge\s+Charge\s+\.+\s+(-?\d+)', line)
        if match:
            charge = int(match[1])
        match = re.match(r'\s*Multiplicity\s+Mult\s+\.+\s+(\d+)', line)
        if match:
            mult = int(match[1])
    if charge is None or mult is None:
        # Semi-empirical (xTB) jobs do not print them; use the echoed input
        for line in lines:
            match = re.match(r'\|\s*\d+>\s*\*\s*(?:xyz|xyzfile|int|internal|gzmt|gzmtfile|pdbfile)'
                             r'\s+(-?\d+)\s+(\d+)', line, re.IGNORECASE)
            if match:
                charge, mult = int(match[1]), int(match[2])
                break
    if charge is None or mult is None:
        raise ORCAReadError('charge or multiplicity not found')
    return charge, mult


def _read_frequencies(lines: list[str], start: int) -> list[float]:
    """Read all 3N frequencies (cm-1) from a VIBRATIONAL FREQUENCIES block."""
    freqs = []
    for line in lines[start + 1:]:
        match = re.match(r'\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1', line)
        if match:
            if int(match[1]) != len(freqs):
                raise ORCAReadError('frequencies are not numbered consecutively')
            freqs.append(float(match[2]))
        elif freqs and line.strip():
            break
    return freqs


def _read_normal_modes(lines: list[str], start: int, ncoords: int) -> np.ndarray:
    """Read the ncoords x ncoords normal-mode matrix; column j is mode j."""
    modes = np.full((ncoords, ncoords), np.nan)
    columns: list[int] = []
    for line in lines[start + 1:]:
        fields = line.split()
        if not fields:
            continue
        if 'IR SPECTRUM' in line or (fields[0].startswith('---') and columns):
            break
        if not all(re.fullmatch(r'-?\d+(\.\d+)?', field) for field in fields):
            continue
        if all(field.isdigit() for field in fields):
            columns = [int(field) for field in fields]
        elif columns and fields[0].isdigit():
            row = int(fields[0])
            values = [float(x) for x in fields[1:]]
            if len(values) != len(columns) or row >= ncoords:
                raise ORCAReadError('malformed normal mode block')
            for col, value in zip(columns, values):
                modes[row, col] = value
    if np.isnan(modes).any():
        raise ORCAReadError('incomplete normal mode block')
    return modes


def _is_linear(coords: np.ndarray, tol: float = 1e-3) -> bool:
    """True if all atoms lie on a line (within tol Angstrom)."""
    if len(coords) < 3:
        return True
    centered = coords - coords.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return bool(singular[1] < tol * np.sqrt(len(coords)))


def _count_translations_rotations(coords: np.ndarray, freqs: list[float]) -> int:
    """Number of leading translation/rotation modes: 3 (atom), 5 (linear) or 6."""
    if len(coords) == 1:
        return 3
    expected = 5 if _is_linear(coords) else 6
    # ORCA projects them out and prints exactly 0.00; trust that count if it
    # disagrees with the geometry test (e.g. a nearly linear molecule)
    zeros = 0
    for freq in freqs:
        if freq != 0.0:
            break
        zeros += 1
    if zeros in (5, 6) and zeros != expected:
        return zeros
    return expected


def read_orca_frequencies(file: str) -> SimpleNamespace:
    """Read an ORCA frequency calculation.

    The returned object has the attributes of a cclib data object that
    pyQRC uses: natom, atomnos, atomcoords, charge, mult, vibfreqs,
    vibdisps and metadata. Translations and rotations (the first 6 modes,
    5 for a linear molecule, which ORCA prints as 0.00 cm-1) are dropped,
    as cclib does. Imaginary modes are always kept.

    Args:
        file: Path to the ORCA output.

    Returns:
        Parsed data.

    Raises:
        ORCAReadError: If the output does not contain a complete frequency
            calculation.
    """
    with open(file, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    freq_start = _last_index(lines, 'VIBRATIONAL FREQUENCIES')
    modes_start = _last_index(lines, 'NORMAL MODES')
    if freq_start is None or modes_start is None or modes_start < freq_start:
        raise ORCAReadError('no frequency calculation found')

    atomnos, coords = _read_geometry(lines, freq_start)
    charge, mult = _read_charge_mult(lines[:freq_start])
    ncoords = 3 * len(atomnos)
    freqs = _read_frequencies(lines, freq_start)
    if len(freqs) != ncoords:
        raise ORCAReadError(f'expected {ncoords} frequencies, found {len(freqs)}')
    modes = _read_normal_modes(lines, modes_start, ncoords)

    # Not ORCA's "first frequency considered to be a vibration": that line
    # is for thermochemistry and also skips the imaginary modes
    first_vib = _count_translations_rotations(coords, freqs)

    version = None
    for line in lines[:200]:
        match = re.search(r'Program Version\s+(\S+)', line)
        if match:
            version = match[1]
            break

    natom = len(atomnos)
    vibdisps = modes[:, first_vib:].T.reshape(-1, natom, 3)
    return SimpleNamespace(
        natom=natom,
        atomnos=atomnos,
        atomcoords=np.array([coords]),
        charge=charge,
        mult=mult,
        vibfreqs=np.array(freqs[first_vib:]),
        vibdisps=vibdisps,
        metadata={'package': 'ORCA', 'package_version': version},
    )
