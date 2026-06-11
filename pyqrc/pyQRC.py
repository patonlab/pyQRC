#!/usr/bin/python
"""
pyQRC.py - A Python implementation of Silva and Goodman's QRC approach.

From a computational chemistry frequency calculation, it is possible to create
new input files which are displaced along any normal modes which have an
imaginary frequency.

Based on: Goodman, J. M.; Silva, M. A. Tet. Lett. 2003, 44, 8233-8236;
          Tet. Lett. 2005, 46, 2067-2069.
"""

__version__ = '2.3.0'
__author__ = 'Robert Paton'
__email__ = 'robert.paton@colostate.edu'

import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from typing import Optional, Generator

import cclib
import numpy as np


class QRCParseError(Exception):
    """Raised when parsing a computational chemistry output file fails."""


class QRCModeError(Exception):
    """Raised when a requested normal mode or frequency cannot be matched."""


@contextmanager
def working_directory(path: Path) -> Generator[None, None, None]:
    """Context manager for changing working directory with automatic restoration.

    Args:
        path: Directory to change to.

    Yields:
        None
    """
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

# Constants
ANGSTROM_TO_BOHR = 1.88972612456506
DEFAULT_AMPLITUDE = 0.2
FREQ_MATCH_TOLERANCE = 1.0  # cm-1, for matching a user-requested --freq value
DEFAULT_NPROC = 1
DEFAULT_MEMORY = "4GB"
DEFAULT_SUFFIX = "QRC"

# Periodic table of elements (index = atomic number)
PERIODIC_TABLE = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh",
    "Uus", "Uuo"
]

# Atomic masses (index = atomic number)
ATOMIC_MASSES = [
    0.0, 1.007825, 4.00, 7.00, 9.00, 11.00, 12.01, 14.0067, 15.9994, 19.00,
    20.180, 22.990, 24.305, 26.982, 28.086, 30.973762, 31.972071, 35.453,
    39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845,
    58.933, 58.693, 63.546, 65.38, 69.723, 72.631, 74.922, 78.971, 79.904,
    84.798, 84.468, 87.62, 88.906, 91.224, 92.906, 95.95, 98.907, 101.07,
    102.906, 106.42, 107.868, 112.414, 114.818, 118.711, 121.760, 126.7,
    126.904, 131.294, 132.905, 137.328, 138.905, 140.116, 140.908, 144.243,
    144.913, 150.36, 151.964, 157.25, 158.925, 162.500, 164.930, 167.259,
    168.934, 173.055, 174.967, 178.49, 180.948, 183.84, 186.207, 190.23,
    192.217, 195.085, 196.967, 200.592, 204.383, 207.2, 208.980, 208.982,
    209.987, 222.081, 223.020, 226.025, 227.028, 232.038, 231.036, 238.029,
    237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 262, 261, 262, 266, 264,
    269, 268, 271, 272, 285, 284, 289, 288, 292, 294, 294
]

# Covalent radii (Angstroms)
COVALENT_RADII = {
    "H": 0.32, "He": 0.46, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75,
    "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, "Na": 1.55, "Mg": 1.39,
    "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96,
    "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22,
    "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Zn": 1.18, "Ga": 1.24,
    "Ge": 1.21, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, "Rb": 2.10,
    "Sr": 1.85, "Y": 1.63, "Zr": 1.54, "Nb": 1.47, "Mo": 1.38, "Tc": 1.28,
    "Ru": 1.25, "Rh": 1.25, "Pd": 1.20, "Ag": 1.28, "Cd": 1.36, "In": 1.42,
    "Sn": 1.40, "Sb": 1.40, "Te": 1.36, "I": 1.33, "Xe": 1.31
}


def element_id(mass_no: int) -> str:
    """Convert atomic number to element symbol.

    Args:
        mass_no: Atomic number (Z).

    Returns:
        Element symbol string, or "XX" if atomic number is out of range.
    """
    if mass_no < len(PERIODIC_TABLE):
        return PERIODIC_TABLE[mass_no]
    return "XX"


class Logger:
    """File logger for writing QRC output files.

    Supports context manager protocol for proper file handling.
    """

    def __init__(self, filein: str, suffix: str, append: str):
        """Initialize logger with output file.

        Args:
            filein: Base filename (without extension).
            suffix: File extension.
            append: String to append to filename before extension.
        """
        self.filepath = f"{filein}_{append}.{suffix}"
        # pylint: disable=consider-using-with
        self.log = open(self.filepath, 'w', encoding='utf-8')

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
        return False

    def write(self, message: str) -> None:
        """Write a message to the log file.

        Args:
            message: Text to write (newline appended automatically).
        """
        self.log.write(message + "\n")

    def close(self) -> None:
        """Close the log file."""
        if self.log and not self.log.closed:
            self.log.close()


class OutputData:
    """Parse computational chemistry output files for job metadata.

    Extracts format, job type, level of theory, and termination status
    from Gaussian, ORCA, and Q-Chem output files.
    """

    def __init__(self, file: str):
        """Parse output file for metadata.

        Args:
            file: Path to output file.

        Raises:
            FileNotFoundError: If the output file does not exist.
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"Output file [{file}] does not exist")

        self.file = file
        self.format: Optional[str] = None
        self.JOBTYPE: Optional[str] = None
        self.LEVELOFTHEORY: Optional[str] = None
        self.TERMINATION: Optional[str] = None

        with open(file, 'r', encoding='utf-8') as f:
            outlines = f.readlines()

        self._get_format(outlines)
        self._get_jobtype(outlines)
        self._get_termination(outlines)

    def _get_format(self, outlines: list[str]) -> None:
        """Detect the computational chemistry package from output.

        Args:
            outlines: List of lines from output file.
        """
        for line in outlines:
            if 'Gaussian, Inc.' in line:
                self.format = "Gaussian"
                return
            if '* O   R   C   A *' in line:
                self.format = "ORCA"
                return
            if 'Q-Chem, Inc.' in line:
                self.format = "QChem"
                return

    def _level_of_theory(self) -> str:
        """Read output for the level of theory and basis set used.

        Returns:
            String in format "level/basis_set".
        """
        repeated_theory = 0
        with open(self.file, encoding='utf-8') as f:
            data = f.readlines()
        level, bs = 'none', 'none'

        for line in data:
            if 'External calculation' in line.strip():
                level, bs = 'ext', 'ext'
                break
            if '\\Freq\\' in line.strip() and repeated_theory == 0:
                try:
                    level, bs = line.strip().split("\\")[4:6]
                    repeated_theory = 1
                except IndexError:
                    pass
            elif '|Freq|' in line.strip() and repeated_theory == 0:
                try:
                    level, bs = line.strip().split("|")[4:6]
                    repeated_theory = 1
                except IndexError:
                    pass
            if '\\SP\\' in line.strip() and repeated_theory == 0:
                try:
                    level, bs = line.strip().split("\\")[4:6]
                    repeated_theory = 1
                except IndexError:
                    pass
            elif '|SP|' in line.strip() and repeated_theory == 0:
                try:
                    level, bs = line.strip().split("|")[4:6]
                    repeated_theory = 1
                except IndexError:
                    pass
            if 'DLPNO BASED TRIPLES CORRECTION' in line.strip():
                level = 'DLPNO-CCSD(T)'
            if 'Estimated CBS total energy' in line.strip():
                try:
                    bs = "Extrapol." + line.strip().split()[4]
                except IndexError:
                    pass
            # Remove the restricted R or unrestricted U label
            if level != 'none' and level[0] in ('R', 'U'):
                level = level[1:]

        return f"{level}/{bs}"

    def _get_jobtype(self, outlines: list[str]) -> None:
        """Extract job type and level of theory from output.

        Args:
            outlines: List of lines from output file.
        """
        if self.format == "Gaussian":
            for i, line in enumerate(outlines):
                if line.strip().find('----------') > -1:
                    if outlines[i + 1].strip().find('#') > -1:
                        self.JOBTYPE = ''
                        for j in range(i + 1, len(outlines)):
                            if outlines[j].strip().find('----------') > -1:
                                break
                            self.JOBTYPE += re.sub('#', '', outlines[j].strip())
                        self.JOBTYPE = re.sub(r' geom=\S+', '', self.JOBTYPE)
                        self.LEVELOFTHEORY = self._level_of_theory()
                        break

        elif self.format == "ORCA":
            for line in outlines:
                if '> !' in line.strip():
                    self.JOBTYPE = line.strip().split('> !')[1].lstrip()
                    self.LEVELOFTHEORY = self._level_of_theory()
                    break

    def _get_termination(self, outlines: list[str]) -> None:
        """Check if calculation terminated normally.

        Args:
            outlines: List of lines from output file.
        """
        if self.format == "Gaussian":
            for line in outlines:
                if "Normal termination" in line:
                    self.TERMINATION = "normal"
                    return


def mwdist(coords1: np.ndarray, coords2: np.ndarray, elements: list[int]) -> float:
    """Compute mass-weighted Cartesian displacement between two structures.

    Args:
        coords1: First set of atomic coordinates (N x 3 array).
        coords2: Second set of atomic coordinates (N x 3 array).
        elements: List of atomic numbers for each atom.

    Returns:
        Mass-weighted distance in bohr * amu^(1/2).
    """
    dist = 0.0
    for n, atom in enumerate(elements):
        dist += ATOMIC_MASSES[atom] * (np.linalg.norm(coords1[n] - coords2[n])) ** 2

    return ANGSTROM_TO_BOHR * dist ** 0.5


def gen_overlap(mol_atoms: list[str], coords: np.ndarray, covfrac: float) -> np.ndarray:
    """Generate overlap matrix based on covalent radii.

    Args:
        mol_atoms: List of element symbols.
        coords: Atomic coordinates (N x 3 array).
        covfrac: Fraction of covalent radii sum to use as overlap threshold.

    Returns:
        Upper triangular matrix where 1 indicates overlapping atoms.
    """
    n_atoms = len(mol_atoms)
    over_mat = np.zeros((n_atoms, n_atoms))

    for i, atom_i in enumerate(mol_atoms):
        for j, atom_j in enumerate(mol_atoms):
            if j > i:
                rcov_ij = COVALENT_RADII.get(atom_i, 1.5) + COVALENT_RADII.get(atom_j, 1.5)
                dist_ij = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                if dist_ij / rcov_ij < covfrac:
                    over_mat[i][j] = 1

    return over_mat


def check_overlap(atom_types: list[str], coords: np.ndarray, covfrac: float = 0.8) -> bool:
    """Check if any atoms overlap based on covalent radii.

    Args:
        atom_types: List of element symbols.
        coords: Atomic coordinates (N x 3 array).
        covfrac: Fraction of covalent radii sum to use as overlap threshold.

    Returns:
        True if any atoms overlap, False otherwise.
    """
    over_mat = gen_overlap(atom_types, coords, covfrac)
    return bool(np.any(over_mat))


class QRCGenerator:
    """Generate Quick Reaction Coordinate displaced structures.

    Takes a frequency calculation output and generates new input files
    with geometries displaced along specified normal modes.
    """

    def __init__(
        self,
        file: str,
        amplitude: float,
        nproc: int,
        mem: str,
        route: Optional[str],
        verbose: bool,
        suffix: str,
        val: Optional[float],
        num: Optional[int],
        write: bool = True
    ):
        """Initialize QRC generator and create displaced structure.

        Args:
            file: Path to frequency calculation output file.
            amplitude: Displacement amplitude along normal modes.
            nproc: Number of processors for new calculation.
            mem: Memory allocation string (e.g., "4GB").
            route: Calculation route/keywords (None to clone from input).
            verbose: Whether to write detailed output file.
            suffix: Suffix to append to output filename.
            val: Specific frequency value (cm^-1) to displace along.
            num: Specific mode number to displace along (1-indexed).
            write: Whether to write output files (False to only compute
                the displaced geometry, e.g. for library use).
        """
        self.file = file
        self.amplitude = amplitude
        self.nproc = nproc
        self.mem = mem
        self.route = route
        self.verbose = verbose
        self.suffix = suffix
        self.val = val
        self.num = num

        self._parse()
        self.compute_displacement()
        if write:
            self.write_files()

    def _parse(self) -> None:
        """Parse the output file with cclib and store molecular data.

        Raises:
            QRCParseError: If the file format cannot be determined or
                parsing fails.
        """
        try:
            parser = cclib.io.ccopen(self.file)
            if parser is None:
                raise QRCParseError(
                    f"Could not determine file format for '{self.file}'. "
                    "Ensure it is a valid Gaussian, ORCA, or Q-Chem output file."
                )
            data = parser.parse()
        except Exception as exc:
            if isinstance(exc, QRCParseError):
                raise
            raise QRCParseError(
                f"Failed to parse '{self.file}': {exc}. "
                "The file may be corrupted or from an unsupported format."
            ) from exc

        self.NATOMS = data.natom
        self.CHARGE = data.charge
        self.ATOMNOS = data.atomnos

        try:
            self.MULT = data.mult
        except AttributeError:
            self.MULT = 1
            print('Warning - multiplicity not parsed from input: defaulted to 1 in input files')

        self.ATOMTYPES = [PERIODIC_TABLE[z] for z in self.ATOMNOS]
        self.CARTESIAN = data.atomcoords[-1].copy()
        self.FREQS = data.vibfreqs
        self.DISPS = data.vibdisps

        nmodes = len(self.FREQS)
        self.RMASS = data.vibrmasses if hasattr(data, 'vibrmasses') else [0.0] * nmodes
        self.FCONST = data.vibfconsts if hasattr(data, 'vibfconsts') else [0.0] * nmodes

        # Output format and job metadata from cclib, when available
        self._format_type = None
        self._func = None
        self._basis = None
        if hasattr(data, 'metadata'):
            self._format_type = data.metadata.get('package')
            self._func = data.metadata.get('functional')
            self._basis = data.metadata.get('basis_set')

    def compute_displacement(self) -> np.ndarray:
        """Compute the displaced geometry along the resolved normal modes.

        Pure computation: parsed data is never mutated and no files are
        written. Results are stored on the instance (NEW_CARTESIAN,
        MW_DISTANCE, OVERLAPPED).

        Returns:
            The displaced coordinates (N x 3 array).

        Raises:
            QRCModeError: If a specifically requested mode cannot be matched.
        """
        nmodes = len(self.FREQS)
        self._target_modes = self._resolve_target_modes(self.FREQS, self.val, self.num)

        new_cartesian = self.CARTESIAN.copy()
        for mode in self._target_modes:
            for atom in range(self.NATOMS):
                for coord in range(3):
                    new_cartesian[atom][coord] += self.DISPS[mode][atom][coord] * self.amplitude

        self.NEW_CARTESIAN = new_cartesian
        self.MW_DISTANCE = mwdist(self.NEW_CARTESIAN, self.CARTESIAN, self.ATOMNOS)
        self.OVERLAPPED = check_overlap(self.ATOMTYPES, self.NEW_CARTESIAN)
        return self.NEW_CARTESIAN

    def write_files(self) -> None:
        """Write the verbose .qrc summary (if verbose) and the new input file.

        Raises:
            QRCParseError: If a Q-Chem input is requested but the method or
                basis set could not be parsed from the output. Raised before
                any file is written.
        """
        file_path = Path(self.file)
        nat = self.NATOMS

        # Resolve output format and route, preferring cclib metadata
        format_type = self._format_type
        route = self.route
        if format_type is None or route is None:
            gdata = OutputData(self.file)
            if format_type is None:
                format_type = gdata.format
            if route is None:
                route = gdata.JOBTYPE
            if gdata.format == "Gaussian" and gdata.TERMINATION != "normal":
                print(
                    f'Warning - {self.file} did not terminate normally: '
                    'the parsed geometry and normal modes may be incomplete'
                )

        if format_type == "QChem" and (self._func is None or self._basis is None):
            raise QRCParseError(
                f"Cannot write a Q-Chem input for '{self.file}': the method "
                "and/or basis set could not be parsed from the output, so the "
                "input would contain 'METHOD None'. Check that the output file "
                "is complete."
            )

        if self.verbose:
            with Logger(file_path.stem, "qrc", self.suffix) as log:
                self._write_header(
                    log, self.ATOMTYPES, self.CARTESIAN, nat,
                    self.FREQS, self.RMASS, self.FCONST, len(self.FREQS)
                )
                for mode in sorted(self._target_modes):
                    log.write('\n                -SHIFTING ALONG NORMAL MODE-')
                    log.write(f'                -MODE {mode + 1}: {self.FREQS[mode]:.1f} cm-1')
                    log.write(f'                -AMPLIFIER = {self.amplitude}')
                    log.write(f'{"":>4} {"":>9} {"X":>9} {"Y":>9} {"Z":>9}')
                    for atom in range(nat):
                        log.write(
                            f'{self.ATOMTYPES[atom]:>4} {"":>9} '
                            f'{self.DISPS[mode][atom][0]:9.6f} '
                            f'{self.DISPS[mode][atom][1]:9.6f} '
                            f'{self.DISPS[mode][atom][2]:9.6f}'
                        )
                log.write(f'\n   STRUCTURE MOVED BY {self.MW_DISTANCE:.3f} Bohr amu^1/2 \n')

        self._write_input_file(
            file_path, format_type, self.suffix, self.nproc, self.mem, route,
            self.CHARGE, self.MULT, self.ATOMTYPES, self.NEW_CARTESIAN, nat,
            self._func, self._basis
        )

    def _write_header(
        self,
        log: Logger,
        elements: list[str],
        cartesians: np.ndarray,
        nat: int,
        freq: np.ndarray,
        rmass: list[float],
        fconst: list[float],
        nmodes: int
    ) -> None:
        """Write header information to verbose log file."""
        log.write(' pyQRC - a quick alternative to IRC calculations')
        log.write(f' version: {__version__} / author: {__author__} / email: {__email__}')
        log.write(' Based on: Goodman, J. M.; Silva, M. A. Tet. Lett. 2003, 44, 8233-8236;')
        log.write(' Tet. Lett. 2005, 46, 2067-2069.\n')
        log.write('                -----ORIGINAL GEOMETRY------')
        log.write(f'{"":>4} {"":>9} {"X":>9} {"Y":>9} {"Z":>9}')

        for atom in range(nat):
            x, y, z = cartesians[atom]
            log.write(f'{elements[atom]:>4} {"":>9} {x:9.6f} {y:9.6f} {z:9.6f}')

        log.write('\n                ----HARMONIC FREQUENCIES----')
        log.write(f'{"Freq":>24} {"Red mass":>9} {"F const":>9}')

        for mode in range(nmodes):
            log.write(f'{freq[mode]:24.4f} {rmass[mode]:9.4f} {fconst[mode]:9.4f}')

    @staticmethod
    def _resolve_target_modes(
        freq: np.ndarray,
        val: Optional[float],
        num: Optional[int]
    ) -> set:
        """Resolve which normal modes to displace along.

        Args:
            freq: Vibrational frequencies (cm^-1).
            val: Specific frequency value (cm^-1) requested by the user, if any.
            num: Specific mode number (1-indexed) requested by the user, if any.

        Returns:
            Set of 0-indexed mode numbers to shift. Defaults to all
            imaginary modes when no specific mode was requested.

        Raises:
            QRCModeError: If a specifically requested mode cannot be matched.
        """
        nmodes = len(freq)

        if num is not None:
            if not 1 <= num <= nmodes:
                raise QRCModeError(
                    f"requested mode {num} is out of range: file has {nmodes} normal modes"
                )
            return {num - 1}

        if val is not None:
            if nmodes == 0:
                raise QRCModeError(
                    f"requested frequency {val} cm-1 cannot be matched: no vibrational modes are available"
                )
            nearest = min(range(nmodes), key=lambda mode: abs(freq[mode] - val))
            if abs(freq[nearest] - val) > FREQ_MATCH_TOLERANCE:
                raise QRCModeError(
                    f"no normal mode within {FREQ_MATCH_TOLERANCE} cm-1 of {val} cm-1 "
                    f"(nearest: {freq[nearest]:.4f} cm-1)"
                )
            return {nearest}

        return {mode for mode, wn in enumerate(freq) if wn < 0.0}

    def _write_input_file(
        self,
        file_path: Path,
        format_type: str,
        suffix: str,
        nproc: int,
        mem: str,
        route: str,
        charge: int,
        mult: int,
        elements: list[str],
        cartesians: np.ndarray,
        nat: int,
        func: Optional[str],
        basis: Optional[str]
    ) -> None:
        """Write new computational chemistry input file."""
        if format_type == "Gaussian":
            input_ext = "com"
        elif format_type in ("ORCA", "QChem"):
            input_ext = "inp"
        else:
            input_ext = "com"

        with Logger(file_path.stem, input_ext, suffix) as new_input:
            if format_type == "Gaussian":
                new_input.write(f'%chk={file_path.stem}_{suffix}.chk')
                new_input.write(f'%nproc={nproc}\n%mem={mem}\n#{route}')
                new_input.write(f'\n{file_path.stem}_{suffix}\n\n{charge} {mult}')

            elif format_type == "ORCA":
                # Parse memory string for ORCA
                memory_number = re.findall(r'\d+', mem)
                unit = re.findall(r'GB', mem)
                if unit:
                    mem_val = int(memory_number[0]) * 1024
                else:
                    mem_val = memory_number[0]

                new_input.write(
                    f'! {route}\n %pal nprocs {nproc} end\n %maxcore {mem_val}\n\n'
                    f'# {file_path.stem}_{suffix}\n\n* xyz {charge} {mult}'
                )

            elif format_type == "QChem":
                new_input.write(f'$molecule\n{charge} {mult}')

            # Write coordinates
            for atom in range(nat):
                new_input.write(
                    f'{elements[atom]:>2} {cartesians[atom][0]:12.8f} '
                    f'{cartesians[atom][1]:12.8f} {cartesians[atom][2]:12.8f}'
                )

            # Write format-specific footer
            if format_type == "Gaussian":
                new_input.write("")
            elif format_type == "ORCA":
                new_input.write("*")
            elif format_type == "QChem":
                new_input.write("$end\n\n$rem")
                new_input.write(f"   JOBTYPE opt\n   METHOD {func}\n   BASIS {basis}")
                new_input.write("$end\n\n@@@\n\n$molecule\n   read\n$end\n\n$rem")
                new_input.write(f"   JOBTYPE freq\n   METHOD {func}\n   BASIS {basis}")
                new_input.write("$end\n")


def g16_opt(comfile: str) -> None:
    """Run Gaussian 16 optimization using shell script.

    Args:
        comfile: Path to Gaussian input file.
    """
    script_dir = Path(__file__).parent.absolute()
    command = [str(script_dir / 'run_g16.sh'), str(comfile)]
    subprocess.run(command, check=False)


def run_irc(
    file: str,
    options: Namespace,
    num: int,
    amp: float,
    lot_bs: str,
    suffix: str,
    log_output: Logger
) -> None:
    """Run IRC-like displacement calculation.

    Args:
        file: Input file path.
        options: Command-line options namespace.
        num: Mode number to displace along.
        amp: Displacement amplitude.
        lot_bs: Level of theory and basis set.
        suffix: Output file suffix.
        log_output: Logger for output messages.
    """
    file_path = Path(file)
    qrc = QRCGenerator(
        file, amp, options.nproc, options.mem, lot_bs,
        options.verbose, suffix, None, num
    )

    if not qrc.OVERLAPPED:
        g16_opt(f'{file_path.stem}_{suffix}.com')
    else:
        log_output.write(f'x  Skipping {file_path.stem}_{suffix}.com due to overlap in atoms')


def main() -> int:
    """Main entry point for pyQRC command-line interface.

    Returns:
        Exit code: 0 for success, 1 if any files failed to process.
    """
    parser = ArgumentParser(
        description="pyQRC - a quick alternative to IRC calculations",
        usage="%(prog)s [options] <input1>.log <input2>.log ..."
    )

    parser.add_argument(
        "files", nargs="*", metavar="FILE",
        help="Output files to process (.out or .log)"
    )
    parser.add_argument(
        "--amp", dest="amplitude", type=float, default=DEFAULT_AMPLITUDE,
        metavar="AMPLITUDE", help=f"amplitude (default {DEFAULT_AMPLITUDE})"
    )
    parser.add_argument(
        "--nproc", dest="nproc", type=int, default=DEFAULT_NPROC,
        metavar="NPROC", help=f"number of processors (default {DEFAULT_NPROC})"
    )
    parser.add_argument(
        "--mem", dest="mem", type=str, default=DEFAULT_MEMORY,
        metavar="MEM", help=f"memory (default {DEFAULT_MEMORY})"
    )
    parser.add_argument(
        "--route", dest="route", type=str, default=None,
        metavar="ROUTE", help="calculation route (defaults to same as original file)"
    )
    parser.add_argument(
        "-q", "--quiet", dest="verbose", action="store_false", default=True,
        help="suppress verbose output"
    )
    parser.add_argument(
        "--auto", dest="auto", action="store_true", default=False,
        help="turn on automatic batch processing"
    )
    parser.add_argument(
        "--name", dest="suffix", type=str, default=DEFAULT_SUFFIX,
        metavar="SUFFIX", help=f"append to file name (defaults to {DEFAULT_SUFFIX})"
    )
    parser.add_argument(
        "-f", "--freq", dest="freq", type=float, default=None,
        metavar="FREQ", help="request motion along a particular frequency (cm-1)"
    )
    parser.add_argument(
        "--freqnum", dest="freqnum", type=int, default=None,
        metavar="FREQNUM", help="request motion along a particular frequency (number)"
    )
    parser.add_argument(
        "--qcoord", dest="qcoord", action="store_true", default=False,
        help="request automatic single point calculation along a particular normal mode"
    )
    parser.add_argument(
        "--nummodes", dest="nummodes", type=str, default='all',
        metavar="NUMMODES", help="number of modes for automatic single point calculation"
    )

    args = parser.parse_args()

    # Collect input files, expanding any glob patterns the shell left unexpanded
    files = []
    exit_code = 0
    for pattern in args.files:
        matches = sorted(glob(pattern))
        if not matches:
            print(f'x   {pattern}: no such file')
            exit_code = 1
            continue
        for file in matches:
            if os.path.splitext(file)[1] not in ('.out', '.log'):
                print(f'x   {file} skipped: expected a .log or .out file')
                exit_code = 1
                continue
            files.append(file)
    for file in files:
        # Parse output with cclib and count imaginary frequencies
        try:
            parser_cc = cclib.io.ccopen(file)
            if parser_cc is None:
                print(f'x   {file} could not be parsed (unknown format): skipping')
                exit_code = 1
                continue
            data = parser_cc.parse()
        except Exception as exc:
            print(f'x   {file} failed to parse: {exc}')
            exit_code = 1
            continue

        if hasattr(data, 'vibfreqs'):
            im_freq = len([val for val in data.vibfreqs if val < 0])
        else:
            print(f'x   {file} has no frequency information: skipping')
            continue

        if not args.qcoord:
            if im_freq == 0 and args.auto:
                print(f'x   {file} has no imaginary frequencies: skipping')
            else:
                try:
                    QRCGenerator(
                        file, args.amplitude, args.nproc, args.mem, args.route,
                        args.verbose, args.suffix, args.freq, args.freqnum
                    )
                except (QRCParseError, QRCModeError) as exc:
                    print(f'x   {file} failed: {exc}')
                    exit_code = 1
                    continue

                if args.freq is not None:
                    print(f'o   {file} was distorted along {args.freq} cm-1')
                elif args.freqnum is not None:
                    print(f'o   {file} was distorted along freq #{args.freqnum}')
                else:
                    print(f'o   {file} had {im_freq} imaginary frequencies: processed')

        else:
            # Automatic calculations (single points for stability check)
            log_output = Logger("RUNIRC", 'dat', args.nummodes)

            if im_freq == 0:
                log_output.write(f'o   {file} has no imaginary frequencies: check for stability')

            amp_base = [round(elem, 2) for elem in np.arange(0, 1, 0.1)]
            root_dir = os.getcwd()
            file_path = Path(file)
            parent_dir = Path(root_dir) / file_path.stem

            if not parent_dir.exists():
                parent_dir.mkdir(parents=True)

            log_output.write(f'o  Entering directory {parent_dir}')

            # Determine frequency range
            if args.nummodes == 'all':
                freq_range = range(1, len(data.vibfreqs) + 1)
            else:
                freq_range = range(1, min(int(args.nummodes) + 1, len(data.vibfreqs) + 1))

            for num in freq_range:
                num_dir = parent_dir / f'num_{num}'
                if not num_dir.exists():
                    num_dir.mkdir()

                log_output.write(f'o  Entering directory {num_dir}')
                shutil.copyfile(file, num_dir / file_path.name)

                with working_directory(num_dir):
                    for amp in amp_base:
                        suffix = f'num_{num}_amp_{str(amp).replace(".", "")}'
                        gdata = OutputData(file)
                        run_irc(
                            file_path.name, args, num, amp, gdata.LEVELOFTHEORY,
                            suffix, log_output
                        )
                        log_output.write(f'o  Writing to file {file_path.stem}_{suffix}')

            log_output.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
