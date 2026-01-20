#!/usr/bin/env python
"""pytest configuration and fixtures for pyQRC tests."""

from pathlib import Path

import pytest

# Determine the base path for test data
try:
    import pyqrc
    BASEPATH = Path(pyqrc.__path__[0])
except ImportError:
    here = Path(__file__).parent
    BASEPATH = here.parent / 'pyqrc'

EXAMPLES_PATH = BASEPATH.parent / 'examples'


def datapath(path: str) -> Path:
    """Return the full path to a test data file.

    Args:
        path: Relative path within the examples directory.

    Returns:
        Full path to the test data file.
    """
    return EXAMPLES_PATH / path


def get_all_example_files():
    """Discover all example output files in the examples directory.

    Returns:
        List of tuples (filepath, format_name) for each example file.
    """
    examples = []

    # Gaussian files (.log)
    g16_path = EXAMPLES_PATH / 'g16'
    if g16_path.exists():
        for f in g16_path.glob('*.log'):
            examples.append((f, 'Gaussian'))

    # ORCA files (.out)
    orca_path = EXAMPLES_PATH / 'orca6'
    if orca_path.exists():
        for f in orca_path.glob('*.out'):
            examples.append((f, 'ORCA'))

    # Q-Chem files (.out)
    qchem_path = EXAMPLES_PATH / 'qchem'
    if qchem_path.exists():
        for f in qchem_path.glob('*.out'):
            examples.append((f, 'QChem'))

    return examples


def get_example_files_by_type(file_type: str):
    """Get example files filtered by type.

    Args:
        file_type: One of 'all', 'minima', 'ts', 'saddle'
            - 'all': All files
            - 'minima': Files without imaginary frequencies (acetaldehyde, optimized structures)
            - 'ts': Transition states (claisen_ts)
            - 'saddle': Higher-order saddle points (planar_chex)

    Returns:
        List of tuples (filepath, format_name).
    """
    all_files = get_all_example_files()

    if file_type == 'all':
        return all_files

    filtered = []
    for filepath, fmt in all_files:
        name = filepath.stem.lower()

        if file_type == 'minima':
            # Acetaldehyde and optimized mode files are minima
            if 'acetaldehyde' in name or '_mode' in name:
                filtered.append((filepath, fmt))

        elif file_type == 'ts':
            # Claisen TS files
            if 'claisen_ts' in name:
                filtered.append((filepath, fmt))

        elif file_type == 'saddle':
            # Planar cyclohexane (higher-order saddle point)
            if name == 'planar_chex':
                filtered.append((filepath, fmt))

    return filtered


# Individual fixtures for specific files (for tests that need specific files)
@pytest.fixture
def g16_acetaldehyde():
    """Gaussian 16 acetaldehyde frequency output."""
    return datapath('g16/acetaldehyde.log')


@pytest.fixture
def g16_claisen_ts():
    """Gaussian 16 Claisen TS frequency output."""
    return datapath('g16/claisen_ts.log')


@pytest.fixture
def g16_planar_chex():
    """Gaussian 16 planar cyclohexane (3rd order saddle point)."""
    return datapath('g16/planar_chex.log')


@pytest.fixture
def orca_acetaldehyde():
    """ORCA 6 acetaldehyde frequency output."""
    return datapath('orca6/acetaldehyde.out')


@pytest.fixture
def orca_claisen_ts():
    """ORCA 6 Claisen TS frequency output."""
    return datapath('orca6/claisen_ts.out')


@pytest.fixture
def qchem_acetaldehyde():
    """Q-Chem acetaldehyde frequency output."""
    return datapath('qchem/acetaldehyde.out')


@pytest.fixture
def qchem_claisen_ts():
    """Q-Chem Claisen TS frequency output."""
    return datapath('qchem/claisen_ts.out')


@pytest.fixture
def temp_workdir(tmp_path, monkeypatch):
    """Change to a temporary working directory for tests that create files."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# Parametrized fixtures for running tests against all example files
def pytest_generate_tests(metafunc):
    """Generate test parameters for example file fixtures."""

    if 'example_file' in metafunc.fixturenames:
        examples = get_all_example_files()
        if examples:
            metafunc.parametrize(
                'example_file,example_format',
                [(str(f), fmt) for f, fmt in examples],
                ids=[f.stem for f, _ in examples]
            )

    if 'ts_file' in metafunc.fixturenames:
        ts_files = get_example_files_by_type('ts')
        if ts_files:
            metafunc.parametrize(
                'ts_file,ts_format',
                [(str(f), fmt) for f, fmt in ts_files],
                ids=[f.stem for f, _ in ts_files]
            )

    if 'saddle_file' in metafunc.fixturenames:
        saddle_files = get_example_files_by_type('saddle')
        if saddle_files:
            metafunc.parametrize(
                'saddle_file,saddle_format',
                [(str(f), fmt) for f, fmt in saddle_files],
                ids=[f.stem for f, _ in saddle_files]
            )
