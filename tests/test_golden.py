#!/usr/bin/env python
"""Golden-file tests: the input pyQRC writes for every example output.

Each example output in examples/ (collected as in conftest) is run through
QRCGenerator with fixed options, and the input file written is compared
with tests/golden/<example dir>/<name>_QRC.<ext>. Any change to a
generated input, intended or not, shows up here as a diff.

To add an example: put the output (and, for Gaussian, its .com/.gjf input)
in examples/<program>/, run `pytest tests/test_golden.py --update-golden`,
check the new file in tests/golden/, and commit both.
"""

import difflib
from pathlib import Path

import pytest

from pyqrc.pyQRC import QRCGenerator
from tests.conftest import GOLDEN_PATH, QCHEM_DEV_CCLIB_SKIP, get_all_example_files


def _golden_params():
    return [
        pytest.param(path, id=f'{path.parent.name}/{path.stem}',
                     marks=[QCHEM_DEV_CCLIB_SKIP] if fmt == 'QChem' else [])
        for path, fmt in sorted(get_all_example_files())
    ]


@pytest.mark.parametrize('example', _golden_params())
def test_generated_input_matches_golden(example, tmp_path, update_golden):
    """Every example gives exactly the committed input (amp 0.3, 4 cores, 8GB)."""
    qrc = QRCGenerator(str(example), 0.3, 4, '8GB', None, False, 'QRC', None, None,
                       write=False, outdir=str(tmp_path))
    qrc.write_files()
    (written,) = qrc.output_paths()
    golden = GOLDEN_PATH / example.parent.name / written.name
    text = written.read_text(encoding='utf-8')

    if update_golden:
        golden.parent.mkdir(parents=True, exist_ok=True)
        with open(golden, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        return
    assert golden.exists(), (
        f'no golden file for {example.parent.name}/{example.name}: run '
        '`pytest tests/test_golden.py --update-golden` and review the new file'
    )
    expected = golden.read_text(encoding='utf-8')
    diff = ''.join(difflib.unified_diff(
        expected.splitlines(keepends=True), text.splitlines(keepends=True),
        fromfile=str(golden.relative_to(GOLDEN_PATH.parent.parent)), tofile='generated'))
    assert text == expected, f'generated input differs from the golden file:\n{diff}'


def test_every_golden_file_has_an_example():
    """No stale golden files for examples that were removed or renamed."""
    expected = {
        (path.parent.name, f'{path.stem}_QRC') for path, _ in get_all_example_files()
    }
    stale = [str(p.relative_to(GOLDEN_PATH)) for p in GOLDEN_PATH.glob('*/*')
             if (p.parent.name, p.stem) not in expected]
    assert not stale, f'golden files without an example: {stale}'


def test_golden_files_are_committed_text():
    """Golden inputs use LF line endings and end with a newline."""
    files = list(GOLDEN_PATH.glob('*/*'))
    assert files, 'tests/golden is empty'
    for path in files:
        data = Path(path).read_bytes()
        assert b'\r' not in data and data.endswith(b'\n'), path
