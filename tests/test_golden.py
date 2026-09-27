#!/usr/bin/env python
"""Golden-file tests: the input pyQRC writes for every example output.

Each example output in examples/ and each real output from the GoodVibes
test suite in tests/data/goodvibes/ is run through QRCGenerator with fixed
options (amplitude 0.3, 4 cores, 8GB; all imaginary modes, or mode 1 for a
minimum so that its input is still exercised), and the input file written
is compared with tests/golden/[goodvibes/]<program dir>/<name>_QRC.<ext>.
Any change to a generated input, intended or not, shows up here as a diff.

To add an example: put the output (and, for Gaussian, its .com/.gjf input)
in examples/<program>/, run `pytest tests/test_golden.py --update-golden`,
check the new file in tests/golden/, and commit both.
"""

import difflib
from pathlib import Path

import pytest

from pyqrc.pyQRC import QRCGenerator, parse_output
from tests.conftest import (
    GOLDEN_PATH, QCHEM_DEV_CCLIB_SKIP, get_all_example_files, get_goodvibes_files, golden_name,
)


def _all_outputs():
    return sorted(get_all_example_files()) + get_goodvibes_files()


def _golden_params():
    return [
        pytest.param(path, id=f'{golden_name(path)}/{path.stem}',
                     marks=[QCHEM_DEV_CCLIB_SKIP] if fmt == 'QChem' else [])
        for path, fmt in _all_outputs()
    ]


@pytest.mark.parametrize('example', _golden_params())
def test_generated_input_matches_golden(example, tmp_path, update_golden):
    """Every output gives exactly the committed input (amp 0.3, 4 cores, 8GB)."""
    has_imaginary = any(freq < 0 for freq in parse_output(str(example)).vibfreqs)
    qrc = QRCGenerator(str(example), 0.3, 4, '8GB', None, False, 'QRC', None,
                       None if has_imaginary else 1, write=False, outdir=str(tmp_path))
    qrc.write_files()
    (written,) = qrc.output_paths()
    golden = GOLDEN_PATH / golden_name(example) / written.name
    text = written.read_text(encoding='utf-8')

    if update_golden:
        golden.parent.mkdir(parents=True, exist_ok=True)
        with open(golden, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        return
    assert golden.exists(), (
        f'no golden file for {golden_name(example)}/{example.name}: run '
        '`pytest tests/test_golden.py --update-golden` and review the new file'
    )
    expected = golden.read_text(encoding='utf-8')
    diff = ''.join(difflib.unified_diff(
        expected.splitlines(keepends=True), text.splitlines(keepends=True),
        fromfile=str(golden.relative_to(GOLDEN_PATH.parent.parent)), tofile='generated'))
    assert text == expected, f'generated input differs from the golden file:\n{diff}'


def test_every_golden_file_has_an_example():
    """No stale golden files for examples that were removed or renamed."""
    expected = {(golden_name(path), f'{path.stem}_QRC') for path, _ in _all_outputs()}
    stale = [str(p.relative_to(GOLDEN_PATH)) for p in GOLDEN_PATH.rglob('*') if p.is_file()
             and (str(p.parent.relative_to(GOLDEN_PATH)), p.stem) not in expected]
    assert not stale, f'golden files without an example: {stale}'


def test_golden_files_are_committed_text():
    """Golden inputs use LF line endings and end with a newline."""
    files = [p for p in GOLDEN_PATH.rglob('*') if p.is_file()]
    assert files, 'tests/golden is empty'
    for path in files:
        data = Path(path).read_bytes()
        assert b'\r' not in data and data.endswith(b'\n'), path
