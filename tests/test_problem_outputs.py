#!/usr/bin/env python
"""Outputs that commonly go wrong: truncated, multi-step, Windows, non-text.

Each must either be processed correctly or fail with a clear message and
a non-zero exit code; never a traceback.
"""

from pathlib import Path

import numpy as np
import pytest

from pyqrc.pyQRC import OutputData, QRCGenerator, main
from tests.conftest import QCHEM_DEV_CCLIB_SKIP, datapath

G16_TS = datapath('g16/claisen_ts.log')
ORCA6_TS = datapath('orca6/claisen_ts.out')
QCHEM_TS = datapath('qchem/claisen_ts.out')


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr('sys.argv', ['pyqrc', *map(str, argv)])
    code = main()
    return code, capsys.readouterr().out


def _crlf(src, dest):
    Path(dest).write_bytes(Path(src).read_bytes().replace(b'\n', b'\r\n'))
    return dest


class TestTruncatedOutputs:
    """Jobs that died or were cut off."""

    def test_gaussian_cut_in_frequency_block(self, temp_workdir, monkeypatch, capsys):
        """Still processed (the imaginary mode is complete) but with both warnings."""
        text = G16_TS.read_text()
        first = text.index('Frequencies --')
        (temp_workdir / 'cut.log').write_text(text[:text.index('Frequencies --', first + 10)])
        code, out = _run(monkeypatch, capsys, 'cut.log', '-q')
        assert code == 0
        assert 'only 3 of 36 normal modes were read' in out
        assert 'did not terminate normally' in out

    def test_second_step_termination_required(self, tmp_path):
        """An opt+freq job has two steps; the opt step's termination is not enough."""
        text = G16_TS.read_text()
        cut = tmp_path / 'cut.log'
        cut.write_text(text[:text.rindex('Normal termination')])
        assert OutputData(str(G16_TS)).TERMINATION == 'normal'
        assert OutputData(str(cut)).TERMINATION is None

    def test_gaussian_without_frequencies(self, temp_workdir, monkeypatch, capsys):
        text = G16_TS.read_text()
        (temp_workdir / 'opt.log').write_text(text[:text.index(' Harmonic frequencies')])
        code, out = _run(monkeypatch, capsys, 'opt.log')
        assert 'has no frequency information: skipping' in out
        assert code == 1
        assert not list(temp_workdir.glob('opt_QRC*'))

    def test_orca_cut_in_normal_modes(self, temp_workdir, monkeypatch, capsys):
        text = ORCA6_TS.read_text()
        (temp_workdir / 'cut.out').write_text(text[:text.index('NORMAL MODES') + 3000])
        code, out = _run(monkeypatch, capsys, 'cut.out')
        assert code == 1
        assert 'failed to parse' in out and 'pyQRC ORCA reader' in out
        assert not list(temp_workdir.glob('cut_QRC*'))


class TestMultiStepGaussian:
    """opt+freq jobs print two route sections (the second is the freq step)."""

    def test_first_route_is_used(self):
        assert OutputData(str(datapath('g16/acetaldehyde.log'))).JOBTYPE.strip() == 'opt freq M062X/6-31G*'

    def test_geometry_is_from_the_freq_step(self):
        qrc = QRCGenerator(str(G16_TS), 0.3, 1, '4GB', None, False, 'QRC', None, None, write=False)
        text = G16_TS.read_text().splitlines()
        # Last "Standard orientation" block in the file is the frequency step's geometry
        start = max(i for i, line in enumerate(text) if 'Standard orientation' in line)
        first_atom = [float(x) for x in text[start + 5].split()[3:6]]
        np.testing.assert_allclose(qrc.CARTESIAN[0], first_atom)


class TestWindowsLineEndings:
    """CRLF outputs and inputs give the same result as LF, and LF files are written."""

    @pytest.mark.parametrize('source,ext', [
        (G16_TS, '.log'),
        (ORCA6_TS, '.out'),
        pytest.param(QCHEM_TS, '.out', marks=QCHEM_DEV_CCLIB_SKIP),
    ])
    def test_crlf_matches_lf(self, source, ext, tmp_path):
        crlf_dir, lf_dir = tmp_path / 'crlf', tmp_path / 'lf'
        crlf_dir.mkdir()
        crlf = _crlf(source, crlf_dir / f'ts{ext}')
        lf = lf_dir / f'ts{ext}'
        lf_dir.mkdir()
        lf.write_bytes(source.read_bytes())
        if ext == '.log':  # the Gaussian input next to the output, too
            _crlf(source.with_suffix('.com'), crlf_dir / 'ts.com')
            (lf_dir / 'ts.com').write_bytes(source.with_suffix('.com').read_bytes())
        written = []
        for path, out in ((crlf, tmp_path / 'o1'), (lf, tmp_path / 'o2')):
            qrc = QRCGenerator(str(path), 0.3, 1, '4GB', None, False, 'QRC', None, None,
                               write=False, outdir=str(out))
            qrc.write_files()
            written.append(qrc.output_paths()[0].read_bytes())
        assert written[0] == written[1]
        assert b'\r' not in written[0]


class TestNotAnOutput:
    """Files that are not frequency outputs fail cleanly."""

    @pytest.mark.parametrize('name,content', [
        ('empty.log', b''),
        ('binary.out', bytes(range(256)) * 50),
        ('notes.log', b'just some text\nnothing to see here\n'),
    ])
    def test_unrecognized(self, name, content, temp_workdir, monkeypatch, capsys):
        (temp_workdir / name).write_bytes(content)
        code, out = _run(monkeypatch, capsys, name)
        assert code == 1
        assert f'x   {name}' in out
        assert sorted(p.name for p in temp_workdir.iterdir()) == [name]

    def test_input_file_passed_by_mistake(self, temp_workdir, monkeypatch, capsys):
        code, out = _run(monkeypatch, capsys, datapath('g16/claisen_ts.com'))
        assert code == 1
        assert 'expected a .log or .out file' in out

    def test_one_bad_file_does_not_stop_the_batch(self, temp_workdir, monkeypatch, capsys):
        (temp_workdir / 'empty.log').write_bytes(b'')
        code, out = _run(monkeypatch, capsys, 'empty.log', G16_TS, '-q')
        assert code == 1
        assert 'x   empty.log' in out and 'claisen_ts.log had 1 imaginary' in out
        assert (temp_workdir / 'claisen_ts_QRC.com').exists()
