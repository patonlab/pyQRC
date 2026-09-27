#!/usr/bin/env python
"""Error paths and less common inputs for the ORCA reader and input carry-over."""

import re
from pathlib import Path

import numpy as np
import pytest

from pyqrc.orca_reader import ORCAReadError, _count_translations_rotations, read_orca_frequencies
from pyqrc.pyQRC import (
    QRCGenerator, QRCParseError, _blank_line_sections, parse_output,
    read_gaussian_input_tail, read_orca_input, read_qchem_charge_mult, read_qchem_input,
)
from tests.conftest import QCHEM_DEV_CCLIB_SKIP, datapath

ORCA6_TS = datapath('orca6/claisen_ts.out')
QCHEM_TS = datapath('qchem/claisen_ts.out')


def _edit_last(text, marker, edit):
    """Apply edit(block) to the text from the last occurrence of marker onwards."""
    start = text.rindex(marker)
    return text[:start] + edit(text[start:])


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


class TestORCAReaderErrors:
    """Malformed ORCA outputs give ORCAReadError with a clear reason."""

    @pytest.fixture
    def text(self):
        return ORCA6_TS.read_text()

    def _geometry_edit(self, text, edit):
        """Edit the last geometry printed before the frequency block."""
        freq = text.rindex('VIBRATIONAL FREQUENCIES')
        start = text.rindex('CARTESIAN COORDINATES (ANGSTROEM)', 0, freq)
        return text[:start] + edit(text[start:freq]) + text[freq:]

    def test_no_coordinates(self, text, tmp_path):
        path = _write(tmp_path, 'a.out', text.replace('CARTESIAN COORDINATES (ANGSTROEM)', 'COORDS'))
        with pytest.raises(ORCAReadError, match='no Cartesian coordinates'):
            read_orca_frequencies(path)

    def test_unknown_element(self, text, tmp_path):
        text = self._geometry_edit(text, lambda block: re.sub(r'\n  C ', '\n  Qq ', block, count=1))
        with pytest.raises(ORCAReadError, match='unknown element'):
            read_orca_frequencies(_write(tmp_path, 'a.out', text))

    def test_empty_geometry(self, text, tmp_path):
        def empty(block):
            header, _, rest = block.partition('\n---------------------------------\n')
            return header + '\n---------------------------------\n\n' + rest.split('\n\n', 1)[1]
        with pytest.raises(ORCAReadError, match='empty Cartesian'):
            read_orca_frequencies(_write(tmp_path, 'a.out', self._geometry_edit(text, empty)))

    def test_no_charge_anywhere(self, text, tmp_path):
        text = re.sub(r'Total Charge.*\n', '\n', text)
        text = re.sub(r'(\|\s*\d+>\s*)\*\s*xyz\s+0\s+1', r'\1* xyz', text)
        with pytest.raises(ORCAReadError, match='charge or multiplicity'):
            read_orca_frequencies(_write(tmp_path, 'a.out', text))

    def test_frequencies_out_of_order(self, text, tmp_path):
        text = _edit_last(text, 'VIBRATIONAL FREQUENCIES',
                          lambda block: block.replace('     7:', '     9:', 1))
        with pytest.raises(ORCAReadError, match='not numbered consecutively'):
            read_orca_frequencies(_write(tmp_path, 'a.out', text))

    def test_missing_frequency(self, text, tmp_path):
        text = _edit_last(text, 'VIBRATIONAL FREQUENCIES',
                          lambda block: re.sub(r'\n\s+41:\s+-?\d+\.\d+ cm\*\*-1[^\n]*', '', block, count=1))
        with pytest.raises(ORCAReadError, match='expected 42 frequencies, found 41'):
            read_orca_frequencies(_write(tmp_path, 'a.out', text))

    def test_incomplete_normal_modes(self, text, tmp_path):
        text = _edit_last(text, 'NORMAL MODES',
                          lambda block: re.sub(r'\n\s+41(\s+-?\d+\.\d+){6}', '', block, count=1))
        with pytest.raises(ORCAReadError, match='incomplete normal mode block'):
            read_orca_frequencies(_write(tmp_path, 'a.out', text))


class TestTranslationsRotations:
    def test_atom(self):
        assert _count_translations_rotations(np.zeros((1, 3)), [0.0, 0.0, 0.0]) == 3

    def test_linear_from_geometry(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [0.0, 0.0, 2.2]])
        assert _count_translations_rotations(coords, [0.0] * 5 + [500.0] * 4) == 5

    def test_printed_zeros_win_for_nearly_linear(self):
        """ORCA's own count of zero modes wins over the geometry test."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 1.1], [0.0, 0.0, 2.2]])
        assert _count_translations_rotations(coords, [0.0] * 5 + [500.0] * 4) == 5
        assert _count_translations_rotations(coords, [0.0] * 6 + [500.0] * 3) == 6


class TestParseOutputErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            parse_output(str(tmp_path / 'missing.out'))

    def test_generator_keeps_orca_reader_error(self, tmp_path):
        text = ORCA6_TS.read_text()
        path = _write(tmp_path, 'cut.out', text[:text.rindex('NORMAL MODES')])
        with pytest.raises(QRCParseError, match='pyQRC ORCA reader'):
            QRCGenerator(path, 0.3, 1, '4GB', None, False, 'QRC', None, None, write=False)


class TestGaussianTailEdgeCases:
    def test_sections_without_trailing_blank_line(self):
        assert _blank_line_sections(['a', '', 'b', 'c']) == [['a'], ['b', 'c']]

    def test_empty_input_file(self, tmp_path):
        (tmp_path / 'x.log').write_text('')
        (tmp_path / 'x.com').write_text('\n\n')
        assert read_gaussian_input_tail(str(tmp_path / 'x.log')) == (str(tmp_path / 'x.com'), None, None)

    def test_allcheck_input(self, tmp_path):
        """geom=allcheck inputs have no title or charge sections."""
        (tmp_path / 'x.log').write_text('')
        (tmp_path / 'x.com').write_text('%chk=x\n# opt b3lyp/gen geom=allcheck\n\nC 0\n6-31G\n****\n\n')
        _, route, tail = read_gaussian_input_tail(str(tmp_path / 'x.log'))
        assert 'allcheck' in route and tail == 'C 0\n6-31G\n****'

    def test_non_numeric_cartesian_line_is_zmatrix(self, tmp_path):
        (tmp_path / 'x.log').write_text('')
        (tmp_path / 'x.com').write_text('# opt b3lyp/gen\n\nt\n\n0 1\nC 0.0 0.0 0.0\nH 0.0 0.0 r\n\nH 0\n6-31G\n****\n\n')
        assert read_gaussian_input_tail(str(tmp_path / 'x.log'))[2] is None


class TestORCAInputEdgeCases:
    ECHO = ['INPUT FILE\n', '=====\n', 'NAME = x.inp\n']

    def _echo(self, *lines):
        body = [f'| {i + 1:2d}> {line}\n' for i, line in enumerate(lines)]
        return self.ECHO + body + ['|  99>                          ****END OF INPUT****\n']

    def test_xyzfile_line_skipped(self):
        assert read_orca_input(self._echo('! Opt', '* xyzfile 0 1 start.xyz', '%scf maxiter 200 end')) \
            == ('Opt', ['%scf maxiter 200 end'])

    def test_coords_block_skipped(self):
        lines = ('! Opt', '%coords', '  CTyp xyz', '  Charge 0', '  Mult 1', '  coords',
                 '    H 0 0 0', '    H 0 0 0.74', '  end', 'end', '%scf maxiter 200 end')
        assert read_orca_input(self._echo(*lines)) == ('Opt', ['%scf maxiter 200 end'])


class TestQChemEdgeCases:
    def test_no_molecule_section(self, tmp_path):
        assert read_qchem_charge_mult(_write(tmp_path, 'q.out', 'nothing\n')) is None

    def test_no_echoed_input(self, tmp_path):
        assert read_qchem_input(_write(tmp_path, 'q.out', 'nothing\n')) is None

    def test_echo_without_rem(self, tmp_path):
        text = 'User input:\n----\n$molecule\n0 1\nH 0 0 0\nH 0 0 0.74\n$end\n----\n'
        assert read_qchem_input(_write(tmp_path, 'q.out', text)) is None

    @QCHEM_DEV_CCLIB_SKIP
    def test_route_ignored_with_warning(self, tmp_path, capsys):
        QRCGenerator(str(QCHEM_TS), 0.3, 1, '4GB', 'opt b3lyp', False, 'QRC', None, None,
                     outdir=str(tmp_path))
        assert '--route is ignored for Q-Chem' in capsys.readouterr().out

    @QCHEM_DEV_CCLIB_SKIP
    def test_metadata_fallback_without_echo(self, tmp_path, monkeypatch):
        """Without echoed input, METHOD and BASIS come from cclib metadata."""
        monkeypatch.setattr('pyqrc.pyQRC.read_qchem_input', lambda _: None)
        QRCGenerator(str(QCHEM_TS), 0.3, 1, '4GB', None, False, 'QRC', None, None, outdir=str(tmp_path))
        opt_job, freq_job = (tmp_path / 'claisen_ts_QRC.inp').read_text().split('@@@')
        for job in (opt_job, freq_job):
            assert re.search(r'METHOD \S+', job) and re.search(r'BASIS \S+', job)
            assert 'SCF_CONVERGENCE' not in job


def test_output_paths_without_cclib_format(tmp_path, monkeypatch):
    """output_paths falls back to OutputData when cclib gave no package name."""
    qrc = QRCGenerator(str(datapath('g16/claisen_ts.log')), 0.3, 1, '4GB', None, True, 'QRC', None, None,
                       write=False, outdir=str(tmp_path))
    qrc._format_type = None  # pylint: disable=protected-access
    assert [Path(p).name for p in qrc.output_paths()] == ['claisen_ts_QRC.com', 'claisen_ts_QRC.qrc']
