#!/usr/bin/env python
"""Real Gaussian 16, ORCA 5/6 and Q-Chem 6 outputs from the GoodVibes tests.

The expected values below were checked against each program's own printout
(imaginary-mode counts, charge and multiplicity from the input), not against
pyQRC. See tests/data/goodvibes/README.md for where the files come from.
"""

import pytest
import numpy as np

from pyqrc.pyQRC import QRCGenerator, main, parse_output
from tests.conftest import GOODVIBES_PATH, QCHEM_DEV_CCLIB_SKIP

# file: (natom, number of vibrations, imaginary modes, charge, multiplicity)
EXPECTED = {
    'g16/04_benzene_radical_cation.log': (12, 30, 0, 1, 2),
    'g16/16_o2_superoxide_anion.log': (2, 1, 0, -1, 2),
    'g16/19_acetic_acid_smd_dmso.log': (8, 18, 0, 0, 1),
    'g16/22_hcn_linear_freq_noraman.log': (3, 4, 0, 0, 1),
    'g16/24_iodobenzene_genecp_sdd.log': (12, 30, 0, 0, 1),
    'g16/44_ts_sn2_identity_chloride.log': (6, 12, 1, -1, 1),
    'g16/45_ts_diels_alder_butadiene_ethylene.log': (16, 42, 1, 0, 1),
    'g16/46_ts_h3_hydrogen_abstraction.log': (3, 4, 1, 0, 2),
    'g16/47_ts_e2_elimination_ethylchloride.log': (10, 24, 1, -1, 1),
    'g16/48_ts_nh3_umbrella_inversion.log': (4, 6, 1, 0, 1),
    'orca5/04_benzene_radical_cation.out': (12, 30, 1, 1, 2),
    'orca5/16_o2_superoxide_anion.out': (2, 1, 0, -1, 2),
    'orca5/22_hcn_linear_freq_noraman.out': (3, 4, 0, 0, 1),
    'orca5/25_pd_complex_genecp_def2.out': (19, 51, 1, 0, 1),
    'orca5/29_aniline_cpcm_chloroform.out': (14, 36, 1, 0, 1),
    'orca5/44_ts_sn2_identity_chloride.out': (6, 12, 1, -1, 1),
    'orca5/45_ts_diels_alder_butadiene_ethylene.out': (16, 42, 1, 0, 1),
    'orca5/47_ts_e2_elimination_ethylchloride.out': (10, 24, 1, -1, 1),
    'orca5/48_ts_nh3_umbrella_inversion.out': (4, 6, 1, 0, 1),
    'orca6/04_benzene_radical_cation.out': (12, 30, 1, 1, 2),
    'orca6/16_o2_superoxide_anion.out': (2, 1, 0, -1, 2),
    'orca6/21_naphthalene_xtb2_semiempirical.out': (18, 48, 0, 0, 1),
    'orca6/22_hcn_linear_freq_noraman.out': (3, 4, 0, 0, 1),
    'orca6/25_pd_complex_genecp_def2.out': (19, 51, 1, 0, 1),
    'orca6/29_aniline_cpcm_chloroform.out': (14, 36, 1, 0, 1),
    'orca6/37_planar_cyclohexane_3rd_order_saddle.out': (18, 48, 3, 0, 1),
    'orca6/44_ts_sn2_identity_chloride.out': (6, 12, 1, -1, 1),
    'orca6/45_ts_diels_alder_butadiene_ethylene.out': (16, 42, 1, 0, 1),
    'orca6/47_ts_e2_elimination_ethylchloride.out': (10, 24, 1, -1, 1),
    'orca6/48_ts_nh3_umbrella_inversion.out': (4, 6, 1, 0, 1),
    'qchem6/04_benzene_radical_cation.out': (12, 30, 1, 1, 2),
    'qchem6/16_o2_superoxide_anion.out': (2, 1, 0, -1, 2),
    'qchem6/19_acetic_acid_smd_dmso.out': (8, 18, 1, 0, 1),
    'qchem6/22_hcn_linear_freq_noraman.out': (3, 4, 0, 0, 1),
    'qchem6/24_iodobenzene_genecp_sdd.out': (12, 30, 0, 0, 1),
    'qchem6/44_ts_sn2_identity_chloride.out': (6, 12, 1, -1, 1),
    'qchem6/45_ts_diels_alder_butadiene_ethylene.out': (16, 42, 1, 0, 1),
    'qchem6/46_ts_h3_hydrogen_abstraction.out': (3, 4, 1, 0, 2),
    'qchem6/47_ts_e2_elimination_ethylchloride.out': (10, 24, 1, -1, 1),
    'qchem6/48_ts_nh3_umbrella_inversion.out': (4, 6, 1, 0, 1),
}


def _param(name):
    marks = [QCHEM_DEV_CCLIB_SKIP] if name.startswith('qchem6') else []
    return pytest.param(name, id=name, marks=marks)


def _gv(name):
    return GOODVIBES_PATH / name


def _qrc(name, **kwargs):
    kwargs.setdefault('write', False)
    return QRCGenerator(str(_gv(name)), kwargs.pop('amplitude', 0.3), 1, '4GB', None, False, 'QRC',
                        None, kwargs.pop('num', None), **kwargs)


def test_manifest_lists_every_fixture():
    on_disk = {f'{p.parent.name}/{p.name}' for p in GOODVIBES_PATH.glob('*/*') if p.suffix in ('.log', '.out')}
    assert on_disk == set(EXPECTED)


@pytest.mark.parametrize('name', [_param(name) for name in sorted(EXPECTED)])
def test_parsed_as_expected(name):
    natom, nvib, nimag, charge, mult = EXPECTED[name]
    qrc = _qrc(name, num=None if nimag else 1)
    assert qrc.NATOMS == natom
    assert len(qrc.FREQS) == nvib
    assert int(np.sum(np.asarray(qrc.FREQS) < 0)) == nimag
    assert (qrc.CHARGE, qrc.MULT) == (charge, mult)
    # Imaginary modes by default; mode 1 for minima
    assert qrc._target_modes == (set(range(nimag)) if nimag else {0})
    assert qrc.MW_DISTANCE > 0


@pytest.mark.parametrize('name', [_param(name) for name in sorted(EXPECTED) if EXPECTED[name][2]])
def test_cli_processes_saddle_points(name, temp_workdir, monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['pyqrc', str(_gv(name)), '--both', '-q'])
    assert main() == 0
    out = capsys.readouterr().out
    assert 'imaginary frequencies: processed' in out
    assert len(list(temp_workdir.iterdir())) == 2


class TestFeaturesOnRealOutputs:
    """Features and bug fixes, checked on real outputs."""

    def test_orca_keeps_all_imaginary_modes(self):
        """ORCA's 'first frequency considered to be a vibration' skips imaginary modes too."""
        data = parse_output(str(_gv('orca6/37_planar_cyclohexane_3rd_order_saddle.out')))
        np.testing.assert_allclose(data.vibfreqs[:4], [-345.73, -242.14, -241.53, 480.06])

    def test_orca_linear_molecule(self):
        data = parse_output(str(_gv('orca6/22_hcn_linear_freq_noraman.out')))
        np.testing.assert_allclose(data.vibfreqs, [788.79, 788.79, 2252.99, 3468.88])

    def test_orca_xtb_charge_from_input(self):
        """xTB jobs print no 'Total Charge' line; the echoed '* xyz 0 1' is used."""
        data = parse_output(str(_gv('orca6/21_naphthalene_xtb2_semiempirical.out')))
        assert (data.charge, data.mult) == (0, 1)

    @QCHEM_DEV_CCLIB_SKIP
    def test_qchem_ecp_charge_from_input(self):
        """cclib gives charge +46 for iodobenzene with an ECP; the input says 0."""
        assert (_qrc('qchem6/24_iodobenzene_genecp_sdd.out', num=1).CHARGE) == 0

    def test_gaussian_genecp_sections_copied(self, tmp_path, capsys):
        qrc = _qrc('g16/24_iodobenzene_genecp_sdd.log', num=1, outdir=str(tmp_path))
        qrc.write_files()
        text = (tmp_path / '24_iodobenzene_genecp_sdd_QRC.com').read_text()
        assert '# B3LYP/GenECP emp=gd3bj Opt Freq' in text
        assert text.endswith('\n\nC H 0\n6-311+G(d,p)\n****\nI 0\nSDD\n****\n\nI 0\nSDD\n\n')
        assert 'copied from' in capsys.readouterr().out

    @QCHEM_DEV_CCLIB_SKIP
    def test_qchem_genecp_sections_copied(self, tmp_path):
        qrc = _qrc('qchem6/24_iodobenzene_genecp_sdd.out', num=1, outdir=str(tmp_path))
        qrc.write_files()
        opt_job, freq_job = (tmp_path / '24_iodobenzene_genecp_sdd_QRC.inp').read_text().split('@@@')
        for job in (opt_job, freq_job):
            assert 'BASIS    gen' in job and 'ECP      gen' in job
            assert job.count('$basis') == 1 and job.count('$ecp') == 1

    def test_orca_basis_block_copied_and_pal_removed(self, tmp_path):
        qrc = _qrc('orca6/25_pd_complex_genecp_def2.out', outdir=str(tmp_path))
        qrc.write_files()
        lines = (tmp_path / '25_pd_complex_genecp_def2_QRC.inp').read_text().splitlines()
        assert lines[0] == '! PBE0 D3BJ Opt Freq'
        assert '  NewECP Pd "def2-ECP" end' in lines

    def test_gaussian_unfinished_freq_step_warns(self, tmp_path, capsys):
        """This output stops after the frequency step's archive block."""
        _qrc('g16/46_ts_h3_hydrogen_abstraction.log', write=True, outdir=str(tmp_path))
        assert 'did not terminate normally' in capsys.readouterr().out

    @pytest.mark.parametrize('name', [
        'g16/44_ts_sn2_identity_chloride.log',
        'orca6/44_ts_sn2_identity_chloride.out',
        _param('qchem6/44_ts_sn2_identity_chloride.out'),
    ])
    def test_sn2_reaction_coordinate(self, name):
        """The SN2 TS mode shortens one C-Cl bond while lengthening the other."""
        fwd, rev = _qrc(name, amplitude=0.3), _qrc(name, amplitude=-0.3)
        cl = [i for i, el in enumerate(fwd.ATOMTYPES) if el == 'Cl']
        c = fwd.ATOMTYPES.index('C')

        def bonds(xyz):
            return [np.linalg.norm(xyz[c] - xyz[i]) for i in cl]

        (f1, f2), (r1, r2) = bonds(fwd.NEW_CARTESIAN), bonds(rev.NEW_CARTESIAN)
        assert (f1 - f2) * (r1 - r2) < 0, 'the two directions must favour opposite chlorides'
        assert abs(f1 - f2) > 0.1
