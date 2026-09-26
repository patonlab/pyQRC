#!/usr/bin/env python
"""Tests for pyQRC package."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import QCHEM_DEV_CCLIB_SKIP, datapath
from pyqrc.pyQRC import (
    ATOMIC_MASSES,
    COVALENT_RADII,
    PERIODIC_TABLE,
    Logger,
    OutputData,
    QRCGenerator,
    QRCModeError,
    QRCParseError,
    check_overlap,
    element_id,
    g16_opt,
    gen_overlap,
    main,
    mwdist,
    run_irc,
    working_directory,
)


class TestConstants:
    """Tests for module constants."""

    def test_periodic_table_length(self):
        """Periodic table should have 119 elements (index 0 is empty)."""
        assert len(PERIODIC_TABLE) == 119

    def test_periodic_table_common_elements(self):
        """Check common elements are at correct positions."""
        assert PERIODIC_TABLE[1] == "H"
        assert PERIODIC_TABLE[6] == "C"
        assert PERIODIC_TABLE[7] == "N"
        assert PERIODIC_TABLE[8] == "O"

    def test_atomic_masses_length(self):
        """Atomic masses array should match periodic table."""
        assert len(ATOMIC_MASSES) == len(PERIODIC_TABLE)

    def test_atomic_masses_values(self):
        """Check some known atomic masses."""
        assert ATOMIC_MASSES[1] == pytest.approx(1.007825, rel=1e-3)
        assert ATOMIC_MASSES[6] == pytest.approx(12.01, rel=1e-2)

    def test_covalent_radii_common_elements(self):
        """Check covalent radii for common elements."""
        assert "C" in COVALENT_RADII
        assert "H" in COVALENT_RADII
        assert COVALENT_RADII["C"] == pytest.approx(0.75, rel=0.1)


class TestElementId:
    """Tests for element_id function."""

    def test_valid_atomic_numbers(self):
        """Test conversion of valid atomic numbers."""
        assert element_id(1) == "H"
        assert element_id(6) == "C"
        assert element_id(26) == "Fe"

    def test_invalid_atomic_number(self):
        """Test that invalid atomic numbers return 'XX'."""
        assert element_id(999) == "XX"
        assert element_id(200) == "XX"


class TestLogger:
    """Tests for Logger class."""

    def test_logger_creates_file(self, tmp_path, monkeypatch):
        """Test that Logger creates a file."""
        monkeypatch.chdir(tmp_path)
        log = Logger("test", "log", "suffix")
        log.write("test message")
        log.close()

        expected_file = tmp_path / "test_suffix.log"
        assert expected_file.exists()
        assert "test message" in expected_file.read_text()

    def test_logger_context_manager(self, tmp_path, monkeypatch):
        """Test Logger as context manager."""
        monkeypatch.chdir(tmp_path)
        with Logger("test", "txt", "cm") as log:
            log.write("context manager test")

        expected_file = tmp_path / "test_cm.txt"
        assert expected_file.exists()


class TestMwdist:
    """Tests for mass-weighted distance function."""

    def test_identical_structures(self):
        """Distance between identical structures should be zero."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = [6, 1]  # C, H
        assert mwdist(coords, coords, elements) == pytest.approx(0.0)

    def test_simple_displacement(self):
        """Test distance with simple displacement."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0]])
        elements = [1]  # H
        dist = mwdist(coords1, coords2, elements)
        assert dist > 0


class TestOverlap:
    """Tests for overlap detection functions."""

    def test_no_overlap(self):
        """Test atoms that don't overlap."""
        atoms = ["C", "C"]
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        assert not check_overlap(atoms, coords)

    def test_overlap_detected(self):
        """Test atoms that do overlap."""
        atoms = ["C", "C"]
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        assert check_overlap(atoms, coords)

    def test_gen_overlap_matrix(self):
        """Test overlap matrix generation."""
        atoms = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        matrix = gen_overlap(atoms, coords, 0.8)
        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == 1  # Should overlap


class TestOutputData:
    """Tests for OutputData class."""

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            OutputData("/nonexistent/file.log")

    def test_gaussian_format_detection(self, g16_acetaldehyde):
        """Test Gaussian format is detected."""
        data = OutputData(str(g16_acetaldehyde))
        assert data.format == "Gaussian"

    def test_orca_format_detection(self, orca_acetaldehyde):
        """Test ORCA format is detected."""
        data = OutputData(str(orca_acetaldehyde))
        assert data.format == "ORCA"

    def test_qchem_format_detection(self, qchem_acetaldehyde):
        """Test Q-Chem format is detected."""
        data = OutputData(str(qchem_acetaldehyde))
        assert data.format == "QChem"


class TestQRCGeneratorAllFiles:
    """Tests for QRCGenerator against all example files."""

    def test_qrc_generation(self, example_file, example_format, temp_workdir):
        """Test QRC generation for all example files."""
        filepath = Path(example_file)

        # Copy file to temp directory
        shutil.copy(filepath, temp_workdir)
        local_file = temp_workdir / filepath.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None
        )

        assert qrc.CARTESIAN is not None
        assert qrc.NEW_CARTESIAN is not None
        assert qrc.ATOMTYPES is not None
        assert len(qrc.ATOMTYPES) > 0

        # Check output files were created
        stem = local_file.stem
        if example_format == "Gaussian":
            assert (temp_workdir / f"{stem}_QRC.com").exists(), "Gaussian input not created"
        elif example_format in ("ORCA", "QChem"):
            assert (temp_workdir / f"{stem}_QRC.inp").exists(), f"{example_format} input not created"

        # Verbose mode should create .qrc summary file
        assert (temp_workdir / f"{stem}_QRC.qrc").exists(), "QRC summary not created"


class TestQRCGeneratorTS:
    """Tests for QRCGenerator with transition state files."""

    def test_ts_displacement(self, ts_file, ts_format, temp_workdir):
        """Test that TS structures are displaced along imaginary mode."""
        filepath = Path(ts_file)

        shutil.copy(filepath, temp_workdir)
        local_file = temp_workdir / filepath.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None
        )

        # Structure should have been displaced (TS has imaginary frequency)
        displacement = np.linalg.norm(
            np.array(qrc.NEW_CARTESIAN) - np.array(qrc.CARTESIAN)
        )
        assert displacement > 0, "TS structure should be displaced along imaginary mode"


class TestQRCGeneratorSaddle:
    """Tests for QRCGenerator with higher-order saddle points."""

    def test_saddle_point_displacement(self, saddle_file, saddle_format, temp_workdir):
        """Test that saddle point structures are displaced along imaginary modes."""
        filepath = Path(saddle_file)

        shutil.copy(filepath, temp_workdir)
        local_file = temp_workdir / filepath.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None
        )

        # Structure should have been displaced
        displacement = np.linalg.norm(
            np.array(qrc.NEW_CARTESIAN) - np.array(qrc.CARTESIAN)
        )
        assert displacement > 0, "Saddle point should be displaced along imaginary modes"


class TestQRCGeneratorOptions:
    """Tests for QRCGenerator with various options."""

    def test_specific_frequency_number(self, g16_claisen_ts, temp_workdir):
        """Test displacement along specific frequency number."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.3,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC_mode1",
            val=None,
            num=1  # First frequency
        )

        displacement = np.linalg.norm(
            np.array(qrc.NEW_CARTESIAN) - np.array(qrc.CARTESIAN)
        )
        assert displacement > 0

    def test_specific_mode_on_saddle(self, g16_planar_chex, temp_workdir):
        """Test displacement along specific mode on higher-order saddle point."""

        shutil.copy(g16_planar_chex, temp_workdir)
        local_file = temp_workdir / g16_planar_chex.name

        # Displace along mode 1 only
        qrc_mode1 = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="mode1",
            val=None,
            num=1
        )

        # Need to recopy for second test
        shutil.copy(g16_planar_chex, temp_workdir)

        # Displace along mode 3 only
        qrc_mode3 = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="mode3",
            val=None,
            num=3
        )

        # Different modes should give different displacements
        diff = np.linalg.norm(
            np.array(qrc_mode1.NEW_CARTESIAN) - np.array(qrc_mode3.NEW_CARTESIAN)
        )
        assert diff > 0, "Different modes should produce different displacements"

    def test_negative_amplitude(self, g16_claisen_ts, temp_workdir):
        """Test reverse displacement with negative amplitude."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        # Forward displacement
        qrc_fwd = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRCF",
            val=None,
            num=None
        )

        # Need to recopy since file may be modified
        shutil.copy(g16_claisen_ts, temp_workdir)

        # Reverse displacement
        qrc_rev = QRCGenerator(
            file=str(local_file),
            amplitude=-0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRCR",
            val=None,
            num=None
        )

        # Forward and reverse should be different
        diff = np.linalg.norm(
            np.array(qrc_fwd.NEW_CARTESIAN) - np.array(qrc_rev.NEW_CARTESIAN)
        )
        assert diff > 0, "Forward and reverse displacements should differ"

    def test_custom_nproc_and_mem(self, g16_acetaldehyde, temp_workdir):
        """Test that nproc and mem are written to output file."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=8,
            mem="16GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None
        )

        # Check the generated input file has correct settings
        output_file = temp_workdir / f"{local_file.stem}_QRC.com"
        content = output_file.read_text()
        assert "%nproc=8" in content
        assert "%mem=16GB" in content


class TestIntegration:
    """Integration tests for full workflow."""

    def test_no_overlap_warning(self, g16_acetaldehyde, temp_workdir):
        """Test that normal displacements don't cause overlap."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRC",
            val=None,
            num=None
        )

        assert not qrc.OVERLAPPED, "Normal amplitude should not cause overlap"

    def test_large_amplitude_runs(self, g16_claisen_ts, temp_workdir):
        """Test that very large amplitudes still run (may cause overlap)."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=5.0,  # Very large amplitude
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRC_large",
            val=None,
            num=None
        )

        # Test just checks it runs without error
        assert qrc.NEW_CARTESIAN is not None


class TestMain:
    """Tests for main() CLI function."""

    def test_main_no_args(self, monkeypatch, capsys):
        """Test main with no arguments does not crash."""
        monkeypatch.setattr('sys.argv', ['pyqrc'])
        # Should run without error (processes no files)
        main()

    def test_main_with_single_file(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with a single file argument."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
        main()

        # Check output file was created
        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_amplitude_option(self, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with --amp option."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--amp', '0.5', str(local_file)])
        main()

        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_nproc_and_mem(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with --nproc and --mem options."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', [
            'pyqrc', '--nproc', '4', '--mem', '8GB', str(local_file)
        ])
        main()

        output_file = temp_workdir / f"{local_file.stem}_QRC.com"
        assert output_file.exists()
        content = output_file.read_text()
        assert '%nproc=4' in content
        assert '%mem=8GB' in content

    def test_main_with_custom_suffix(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with --name option for custom suffix."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--name', 'CUSTOM', str(local_file)])
        main()

        assert (temp_workdir / f"{local_file.stem}_CUSTOM.com").exists()

    def test_main_with_freq_option(self, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with --freq option to specify frequency value."""

        import cclib

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        # A value within the matching tolerance of the imaginary frequency
        data = cclib.io.ccopen(str(local_file)).parse()
        target = round(float(data.vibfreqs[0]), 1)

        monkeypatch.setattr('sys.argv', ['pyqrc', '-f', str(target), str(local_file)])
        exit_code = main()

        assert exit_code == 0
        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_freqnum_option(self, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with --freqnum option."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--freqnum', '1', str(local_file)])
        main()

        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_custom_route(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with --route option."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', [
            'pyqrc', '--route', 'B3LYP/6-31G* opt freq', str(local_file)
        ])
        main()

        output_file = temp_workdir / f"{local_file.stem}_QRC.com"
        assert output_file.exists()
        content = output_file.read_text()
        assert 'B3LYP/6-31G* opt freq' in content

    def test_main_auto_processes_imaginary(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """Test main with --auto processes files with imaginary frequencies."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--auto', str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'imaginary frequencies: processed' in captured.out


class TestOutputDataExtended:
    """Extended tests for OutputData class."""

    def test_gaussian_termination_normal(self, g16_acetaldehyde):
        """Test Gaussian normal termination is detected."""
        data = OutputData(str(g16_acetaldehyde))
        assert data.TERMINATION == "normal"

    def test_gaussian_jobtype_extraction(self, g16_acetaldehyde):
        """Test Gaussian job type is extracted."""
        data = OutputData(str(g16_acetaldehyde))
        assert data.JOBTYPE is not None

    def test_gaussian_level_of_theory(self, g16_acetaldehyde):
        """Test Gaussian level of theory is extracted."""
        data = OutputData(str(g16_acetaldehyde))
        assert data.LEVELOFTHEORY is not None
        # Should be in format "level/basis"
        assert '/' in data.LEVELOFTHEORY

    def test_orca_jobtype_extraction(self, orca_acetaldehyde):
        """Test ORCA job type is extracted."""
        data = OutputData(str(orca_acetaldehyde))
        assert data.JOBTYPE is not None

    def test_qchem_format_no_termination(self, qchem_acetaldehyde):
        """Test Q-Chem format detection (termination not implemented for Q-Chem)."""
        data = OutputData(str(qchem_acetaldehyde))
        assert data.format == "QChem"
        # Q-Chem termination detection not implemented
        assert data.TERMINATION is None


class TestQRCGeneratorFormats:
    """Tests for QRCGenerator output format handling."""

    def test_orca_output_format(self, orca_acetaldehyde, temp_workdir):
        """Test ORCA input file generation."""

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=2,
            mem="8GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None
        )

        output_file = temp_workdir / f"{local_file.stem}_QRC.inp"
        assert output_file.exists()
        content = output_file.read_text()
        # ORCA format checks
        assert '%pal nprocs 2 end' in content
        assert '%maxcore' in content
        assert '* xyz' in content

    def test_orca_memory_gb_conversion(self, orca_acetaldehyde, temp_workdir):
        """Test ORCA memory is converted from GB to MB."""

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRC",
            val=None,
            num=None
        )

        output_file = temp_workdir / f"{local_file.stem}_QRC.inp"
        content = output_file.read_text()
        # 4GB should be converted to 4096 MB
        assert '%maxcore 4096' in content

    def test_orca_memory_mb(self, orca_acetaldehyde, temp_workdir):
        """Test ORCA memory with MB input."""

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="2000MB",
            route=None,
            verbose=False,
            suffix="QRC_MB",
            val=None,
            num=None
        )

        output_file = temp_workdir / f"{local_file.stem}_QRC_MB.inp"
        content = output_file.read_text()
        # Should keep MB value
        assert '%maxcore 2000' in content

    @QCHEM_DEV_CCLIB_SKIP
    def test_qchem_output_format(self, qchem_acetaldehyde, temp_workdir):
        """Test Q-Chem input file generation."""

        shutil.copy(qchem_acetaldehyde, temp_workdir)
        local_file = temp_workdir / qchem_acetaldehyde.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None
        )

        output_file = temp_workdir / f"{local_file.stem}_QRC.inp"
        assert output_file.exists()
        content = output_file.read_text()
        # Q-Chem format checks
        assert '$molecule' in content
        assert '$end' in content
        assert '$rem' in content


class TestQRCGeneratorCustomRoute:
    """Tests for QRCGenerator with custom route options."""

    def test_custom_route_gaussian(self, g16_acetaldehyde, temp_workdir):
        """Test custom route for Gaussian format."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        custom_route = "M062X/def2-TZVP opt=(calcfc,ts,noeigen)"

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=custom_route,
            verbose=False,
            suffix="QRC_custom",
            val=None,
            num=None
        )

        output_file = temp_workdir / f"{local_file.stem}_QRC_custom.com"
        content = output_file.read_text()
        assert custom_route in content

    def test_custom_route_orca(self, orca_acetaldehyde, temp_workdir):
        """Test custom route for ORCA format."""

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        custom_route = "BP86 def2-SVP TightSCF"

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=custom_route,
            verbose=False,
            suffix="QRC_custom",
            val=None,
            num=None
        )

        output_file = temp_workdir / f"{local_file.stem}_QRC_custom.inp"
        content = output_file.read_text()
        assert custom_route in content


class TestSpecificFrequencyValue:
    """Tests for displacement along specific frequency value."""

    def test_val_parameter(self, g16_claisen_ts, temp_workdir):
        """Test QRCGenerator with val parameter for specific frequency."""

        import cclib

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        # Get actual frequency value from file
        parser = cclib.io.ccopen(str(local_file))
        data = parser.parse()
        target_freq = data.vibfreqs[0]  # First (imaginary) frequency

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.3,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC_val",
            val=target_freq,  # Use exact frequency value
            num=None
        )

        displacement = np.linalg.norm(
            np.array(qrc.NEW_CARTESIAN) - np.array(qrc.CARTESIAN)
        )
        assert displacement > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_element_covalent_radius(self, temp_workdir):
        """Test gen_overlap handles unknown elements gracefully."""
        # Unknown elements should use default radius of 1.5
        atoms = ["C", "Xx"]  # Xx is not in COVALENT_RADII
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        # Should not raise an error
        result = gen_overlap(atoms, coords, 0.8)
        assert result.shape == (2, 2)

    def test_element_id_boundary(self):
        """Test element_id at boundaries."""
        assert element_id(0) == ""  # Index 0 is empty string
        assert element_id(1) == "H"
        assert element_id(118) == "Uuo"  # Last element
        assert element_id(119) == "XX"  # Out of range

    def test_mwdist_with_different_masses(self):
        """Test mwdist properly weights by mass."""
        # Heavy atom should contribute more
        coords1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Two hydrogens
        dist_hh = mwdist(coords1, coords2, [1, 1])

        # Two carbons (heavier)
        dist_cc = mwdist(coords1, coords2, [6, 6])

        # Carbon displacement should be larger due to mass weighting
        assert dist_cc > dist_hh

    def test_verbose_false_no_qrc_file(self, g16_acetaldehyde, temp_workdir):
        """Test that verbose=False does not create .qrc file."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRC_quiet",
            val=None,
            num=None
        )

        # .qrc file should NOT be created when verbose=False
        qrc_file = temp_workdir / f"{local_file.stem}_QRC_quiet.qrc"
        assert not qrc_file.exists()

    def test_positive_freq_mode_displacement(self, g16_acetaldehyde, temp_workdir):
        """Test displacement along a positive frequency mode."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        # Displace along mode 5 (a positive frequency mode)
        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRC",
            val=None,
            num=5
        )

        # With a specific mode number, structure should be displaced
        displacement = np.linalg.norm(
            np.array(qrc.NEW_CARTESIAN) - np.array(qrc.CARTESIAN)
        )
        assert displacement > 0


class TestLevelOfTheory:
    """Tests for _level_of_theory parsing."""

    def test_level_of_theory_gaussian_freq(self, g16_claisen_ts):
        """Test level of theory extraction from Gaussian freq output."""
        data = OutputData(str(g16_claisen_ts))
        lot = data.LEVELOFTHEORY
        assert lot is not None
        assert '/' in lot
        # Should have method and basis set
        parts = lot.split('/')
        assert len(parts) == 2
        assert parts[0] != 'none'

    def test_level_of_theory_orca(self, orca_claisen_ts):
        """Test level of theory extraction from ORCA output."""
        data = OutputData(str(orca_claisen_ts))
        # ORCA uses _level_of_theory internally
        lot = data.LEVELOFTHEORY
        # May or may not be parsed depending on output format
        assert data.format == "ORCA"


class TestMainFileGlobbing:
    """Tests for file globbing in main()."""

    def test_main_with_glob_pattern(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with glob pattern for files."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        # Use glob pattern
        monkeypatch.setattr('sys.argv', ['pyqrc', str(temp_workdir / '*.log')])
        main()

        # Should process the file
        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_multiple_files(self, g16_acetaldehyde, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with multiple file arguments."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file1 = temp_workdir / g16_acetaldehyde.name
        local_file2 = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file1), str(local_file2)])
        main()

        # Both files should be processed
        assert (temp_workdir / f"{local_file1.stem}_QRC.com").exists()
        assert (temp_workdir / f"{local_file2.stem}_QRC.com").exists()


class TestORCAMemoryEdgeCases:
    """Tests for ORCA memory parsing edge cases."""

    def test_orca_memory_no_unit(self, orca_acetaldehyde, temp_workdir):
        """Test ORCA memory with numeric value only."""

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="3000",  # No unit
            route=None,
            verbose=False,
            suffix="QRC_nounit",
            val=None,
            num=None
        )

        output_file = temp_workdir / f"{local_file.stem}_QRC_nounit.inp"
        content = output_file.read_text()
        # Should use the numeric value directly
        assert '%maxcore 3000' in content


class TestPrintOutput:
    """Tests for print output in main()."""

    def test_main_prints_frequency_info(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """Test main prints frequency information."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'imaginary frequencies' in captured.out

    def test_main_prints_freq_value_info(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """Test main prints info when using --freq option."""

        import cclib

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        data = cclib.io.ccopen(str(local_file)).parse()
        target = round(float(data.vibfreqs[0]), 1)

        monkeypatch.setattr('sys.argv', ['pyqrc', '-f', str(target), str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'distorted along' in captured.out
        assert 'cm-1' in captured.out

    def test_main_prints_freqnum_info(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """Test main prints info when using --freqnum option."""

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--freqnum', '2', str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'distorted along freq #2' in captured.out


class TestWorkingDirectory:
    """Tests for working_directory context manager."""

    def test_changes_and_restores_directory(self, tmp_path):
        """Test that working_directory changes to target and restores on exit."""
        original = os.getcwd()
        with working_directory(tmp_path):
            assert os.getcwd() == str(tmp_path)
        assert os.getcwd() == original

    def test_restores_on_exception(self, tmp_path):
        """Test that working_directory restores directory even when an exception occurs."""
        original = os.getcwd()
        with pytest.raises(ValueError):
            with working_directory(tmp_path):
                assert os.getcwd() == str(tmp_path)
                raise ValueError("test error")
        assert os.getcwd() == original


class TestQRCParseError:
    """Tests for QRCParseError exception and error handling."""

    def test_invalid_file_raises_parse_error(self, tmp_path, monkeypatch):
        """Test that a non-chemistry file raises QRCParseError."""
        monkeypatch.chdir(tmp_path)
        bad_file = tmp_path / "garbage.log"
        bad_file.write_text("This is not a chemistry output file.\nJust random text.\n")

        with pytest.raises(QRCParseError, match="Could not determine file format"):
            QRCGenerator(
                file=str(bad_file), amplitude=0.2, nproc=1, mem="4GB",
                route=None, verbose=False, suffix="QRC", val=None, num=None
            )

    def test_corrupt_file_raises_parse_error(self, tmp_path, monkeypatch):
        """Test that a file cclib can't parse raises QRCParseError."""
        monkeypatch.chdir(tmp_path)

        # Mock cclib to raise an exception during parsing
        class FailingParser:
            def parse(self):
                raise RuntimeError("Simulated parse failure")

        with patch('pyqrc.pyQRC.cclib.io.ccopen', return_value=FailingParser()):
            bad_file = tmp_path / "corrupt.log"
            bad_file.write_text("dummy\n")
            with pytest.raises(QRCParseError, match="Failed to parse"):
                QRCGenerator(
                    file=str(bad_file), amplitude=0.2, nproc=1, mem="4GB",
                    route=None, verbose=False, suffix="QRC", val=None, num=None
                )

    def test_main_handles_corrupt_file(self, tmp_path, monkeypatch, capsys):
        """Test main() gracefully handles corrupt files and returns exit code 1."""
        monkeypatch.chdir(tmp_path)
        bad_file = tmp_path / "corrupt.log"
        bad_file.write_text("Not a valid output file\n")

        monkeypatch.setattr('sys.argv', ['pyqrc', str(bad_file)])
        exit_code = main()

        captured = capsys.readouterr()
        assert 'could not be parsed' in captured.out or 'failed to parse' in captured.out
        assert exit_code == 1

    def test_main_handles_parse_exception(self, g16_claisen_ts, tmp_path, monkeypatch, capsys):
        """Test main() catches QRCParseError from QRCGenerator and returns 1."""

        shutil.copy(g16_claisen_ts, tmp_path)
        local_file = tmp_path / g16_claisen_ts.name
        monkeypatch.chdir(tmp_path)

        # Patch QRCGenerator to raise QRCParseError
        with patch('pyqrc.pyQRC.QRCGenerator', side_effect=QRCParseError("test parse failure")):
            monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
            exit_code = main()

        captured = capsys.readouterr()
        assert 'failed' in captured.out
        assert exit_code == 1


class TestMainAutoMode:
    """Tests for main() --auto mode."""

    def test_auto_skips_no_imaginary_freqs(self, temp_workdir, monkeypatch, capsys):
        """Test --auto mode skips files with no imaginary frequencies."""
        # planar_chex_mode1.log has 0 imaginary frequencies (it's an optimized structure)
        mode1_file = Path(__file__).parent.parent / 'examples' / 'g16' / 'planar_chex_mode1.log'

        shutil.copy(mode1_file, temp_workdir)
        local_file = temp_workdir / mode1_file.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--auto', str(local_file)])
        exit_code = main()

        captured = capsys.readouterr()
        assert 'no imaginary frequencies: skipping' in captured.out
        assert exit_code == 0


class TestMainExitCodes:
    """Tests for main() exit code behavior."""

    def test_success_returns_zero(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main() returns 0 on success."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
        assert main() == 0

    def test_no_files_returns_zero(self, monkeypatch):
        """Test main() returns 0 when no files given."""
        monkeypatch.setattr('sys.argv', ['pyqrc'])
        assert main() == 0

    def test_main_quiet_flag(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test that -q/--quiet suppresses verbose output."""

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '-q', str(local_file)])
        main()

        # Verbose .qrc file should NOT be created
        qrc_file = temp_workdir / f"{local_file.stem}_QRC.qrc"
        assert not qrc_file.exists()


class TestLevelOfTheoryBranches:
    """Tests for _level_of_theory() parsing branches."""

    def test_external_calculation(self, tmp_path):
        """Test level of theory with External calculation."""
        output_file = tmp_path / "external.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " External calculation\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert lot == "ext/ext"

    def test_pipe_freq_format(self, tmp_path):
        """Test level of theory with pipe-delimited Freq format."""
        output_file = tmp_path / "pipe_freq.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " 1|1|GINC-NODE|Freq|RB3LYP|6-31G(d)|C2H4O|USER|01-Jan-2025|0||\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert "B3LYP" in lot
        assert "6-31G(d)" in lot

    def test_backslash_sp_format(self, tmp_path):
        """Test level of theory with backslash-delimited SP format."""
        output_file = tmp_path / "sp.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " 1\\1\\GINC-NODE\\SP\\RB3LYP\\6-31G(d)\\C2H4O\\USER\\01-Jan-2025\\0\\\\\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert "B3LYP" in lot
        assert "6-31G(d)" in lot

    def test_pipe_sp_format(self, tmp_path):
        """Test level of theory with pipe-delimited SP format."""
        output_file = tmp_path / "pipe_sp.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " 1|1|GINC-NODE|SP|UM062X|def2TZVP|C2H4O|USER|01-Jan-2025|0||\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert "M062X" in lot
        assert "def2TZVP" in lot

    def test_dlpno_ccsd_detection(self, tmp_path):
        """Test DLPNO-CCSD(T) level of theory detection."""
        output_file = tmp_path / "dlpno.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " DLPNO BASED TRIPLES CORRECTION\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert "DLPNO-CCSD(T)" in lot

    def test_cbs_extrapolation(self, tmp_path):
        """Test CBS extrapolation basis set detection."""
        output_file = tmp_path / "cbs.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " Estimated CBS total energy blah -500.12345\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert "Extrapol." in lot

    def test_cbs_index_error(self, tmp_path):
        """Test _level_of_theory handles IndexError in CBS extrapolation line."""
        output_file = tmp_path / "bad_cbs.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " Estimated CBS total energy\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        # Should not crash
        assert "none" in lot

    def test_unrestricted_label_removal(self, tmp_path):
        """Test that U prefix is removed from unrestricted level of theory."""
        output_file = tmp_path / "unrestricted.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " 1\\1\\GINC-NODE\\Freq\\UB3LYP\\6-31G(d)\\C2H4O\\USER\\01-Jan\\0\\\\\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert lot == "B3LYP/6-31G(d)"

    def test_no_theory_found(self, tmp_path):
        """Test _level_of_theory returns none/none when no theory in file."""
        output_file = tmp_path / "no_theory.log"
        output_file.write_text(
            " Gaussian, Inc.\n"
            " Just some random output\n"
        )
        data = OutputData(str(output_file))
        lot = data._level_of_theory()
        assert lot == "none/none"


class TestG16Opt:
    """Tests for g16_opt function."""

    def test_g16_opt_calls_subprocess(self, tmp_path):
        """Test g16_opt calls subprocess with correct arguments."""
        with patch('pyqrc.pyQRC.subprocess.run') as mock_run:
            g16_opt("test.com")
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[1] == "test.com"
            assert "run_g16.sh" in args[0]


class TestRunIRC:
    """Tests for run_irc function."""

    def test_run_irc_no_overlap(self, g16_claisen_ts, tmp_path, monkeypatch):
        """Test run_irc creates QRC and calls g16_opt when no overlap."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_claisen_ts, tmp_path)
        local_file = tmp_path / g16_claisen_ts.name

        from argparse import Namespace
        options = Namespace(nproc=1, mem="4GB", verbose=False)

        with patch('pyqrc.pyQRC.g16_opt') as mock_g16:
            log = Logger("test", "dat", "irc")
            run_irc(str(local_file), options, 1, 0.2, None, "test_suffix", log)
            mock_g16.assert_called_once()
            log.close()

    def test_run_irc_with_overlap(self, g16_claisen_ts, tmp_path, monkeypatch):
        """Test run_irc skips g16_opt when atoms overlap."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_claisen_ts, tmp_path)
        local_file = tmp_path / g16_claisen_ts.name

        from argparse import Namespace
        options = Namespace(nproc=1, mem="4GB", verbose=False)

        with patch('pyqrc.pyQRC.g16_opt') as mock_g16:
            with patch('pyqrc.pyQRC.check_overlap', return_value=True):
                log = Logger("test", "dat", "irc2")
                run_irc(str(local_file), options, 1, 0.2, None, "test_overlap", log)
                mock_g16.assert_not_called()
                log.close()


class TestMainNoFreqInfo:
    """Tests for main() with files that have no frequency information."""

    def test_main_no_vibfreqs(self, tmp_path, monkeypatch, capsys):
        """Test main skips files with no frequency data."""
        monkeypatch.chdir(tmp_path)

        # Create a minimal Gaussian output that cclib can parse but has no freqs
        sp_file = tmp_path / "sp_only.log"
        # Use a real Gaussian file but remove freq data by mocking
        sp_file.write_text(
            " Gaussian, Inc.\n"
            " Normal termination\n"
        )

        # Mock cclib to return data without vibfreqs
        class MockData:
            pass

        class MockParser:
            def parse(self):
                return MockData()

        with patch('pyqrc.pyQRC.cclib.io.ccopen', return_value=MockParser()):
            monkeypatch.setattr('sys.argv', ['pyqrc', str(sp_file)])
            exit_code = main()

        captured = capsys.readouterr()
        assert 'no frequency information' in captured.out
        assert exit_code == 0


class TestUnknownFormat:
    """Tests for handling unknown file formats."""

    def test_unknown_format_defaults_to_com(self, g16_acetaldehyde, tmp_path, monkeypatch):
        """Test that unknown format falls back to .com extension."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_acetaldehyde, tmp_path)
        local_file = tmp_path / g16_acetaldehyde.name

        # Mock metadata to return an unknown format
        with patch('pyqrc.pyQRC.OutputData') as mock_od:
            mock_od_instance = mock_od.return_value
            mock_od_instance.format = "UnknownPackage"
            mock_od_instance.JOBTYPE = "opt freq"

            qrc = QRCGenerator(
                file=str(local_file), amplitude=0.2, nproc=1, mem="4GB",
                route="opt freq", verbose=False, suffix="QRC_unk", val=None, num=None
            )

        # Should default to .com extension
        assert (tmp_path / f"{local_file.stem}_QRC_unk.com").exists()


class TestMultiplicityFallback:
    """Tests for multiplicity fallback behavior."""

    def test_missing_multiplicity_defaults_to_one(self, g16_acetaldehyde, tmp_path, monkeypatch, capsys):
        """Test that missing multiplicity defaults to 1 with a warning."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_acetaldehyde, tmp_path)
        local_file = tmp_path / g16_acetaldehyde.name

        import cclib

        # Wrap cclib to return data without mult attribute
        original_ccopen = cclib.io.ccopen

        class MultlessData:
            """Wrapper that removes mult attribute from parsed data."""
            def __init__(self, data):
                self._data = data

            def __getattr__(self, name):
                if name == 'mult':
                    raise AttributeError("no mult")
                return getattr(self._data, name)

        class MultlessParser:
            def __init__(self, parser):
                self._parser = parser

            def parse(self):
                return MultlessData(self._parser.parse())

        def mock_ccopen(f, *a, **kw):
            return MultlessParser(original_ccopen(f, *a, **kw))

        with patch('pyqrc.pyQRC.cclib.io.ccopen', side_effect=mock_ccopen):
            QRCGenerator(
                file=str(local_file), amplitude=0.2, nproc=1, mem="4GB",
                route=None, verbose=False, suffix="QRC_mult", val=None, num=None
            )

        captured = capsys.readouterr()
        assert 'multiplicity not parsed' in captured.out


class TestFormatTypeFallback:
    """Tests for format_type fallback when metadata is missing."""

    def test_no_metadata_uses_outputdata_format(self, g16_acetaldehyde, tmp_path, monkeypatch):
        """Test that missing metadata falls back to OutputData format detection."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_acetaldehyde, tmp_path)
        local_file = tmp_path / g16_acetaldehyde.name

        import cclib

        # Wrap cclib to return data without metadata attribute
        original_ccopen = cclib.io.ccopen

        class NoMetadataData:
            def __init__(self, data):
                self._data = data

            def __getattr__(self, name):
                if name == 'metadata':
                    raise AttributeError("no metadata")
                return getattr(self._data, name)

        class NoMetadataParser:
            def __init__(self, parser):
                self._parser = parser

            def parse(self):
                return NoMetadataData(self._parser.parse())

        def mock_ccopen(f, *a, **kw):
            return NoMetadataParser(original_ccopen(f, *a, **kw))

        with patch('pyqrc.pyQRC.cclib.io.ccopen', side_effect=mock_ccopen):
            qrc = QRCGenerator(
                file=str(local_file), amplitude=0.2, nproc=1, mem="4GB",
                route=None, verbose=False, suffix="QRC_nmd", val=None, num=None
            )

        # Should still create a .com file (Gaussian detected via OutputData)
        assert (tmp_path / f"{local_file.stem}_QRC_nmd.com").exists()


class TestMainCclibException:
    """Tests for main() handling cclib exceptions."""

    def test_main_cclib_parse_exception(self, g16_acetaldehyde, tmp_path, monkeypatch, capsys):
        """Test main() handles cclib throwing an exception during parsing."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_acetaldehyde, tmp_path)
        local_file = tmp_path / g16_acetaldehyde.name

        class FailParser:
            def parse(self):
                raise RuntimeError("cclib internal error")

        with patch('pyqrc.pyQRC.cclib.io.ccopen', return_value=FailParser()):
            monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
            exit_code = main()

        captured = capsys.readouterr()
        assert 'failed to parse' in captured.out
        assert exit_code == 1

    def test_main_cclib_returns_none(self, tmp_path, monkeypatch, capsys):
        """Test main() handles cclib returning None for unknown format."""
        monkeypatch.chdir(tmp_path)
        unknown_file = tmp_path / "unknown.log"
        unknown_file.write_text("not a chemistry file\n")

        with patch('pyqrc.pyQRC.cclib.io.ccopen', return_value=None):
            monkeypatch.setattr('sys.argv', ['pyqrc', str(unknown_file)])
            exit_code = main()

        captured = capsys.readouterr()
        assert 'could not be parsed' in captured.out
        assert exit_code == 1


class TestUnknownFormatFallback:
    """Tests for unknown format fallback to an .xyz file."""

    def test_unknown_format_writes_xyz(self, g16_acetaldehyde, tmp_path, monkeypatch):
        """Test that an unrecognized format falls back to an .xyz file."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_acetaldehyde, tmp_path)
        local_file = tmp_path / g16_acetaldehyde.name

        import cclib
        original_ccopen = cclib.io.ccopen

        class NoPackageMetadataData:
            """Wrapper that strips package from metadata."""
            def __init__(self, data):
                self._data = data
                # Override metadata to have no package
                self.metadata = {k: v for k, v in getattr(data, 'metadata', {}).items()
                                 if k != 'package'}

            def __getattr__(self, name):
                return getattr(self._data, name)

        class WrapperParser:
            def __init__(self, parser):
                self._parser = parser

            def parse(self):
                return NoPackageMetadataData(self._parser.parse())

        def mock_ccopen(f, *a, **kw):
            return WrapperParser(original_ccopen(f, *a, **kw))

        # Patch both cclib metadata (to remove package) and OutputData (to report unknown)
        with patch('pyqrc.pyQRC.cclib.io.ccopen', side_effect=mock_ccopen):
            with patch('pyqrc.pyQRC.OutputData') as mock_od:
                mock_od.return_value.format = "UnknownSoftware"
                mock_od.return_value.JOBTYPE = "opt freq"

                qrc = QRCGenerator(
                    file=str(local_file), amplitude=0.2, nproc=1, mem="4GB",
                    route="opt freq", verbose=False, suffix="QRC_uf", val=None, num=None
                )

        # Unknown format should fall back to an .xyz file, not a malformed .com
        assert not (tmp_path / f"{local_file.stem}_QRC_uf.com").exists()
        lines = (tmp_path / f"{local_file.stem}_QRC_uf.xyz").read_text().splitlines()
        assert int(lines[0]) == qrc.NATOMS
        assert len(lines) == qrc.NATOMS + 2


class TestQcoordMode:
    """Tests for --qcoord mode in main()."""

    def test_qcoord_creates_directories_and_runs(self, g16_claisen_ts, tmp_path, monkeypatch, capsys):
        """Test --qcoord mode creates directory structure and runs calculations."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_claisen_ts, tmp_path)
        local_file = tmp_path / g16_claisen_ts.name

        # Mock g16_opt to avoid actually running Gaussian
        with patch('pyqrc.pyQRC.g16_opt'):
            monkeypatch.setattr('sys.argv', [
                'pyqrc', '--qcoord', '--nummodes', '1', str(local_file)
            ])
            exit_code = main()

        # Should create parent directory named after the file stem
        parent_dir = tmp_path / local_file.stem
        assert parent_dir.exists()

        # Should create num_1 subdirectory
        assert (parent_dir / 'num_1').exists()

        # Should create the RUNIRC log
        assert (tmp_path / 'RUNIRC_1.dat').exists()

        assert exit_code == 0

        # Deprecated mode warns once
        assert '--qcoord is deprecated' in capsys.readouterr().out

    def test_default_mode_no_deprecation_warning(
        self, g16_claisen_ts, tmp_path, monkeypatch, capsys
    ):
        """The default (non --qcoord) mode prints no deprecation warning."""
        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_claisen_ts, tmp_path)
        local_file = tmp_path / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
        assert main() == 0
        assert 'deprecated' not in capsys.readouterr().out

    def test_qcoord_limited_modes(self, g16_claisen_ts, tmp_path, monkeypatch):
        """Test --qcoord with limited nummodes creates only specified directories."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_claisen_ts, tmp_path)
        local_file = tmp_path / g16_claisen_ts.name

        # Mock g16_opt to avoid actually running Gaussian
        with patch('pyqrc.pyQRC.g16_opt'):
            monkeypatch.setattr('sys.argv', [
                'pyqrc', '--qcoord', '--nummodes', '2', str(local_file)
            ])
            main()

        parent_dir = tmp_path / local_file.stem
        assert (parent_dir / 'num_1').exists()
        assert (parent_dir / 'num_2').exists()

    def test_qcoord_all_modes_default(self, g16_claisen_ts, tmp_path, monkeypatch):
        """Test --qcoord with default nummodes='all' covers all modes."""

        monkeypatch.chdir(tmp_path)
        shutil.copy(g16_claisen_ts, tmp_path)
        local_file = tmp_path / g16_claisen_ts.name

        # Mock run_irc to avoid slow QRC generation for all 36 modes
        with patch('pyqrc.pyQRC.run_irc'):
            monkeypatch.setattr('sys.argv', [
                'pyqrc', '--qcoord', str(local_file)
            ])
            main()

        # With 'all' modes, should create directories for all 36 modes
        parent_dir = tmp_path / local_file.stem
        assert (parent_dir / 'num_1').exists()
        assert (parent_dir / 'num_36').exists()

    def test_qcoord_no_imaginary_logs_stability_check(
        self, tmp_path, monkeypatch, capsys
    ):
        """Test --qcoord with no imaginary freqs logs stability check message."""
        # Use planar_chex_mode1.log which has 0 imaginary frequencies
        mode1_file = Path(__file__).parent.parent / 'examples' / 'g16' / 'planar_chex_mode1.log'

        monkeypatch.chdir(tmp_path)
        shutil.copy(mode1_file, tmp_path)
        local_file = tmp_path / mode1_file.name

        with patch('pyqrc.pyQRC.g16_opt'):
            monkeypatch.setattr('sys.argv', [
                'pyqrc', '--qcoord', '--nummodes', '1', str(local_file)
            ])
            main()

        # Check the RUNIRC log for stability message
        log_file = tmp_path / 'RUNIRC_1.dat'
        assert log_file.exists()
        log_content = log_file.read_text()
        assert 'no imaginary frequencies: check for stability' in log_content


class TestCLIFailureModes:
    """CLI failure behavior: bad inputs must fail loudly (ROADMAP 0.3/1.2/1.3)."""

    def test_missing_file_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        """A nonexistent input file should produce an error message and exit 1."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr('sys.argv', ['pyqrc', str(tmp_path / 'does_not_exist.log')])
        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 1
        assert 'no such file' in captured.out

    def test_unmatched_freq_errors_and_writes_nothing(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """--freq matching no normal mode should exit 1 and write no input file."""
        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        # -123.45 cm-1 is not within tolerance of any mode in this file
        monkeypatch.setattr('sys.argv', ['pyqrc', '-f', '-123.45', str(local_file)])
        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 1
        assert 'failed' in captured.out
        assert not (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_out_of_range_freqnum_errors_and_writes_nothing(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """--freqnum beyond the number of modes should exit 1 and write no input file."""
        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--freqnum', '99', str(local_file)])
        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 1
        assert 'failed' in captured.out
        assert not (temp_workdir / f"{local_file.stem}_QRC.com").exists()


class TestResolveTargetModes:
    """Unit tests for QRCGenerator._resolve_target_modes."""

    FREQS = np.array([-536.5, 102.3, 250.0, 1700.8])

    def test_default_selects_imaginary_modes(self):
        modes = QRCGenerator._resolve_target_modes(self.FREQS, None, None)
        assert modes == {0}

    def test_freq_nearest_match_within_tolerance(self):
        """A value within 1 cm-1 of a mode matches that mode."""
        modes = QRCGenerator._resolve_target_modes(self.FREQS, -536.0, None)
        assert modes == {0}

    def test_freq_no_match_raises(self):
        with pytest.raises(QRCModeError, match="nearest"):
            QRCGenerator._resolve_target_modes(self.FREQS, -500.0, None)

    def test_freqnum_valid(self):
        modes = QRCGenerator._resolve_target_modes(self.FREQS, None, 3)
        assert modes == {2}

    def test_freqnum_out_of_range_raises(self):
        with pytest.raises(QRCModeError, match="out of range"):
            QRCGenerator._resolve_target_modes(self.FREQS, None, 99)

    def test_freqnum_zero_raises(self):
        with pytest.raises(QRCModeError, match="out of range"):
            QRCGenerator._resolve_target_modes(self.FREQS, None, 0)

    def test_empty_freq_with_val_raises(self):
        with pytest.raises(QRCModeError, match="no vibrational modes are available"):
            QRCGenerator._resolve_target_modes(np.array([]), -500.0, None)


class TestComputeOnly:
    """Tests for write=False library use (ROADMAP 2.2)."""

    def test_compute_without_writing_files(self, g16_claisen_ts, temp_workdir):
        """write=False computes the displaced geometry with zero files written."""
        qrc = QRCGenerator(
            file=str(g16_claisen_ts), amplitude=0.2, nproc=1, mem="4GB",
            route=None, verbose=True, suffix="QRC", val=None, num=None,
            write=False
        )

        displacement = np.linalg.norm(
            np.array(qrc.NEW_CARTESIAN) - np.array(qrc.CARTESIAN)
        )
        assert displacement > 0
        assert qrc.OVERLAPPED is not None
        assert qrc.MW_DISTANCE > 0
        # No files created in the working directory (even with verbose=True)
        assert list(temp_workdir.iterdir()) == []

    def test_compute_does_not_mutate_original(self, g16_claisen_ts, temp_workdir):
        """CARTESIAN keeps the parsed geometry; displacement goes to a copy."""
        qrc = QRCGenerator(
            file=str(g16_claisen_ts), amplitude=0.2, nproc=1, mem="4GB",
            route=None, verbose=False, suffix="QRC", val=None, num=None,
            write=False
        )

        original = qrc.CARTESIAN.copy()
        qrc.compute_displacement()
        assert np.array_equal(qrc.CARTESIAN, original)
        assert not np.array_equal(qrc.NEW_CARTESIAN, qrc.CARTESIAN)


class TestQChemMissingMetadata:
    """Q-Chem inputs must never be written with METHOD None/BASIS None.

    These cover the fallback used when the output does not echo the input.
    """

    @QCHEM_DEV_CCLIB_SKIP
    def test_missing_method_raises_before_writing(
        self, qchem_acetaldehyde, temp_workdir
    ):
        """Missing functional metadata raises QRCParseError, writes nothing."""
        shutil.copy(qchem_acetaldehyde, temp_workdir)
        local_file = temp_workdir / qchem_acetaldehyde.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None,
            write=False,
        )
        qrc._func = None

        # Without the echoed input, cclib metadata is the only source
        with patch('pyqrc.pyQRC.read_qchem_input', return_value=None):
            with pytest.raises(QRCParseError, match="METHOD None"):
                qrc.write_files()

        assert not (temp_workdir / f"{local_file.stem}_QRC.inp").exists()
        assert not (temp_workdir / f"{local_file.stem}_QRC.qrc").exists()

    @QCHEM_DEV_CCLIB_SKIP
    def test_missing_basis_raises_before_writing(
        self, qchem_acetaldehyde, temp_workdir
    ):
        """Missing basis-set metadata raises QRCParseError, writes nothing."""
        shutil.copy(qchem_acetaldehyde, temp_workdir)
        local_file = temp_workdir / qchem_acetaldehyde.name

        qrc = QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None,
            write=False,
        )
        qrc._basis = None

        # Without the echoed input, cclib metadata is the only source
        with patch('pyqrc.pyQRC.read_qchem_input', return_value=None):
            with pytest.raises(QRCParseError, match="basis"):
                qrc.write_files()

        assert not (temp_workdir / f"{local_file.stem}_QRC.inp").exists()


class TestAbnormalTermination:
    """Gaussian outputs without normal termination produce a warning."""

    def test_abnormal_termination_warns_but_writes(
        self, g16_acetaldehyde, temp_workdir, capsys
    ):
        """Stripped 'Normal termination' line warns; input is still written."""
        local_file = temp_workdir / g16_acetaldehyde.name
        lines = g16_acetaldehyde.read_text().splitlines(keepends=True)
        local_file.write_text(
            ''.join(line for line in lines if 'Normal termination' not in line)
        )

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None,
        )

        assert 'did not terminate normally' in capsys.readouterr().out
        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_normal_termination_no_warning(
        self, g16_acetaldehyde, temp_workdir, capsys
    ):
        """A normally terminated output produces no termination warning."""
        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        QRCGenerator(
            file=str(local_file),
            amplitude=0.2,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=True,
            suffix="QRC",
            val=None,
            num=None,
        )

        assert 'did not terminate normally' not in capsys.readouterr().out


class TestMainModule:
    """Tests for __main__.py entry point."""

    def test_main_module_can_be_imported(self):
        """Test that __main__.py module can be imported without errors."""
        import importlib
        mod = importlib.import_module('pyqrc.__main__')
        assert hasattr(mod, 'main')


class TestAseMlipBridge:
    """Tests for the examples/ase_mlip/ase2gaussian.py helper."""

    @pytest.fixture
    def ase2gaussian(self):
        """Import the example helper module from the examples directory."""
        pytest.importorskip("ase")
        import importlib.util
        from tests.conftest import EXAMPLES_PATH
        spec = importlib.util.spec_from_file_location(
            "ase2gaussian", EXAMPLES_PATH / "ase_mlip" / "ase2gaussian.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def fake_ts(self, ase2gaussian, temp_workdir):
        """Write a synthetic 'transition state' log and return its pieces."""
        from ase import Atoms
        atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.119],
                                        [0.0, 0.763, -0.477],
                                        [0.0, -0.763, -0.477]])
        freqs = np.array([-500.0, 1600.0, 3700.0])
        rng = np.random.default_rng(42)
        modes = rng.normal(size=(3, 3, 3))
        logfile = temp_workdir / "fake_ts.log"
        ase2gaussian.write_gaussian_freq_log(
            str(logfile), atoms, freqs, modes, energy=-76.4)
        norms = np.linalg.norm(modes.reshape(3, -1), axis=1)
        return logfile, atoms, freqs, modes / norms[:, None, None]

    def test_cclib_roundtrip(self, fake_ts):
        """cclib must recover the geometry, frequencies, and modes."""
        import cclib
        logfile, atoms, freqs, modes = fake_ts
        data = cclib.io.ccread(str(logfile))
        assert data.natom == 3
        np.testing.assert_allclose(data.atomcoords[-1], atoms.get_positions(),
                                   atol=1e-6)
        np.testing.assert_allclose(data.vibfreqs, freqs, atol=1e-4)
        np.testing.assert_allclose(data.vibdisps, modes, atol=0.006)
        np.testing.assert_allclose(data.scfenergies[-1] / 27.211386245988,
                                   -76.4, atol=1e-6)

    def test_qrc_displaces_along_imaginary_mode(self, fake_ts, temp_workdir):
        """QRCGenerator must displace the synthetic log along its imaginary mode."""
        logfile, atoms, freqs, modes = fake_ts
        qrc = QRCGenerator(
            file=str(logfile),
            amplitude=0.3,
            nproc=1,
            mem="4GB",
            route=None,
            verbose=False,
            suffix="QRC",
            val=None,
            num=None,
        )
        displacement = np.asarray(qrc.NEW_CARTESIAN) - np.asarray(qrc.CARTESIAN)
        norm = np.linalg.norm(displacement)
        assert norm > 0.01, "geometry was not displaced"
        cosine = np.dot(displacement.ravel(), modes[0].ravel()) / norm
        assert abs(cosine) > 0.99, "displacement is not along the imaginary mode"
        assert (temp_workdir / "fake_ts_QRC.com").exists()

    def test_extract_vibrations(self, ase2gaussian):
        """extract_vibrations must sign imaginary modes and drop trans/rot."""
        class FakeVibrations:
            def get_energies(self):
                # 0.062 eV ~ 500 cm-1; tiny values are trans/rot noise
                return np.array([0.0620j, 1e-6j, 2e-6, 0.1984, 0.4587])

            def get_mode(self, i):
                mode = np.zeros((2, 3))
                mode[0, 0] = i + 1.0
                return mode

        freqs, modes = ase2gaussian.extract_vibrations(FakeVibrations())
        assert freqs[0] < -400, "imaginary mode should be negative"
        assert len(freqs) == 3, "near-zero trans/rot modes should be dropped"
        assert modes.shape == (3, 2, 3)
        np.testing.assert_allclose(modes[:, 0, 0], [1.0, 4.0, 5.0])


# --- Minimization routes, carried-over input, memory and CLI conveniences ---

from pyqrc.pyQRC import (  # noqa: E402  pylint: disable=wrong-import-position
    DEFAULT_AMPLITUDE,
    gaussian_route_line,
    memory_to_mb,
    minimization_route_gaussian,
    minimization_route_orca,
    read_gaussian_input_tail,
    read_orca_input,
    read_qchem_input,
)

GENECP_ROUTE = '#p opt=(ts,calcfc,noeigentest,modredundant) freq b3lyp/genecp guess=read geom=connectivity'


def _make_gaussian_pair(g16_claisen_ts, directory, route=GENECP_ROUTE, tail=True, stem='ts'):
    """Write a Claisen TS log with a modified route and a matching .gjf input."""
    original = ' # opt(ts,calcfc,noeigentest) freq=noraman wb97xd/6-31+G*'
    log_text = Path(g16_claisen_ts).read_text()
    assert original in log_text
    log = directory / f'{stem}.log'
    log.write_text(log_text.replace(original, ' ' + route, 1))
    if tail:
        com = Path(str(g16_claisen_ts).replace('.log', '.com')).read_text().splitlines()
        com = [route if line.startswith('#') else line for line in com]
        text = '\n'.join(com).rstrip('\n')
        text += '\n\n 1 2 1.0\n 2\n 3\n\nB 1 4 F\n\nC H O 0\n6-31G(d)\n****\n\n'
        text += '--Link1--\n%chk=x\n# freq geom=check\n\nnext job\n\n0 1\n\n'
        (directory / f'{stem}.gjf').write_text(text)
    return log


class TestMemory:
    """Memory strings are parsed consistently and ORCA gets memory per core."""

    @pytest.mark.parametrize('mem,expected', [
        ('8GB', 8192), ('8gb', 8192), ('4000MB', 4000), ('1.5GB', 1536),
        ('3000', 3000), ('1GW', 8192), (' 2 GB ', 2048),
    ])
    def test_memory_to_mb(self, mem, expected):
        assert memory_to_mb(mem) == expected

    @pytest.mark.parametrize('mem', ['lots', '8XB', '', 'GB'])
    def test_invalid_memory_raises(self, mem):
        with pytest.raises(ValueError):
            memory_to_mb(mem)

    def test_orca_maxcore_is_per_core(self, orca_acetaldehyde, temp_workdir):
        """--mem is the total: 8GB over 4 cores is %maxcore 2048."""
        QRCGenerator(str(orca_acetaldehyde), 0.3, 4, '8GB', None, False, 'QRC', None, None)
        assert '%maxcore 2048' in (temp_workdir / 'acetaldehyde_QRC.inp').read_text()

    def test_cli_rejects_bad_memory(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_acetaldehyde), '--mem', 'lots'])
        with pytest.raises(SystemExit):
            main()


class TestMinimizationRoutes:
    """Cloned routes are changed to optimize to a minimum."""

    @pytest.mark.parametrize('route,expected', [
        ('opt(ts,calcfc,noeigentest) freq=noraman wb97xd/6-31+G*', 'opt=calcfc freq=noraman wb97xd/6-31+G*'),
        ('p Opt=(TS,NoEigenTest) b3lyp/6-31g(d)', 'p Opt b3lyp/6-31g(d)'),
        ('opt=(saddle=2,maxcycles=50) m062x/def2svp', 'opt=maxcycles=50 m062x/def2svp'),
        ('opt=(readfc,ts) freq b3lyp/gen guess=read', 'opt freq b3lyp/gen'),
        ('opt freq m062x/6-31g(d)', 'opt freq m062x/6-31g(d)'),
        ('freq b3lyp/6-31g(d)', 'opt freq b3lyp/6-31g(d)'),
        ('p freq b3lyp/6-31g(d) scrf=(smd,solvent=water)', 'p opt freq b3lyp/6-31g(d) scrf=(smd,solvent=water)'),
        ('opt=(ts,modredundant) guess=(read,mix)', 'opt=modredundant guess=mix'),
    ])
    def test_gaussian(self, route, expected):
        assert minimization_route_gaussian(route) == expected

    @pytest.mark.parametrize('route,expected', [
        ('wB97X-D3 def2-SVP OptTS Freq', 'wB97X-D3 def2-SVP Opt Freq'),
        ('optts wb97x-d3 def2-svp rijcosx freq', 'Opt wb97x-d3 def2-svp rijcosx freq'),
        ('M062X def2-TZVP Opt Freq', 'M062X def2-TZVP Opt Freq'),
        ('r2SCAN-3c TightOpt', 'r2SCAN-3c TightOpt'),
        ('B3LYP def2-SVP Freq', 'B3LYP def2-SVP Freq Opt'),
    ])
    def test_orca(self, route, expected):
        assert minimization_route_orca(route) == expected

    @pytest.mark.parametrize('route,expected', [
        (' opt freq', '# opt freq'), ('p opt', '#p opt'), ('#p opt', '#p opt'),
        ('# opt', '# opt'), ('b3lyp/6-31g(d) opt', '# b3lyp/6-31g(d) opt'),
    ])
    def test_gaussian_route_line(self, route, expected):
        assert gaussian_route_line(route) == expected

    def test_cloned_gaussian_ts_route_becomes_opt(self, g16_claisen_ts, temp_workdir, capsys):
        QRCGenerator(str(g16_claisen_ts), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        content = (temp_workdir / 'claisen_ts_QRC.com').read_text()
        assert '# opt=calcfc freq=noraman wb97xd/6-31+G*' in content
        assert 'ts' not in content.splitlines()[3].lower()
        assert 'route changed to optimize to a minimum' in capsys.readouterr().out

    def test_cloned_orca_ts_keywords_become_opt(self, orca_claisen_ts, temp_workdir):
        QRCGenerator(str(orca_claisen_ts), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        first = (temp_workdir / 'claisen_ts_QRC.inp').read_text().splitlines()[0]
        assert first == '! wB97X-D3 def2-SVP Opt Freq'

    def test_user_route_used_verbatim(self, g16_claisen_ts, temp_workdir):
        """A --route is never modified, and a leading '#' is not doubled."""
        QRCGenerator(str(g16_claisen_ts), 0.3, 1, '4GB', '#p opt=(ts,calcfc) b3lyp/6-31g(d)',
                     False, 'QRC', None, None)
        lines = (temp_workdir / 'claisen_ts_QRC.com').read_text().splitlines()
        assert lines[3] == '#p opt=(ts,calcfc) b3lyp/6-31g(d)'


class TestGaussianInputTail:
    """Input after the geometry is taken from the original .com/.gjf."""

    def test_read_tail(self, g16_claisen_ts, tmp_path):
        log = _make_gaussian_pair(g16_claisen_ts, tmp_path)
        input_file, route, tail = read_gaussian_input_tail(str(log))
        assert input_file.endswith('ts.gjf')
        assert 'genecp' in route
        # Connectivity section dropped (geom= is not carried over), Link1 job ignored
        assert tail == 'B 1 4 F\n\nC H O 0\n6-31G(d)\n****'

    def test_no_input_file(self, g16_claisen_ts, tmp_path):
        log = _make_gaussian_pair(g16_claisen_ts, tmp_path, tail=False)
        assert read_gaussian_input_tail(str(log)) == (None, None, None)

    def test_zmatrix_input_not_copied(self, tmp_path):
        (tmp_path / 'z.log').write_text('')
        (tmp_path / 'z.com').write_text(
            '# opt b3lyp/gen\n\nt\n\n0 1\nO\nH 1 R\nH 1 R 2 A\n\n'
            'R 0.96\nA 104.5\n\nH O 0\n6-31G\n****\n\n'
        )
        input_file, _, tail = read_gaussian_input_tail(str(tmp_path / 'z.log'))
        assert input_file is not None and tail is None

    def test_tail_written_after_geometry(self, g16_claisen_ts, temp_workdir, capsys):
        log = _make_gaussian_pair(g16_claisen_ts, temp_workdir)
        QRCGenerator(str(log), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        content = (temp_workdir / 'ts_QRC.com').read_text()
        assert '#p opt=(calcfc,modredundant) freq b3lyp/genecp\n' in content
        assert 'guess' not in content and 'geom' not in content
        assert content.endswith('\n\nB 1 4 F\n\nC H O 0\n6-31G(d)\n****\n\n')
        assert '1 2 1.0' not in content and 'Link1' not in content
        assert 'copied from' in capsys.readouterr().out

    def test_missing_input_warns(self, g16_claisen_ts, temp_workdir, capsys):
        log = _make_gaussian_pair(g16_claisen_ts, temp_workdir, tail=False)
        QRCGenerator(str(log), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        assert 'add it to the new input file by hand' in capsys.readouterr().out
        assert (temp_workdir / 'ts_QRC.com').read_text().endswith('\n\n')

    def test_route_mismatch_not_copied(self, g16_claisen_ts, temp_workdir, capsys):
        log = _make_gaussian_pair(g16_claisen_ts, temp_workdir)
        gjf = temp_workdir / 'ts.gjf'
        gjf.write_text(gjf.read_text().replace('b3lyp/genecp', 'pbe0/genecp', 1))
        QRCGenerator(str(log), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        assert 'B 1 4 F' not in (temp_workdir / 'ts_QRC.com').read_text()
        assert 'different route' in capsys.readouterr().out

    def test_user_route_without_extra_input_skips_tail(self, g16_claisen_ts, temp_workdir):
        log = _make_gaussian_pair(g16_claisen_ts, temp_workdir)
        QRCGenerator(str(log), 0.3, 1, '4GB', 'opt b3lyp/6-31g(d)', False, 'QRC', None, None)
        assert '****' not in (temp_workdir / 'ts_QRC.com').read_text()

    def test_user_route_with_gen_copies_tail(self, g16_claisen_ts, temp_workdir):
        log = _make_gaussian_pair(g16_claisen_ts, temp_workdir)
        QRCGenerator(str(log), 0.3, 1, '4GB', 'opt b3lyp/gen', False, 'QRC', None, None)
        assert '6-31G(d)\n****' in (temp_workdir / 'ts_QRC.com').read_text()

    def test_example_without_tail_unchanged(self, g16_acetaldehyde, temp_workdir):
        """acetaldehyde.com has nothing after the geometry."""
        QRCGenerator(str(g16_acetaldehyde), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        content = (temp_workdir / 'acetaldehyde_QRC.com').read_text()
        assert content.splitlines()[3] == '# opt freq M062X/6-31G*'
        assert content.endswith('\n\n') and not content.endswith('\n\n\n')


class TestORCAInput:
    """All '!' lines and %blocks of the ORCA input are carried over."""

    ECHO = [
        '                                       INPUT FILE\n',
        '=' * 80 + '\n',
        'NAME = ts.inp\n',
        '|  1> ! wB97X-D3 def2-SVP OptTS Freq PAL8\n',
        '|  2> ! CPCM(water) TightSCF   # solvent\n',
        '|  3> %maxcore 3000\n',
        '|  4> %pal\n',
        '|  5>   nprocs 8\n',
        '|  6> end\n',
        '|  7> %geom\n',
        '|  8>   Constraints\n',
        '|  9>     {B 0 1 C}\n',
        '| 10>   end\n',
        '| 11> end\n',
        '| 12> *xyz 0 1\n',
        '| 13>  H 0.0 0.0 0.0\n',
        '| 14>  H 0.0 0.0 0.7\n',
        '| 15> *\n',
        '| 16> %cpcm smd true end\n',
        '| 17> $new_job\n',
        '| 18> ! SP\n',
        '| 19>                          ****END OF INPUT****\n',
    ]

    def test_read_orca_input(self):
        keywords, blocks = read_orca_input(self.ECHO)
        assert keywords == 'wB97X-D3 def2-SVP OptTS Freq PAL8 CPCM(water) TightSCF'
        assert blocks == ['%geom', '  Constraints', '    {B 0 1 C}', '  end', 'end', '%cpcm smd true end']

    def test_no_echo(self):
        assert read_orca_input(['nothing here\n']) == (None, [])

    def test_blocks_written_and_pal_stripped(self, orca_claisen_ts, temp_workdir):
        text = orca_claisen_ts.read_text().replace(
            '|  1> ! wB97X-D3 def2-SVP OptTS Freq\n|  2> \n',
            '|  1> ! wB97X-D3 def2-SVP OptTS Freq PAL8\n|  2> ! CPCM(water)\n|  3> %cpcm smd true end\n', 1)
        local = temp_workdir / 'multi.out'
        local.write_text(text)
        QRCGenerator(str(local), 0.3, 2, '4GB', None, False, 'QRC', None, None)
        lines = (temp_workdir / 'multi_QRC.inp').read_text().splitlines()
        assert lines[0] == '! wB97X-D3 def2-SVP Opt Freq CPCM(water)'
        assert lines[1:4] == [' %pal nprocs 2 end', ' %maxcore 2048', '%cpcm smd true end']


@QCHEM_DEV_CCLIB_SKIP
class TestQChemInput:
    """Q-Chem $rem settings and extra sections are carried over."""

    def test_read_qchem_input(self, qchem_claisen_ts):
        rem, extra = read_qchem_input(str(qchem_claisen_ts))
        assert not any('JOBTYPE' in line.upper() for line in rem)
        assert any('SCF_CONVERGENCE' in line for line in rem)
        assert extra == []

    def test_rem_and_sections_written_to_both_jobs(self, qchem_claisen_ts, temp_workdir):
        text = qchem_claisen_ts.read_text().replace(
            'SCF_CONVERGENCE      8\n$end\n',
            'SCF_CONVERGENCE      8\nSOLVENT_METHOD SMD\n$end\n\n$smx\nsolvent water\n$end\n')
        local = temp_workdir / 'q.out'
        local.write_text(text)
        QRCGenerator(str(local), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        opt_job, freq_job = (temp_workdir / 'q_QRC.inp').read_text().split('@@@')
        for job, jobtype in ((opt_job, 'opt'), (freq_job, 'freq')):
            assert f'JOBTYPE {jobtype}' in job
            assert 'SOLVENT_METHOD SMD' in job and 'SCF_CONVERGENCE' in job
            assert '$smx\nsolvent water\n$end' in job
            assert 'JOBTYPE' in job and job.upper().count('JOBTYPE') == 1


class TestXYZOutput:
    """--xyz writes the displaced geometry only."""

    def test_xyz_flag(self, g16_claisen_ts, temp_workdir):
        qrc = QRCGenerator(str(g16_claisen_ts), 0.3, 1, '4GB', None, False, 'QRC', None, None, xyz=True)
        lines = (temp_workdir / 'claisen_ts_QRC.xyz').read_text().splitlines()
        assert not (temp_workdir / 'claisen_ts_QRC.com').exists()
        assert int(lines[0]) == qrc.NATOMS and len(lines) == qrc.NATOMS + 2
        assert 'mode(s) 1' in lines[1]
        np.testing.assert_allclose([float(x) for x in lines[2].split()[1:]], qrc.NEW_CARTESIAN[0], atol=1e-7)


class TestCLIConveniences:
    """--both, extension case, default amplitude, unexpected errors."""

    def test_default_amplitude(self):
        assert DEFAULT_AMPLITUDE == 0.3

    def test_both_directions(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_claisen_ts), '--both', '-q'])
        assert main() == 0
        coords = {}
        for suffix in ('QRCF', 'QRCR'):
            lines = (temp_workdir / f'claisen_ts_{suffix}.com').read_text().splitlines()
            coords[suffix] = np.array([[float(x) for x in line.split()[1:]] for line in lines[8:22]])
        original = QRCGenerator(str(g16_claisen_ts), 0.0, 1, '4GB', None, False, 'X', None, None,
                                write=False).CARTESIAN
        np.testing.assert_allclose(coords['QRCF'] - original, original - coords['QRCR'], atol=2e-8)
        assert np.abs(coords['QRCF'] - original).max() > 0.01
        # The route note is printed once, not once per direction
        assert capsys.readouterr().out.count('route changed') == 1

    def test_uppercase_extension(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        shutil.copy(g16_acetaldehyde, temp_workdir / 'ACETALDEHYDE.LOG')
        monkeypatch.setattr('sys.argv', ['pyqrc', 'ACETALDEHYDE.LOG', '-q'])
        assert main() == 0
        assert (temp_workdir / 'ACETALDEHYDE_QRC.com').exists()

    def test_non_utf8_output(self, g16_claisen_ts, temp_workdir, monkeypatch):
        data = g16_claisen_ts.read_bytes().replace(b' Charge =', b' caf\xe9 Charge =', 1)
        (temp_workdir / 'enc.log').write_bytes(data)
        monkeypatch.setattr('sys.argv', ['pyqrc', 'enc.log', '-q'])
        assert main() == 0
        assert (temp_workdir / 'enc_QRC.com').exists()

    def test_unexpected_error_does_not_stop_batch(self, g16_acetaldehyde, g16_claisen_ts,
                                                  temp_workdir, monkeypatch, capsys):
        real_init = QRCGenerator.__init__

        def flaky_init(self, file, *args, **kwargs):
            if 'acetaldehyde' in str(file):
                raise RuntimeError('boom')
            real_init(self, file, *args, **kwargs)

        monkeypatch.setattr(QRCGenerator, '__init__', flaky_init)
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_acetaldehyde), str(g16_claisen_ts), '-q'])
        assert main() == 1
        assert 'unexpected error' in capsys.readouterr().out
        assert (temp_workdir / 'claisen_ts_QRC.com').exists()

    def test_overlap_warning(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_claisen_ts), '--amp', '5', '-q'])
        main()
        assert 'atoms are very close' in capsys.readouterr().out


# --- ORCA reader (fallback for ORCA outputs cclib cannot read, e.g. ORCA 6) ---

import logging  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order
import cclib  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order

from pyqrc.orca_reader import (  # noqa: E402  pylint: disable=wrong-import-position
    ORCAReadError,
    is_orca_output,
    read_orca_frequencies,
)
from pyqrc.pyQRC import parse_output  # noqa: E402  pylint: disable=wrong-import-position

# Reference values from cclib master, which can read ORCA 6
ORCA6_REFERENCE = {
    'acetaldehyde': dict(natom=7, nvib=15, first=-190.74, last=3174.51,
                         disp=[-3.5e-05, 0.000184, -0.093569], coord=[0.236194, 0.411275, 0.001351]),
    'claisen_ts': dict(natom=14, nvib=36, first=-636.62, last=3280.48,
                       disp=[0.368031, 0.185735, 0.076536], coord=[-1.409443, 0.796601, -0.282714]),
    'full_alkyne_TS': dict(natom=130, nvib=384, first=-105.42, last=3258.79,
                           disp=[0.005891, -0.008512, -0.000255], coord=[0.515094, -0.787338, 14.336678]),
}


def _truncate(src, dest, marker):
    """Copy an output, cutting it at the first line containing marker."""
    lines = Path(src).read_text().splitlines(keepends=True)
    cut = next(i for i, line in enumerate(lines) if marker in line)
    Path(dest).write_text(''.join(lines[:cut]))
    return dest


class TestORCAReader:
    """pyQRC's ORCA reader agrees with cclib and reads ORCA 6."""

    @pytest.mark.parametrize('name', ['acetaldehyde', 'claisen_ts'])
    def test_matches_cclib_on_orca5(self, name):
        """On ORCA 5 outputs, which cclib releases read, both parsers agree."""
        path = str(datapath(f'orca5/{name}.out'))
        ours = read_orca_frequencies(path)
        ref = cclib.io.ccopen(path, loglevel=logging.CRITICAL).parse()
        assert ours.natom == ref.natom
        assert list(ours.atomnos) == list(ref.atomnos)
        assert (ours.charge, ours.mult) == (ref.charge, ref.mult)
        np.testing.assert_allclose(ours.atomcoords[-1], ref.atomcoords[-1])
        np.testing.assert_allclose(ours.vibfreqs, ref.vibfreqs, atol=0.01)
        np.testing.assert_allclose(ours.vibdisps, ref.vibdisps)

    @pytest.mark.parametrize('name', sorted(ORCA6_REFERENCE))
    def test_reads_orca6(self, name):
        ref = ORCA6_REFERENCE[name]
        data = read_orca_frequencies(str(datapath(f'orca6/{name}.out')))
        assert data.natom == ref['natom'] and (data.charge, data.mult) == (0, 1)
        assert len(data.vibfreqs) == ref['nvib'] == 3 * ref['natom'] - 6
        assert data.vibfreqs[0] == pytest.approx(ref['first'])
        assert data.vibfreqs[-1] == pytest.approx(ref['last'])
        assert data.vibdisps.shape == (ref['nvib'], ref['natom'], 3)
        np.testing.assert_allclose(data.vibdisps[0][0], ref['disp'])
        np.testing.assert_allclose(data.atomcoords[-1][0], ref['coord'])
        assert data.metadata['package'] == 'ORCA'
        assert data.metadata['package_version'].startswith('6.')

    @pytest.mark.parametrize('name', sorted(ORCA6_REFERENCE))
    def test_parse_output_reads_orca6(self, name, capfd):
        """parse_output gives the same data whichever cclib is installed, quietly."""
        data = parse_output(str(datapath(f'orca6/{name}.out')))
        assert len(data.vibfreqs) == ORCA6_REFERENCE[name]['nvib']
        assert data.vibfreqs[0] == pytest.approx(ORCA6_REFERENCE[name]['first'], abs=0.01)
        # cclib's own error log is silenced when the fallback takes over
        assert 'ERROR' not in capfd.readouterr().err

    def test_fallback_when_cclib_raises(self, orca_claisen_ts):
        class FailingParser:
            def parse(self):
                raise IndexError('list index out of range')

        with patch('pyqrc.pyQRC.cclib.io.ccopen', return_value=FailingParser()):
            data = parse_output(str(orca_claisen_ts))
        assert isinstance(data, SimpleNamespace)
        assert data.vibfreqs[0] == pytest.approx(-633.34, abs=0.01)

    def test_fallback_when_cclib_finds_no_modes(self, orca_claisen_ts):
        class NoModesParser:
            def parse(self):
                return SimpleNamespace(natom=14)

        with patch('pyqrc.pyQRC.cclib.io.ccopen', return_value=NoModesParser()):
            data = parse_output(str(orca_claisen_ts))
        assert data.vibdisps.shape == (36, 14, 3)

    def test_cclib_preferred_when_it_works(self, orca_claisen_ts):
        assert not isinstance(parse_output(str(orca_claisen_ts)), SimpleNamespace)

    def test_non_orca_errors_not_hidden(self, g16_claisen_ts):
        class FailingParser:
            def parse(self):
                raise ValueError('bad gaussian')

        with patch('pyqrc.pyQRC.cclib.io.ccopen', return_value=FailingParser()):
            with pytest.raises(ValueError, match='bad gaussian'):
                parse_output(str(g16_claisen_ts))

    def test_truncated_orca_output_raises(self, orca6_claisen_ts, tmp_path):
        """An output that stops before the normal modes is a clear parse error."""
        path = _truncate(orca6_claisen_ts, tmp_path / 'cut.out', 'NORMAL MODES')
        with pytest.raises(ORCAReadError, match='no frequency calculation'):
            read_orca_frequencies(str(path))
        with patch('pyqrc.pyQRC.cclib.io.ccopen', side_effect=IndexError('boom')):
            with pytest.raises(QRCParseError, match='pyQRC ORCA reader'):
                parse_output(str(path))

    def test_no_frequency_job(self, orca6_claisen_ts, tmp_path):
        path = _truncate(orca6_claisen_ts, tmp_path / 'opt.out', 'VIBRATIONAL FREQUENCIES')
        with pytest.raises(ORCAReadError):
            read_orca_frequencies(str(path))

    def test_without_first_vibration_line(self, orca_acetaldehyde, tmp_path):
        """Falls back to dropping 6 translations/rotations for a non-linear molecule."""
        text = orca_acetaldehyde.read_text()
        path = tmp_path / 'nofirst.out'
        path.write_text(text.replace('The first frequency considered to be a vibration', 'removed'))
        np.testing.assert_allclose(read_orca_frequencies(str(path)).vibfreqs,
                                   read_orca_frequencies(str(orca_acetaldehyde)).vibfreqs)

    def test_is_orca_output(self, orca6_claisen_ts, g16_claisen_ts, qchem_claisen_ts):
        assert is_orca_output(str(orca6_claisen_ts))
        assert not is_orca_output(str(g16_claisen_ts))
        assert not is_orca_output(str(qchem_claisen_ts))


class TestORCA6EndToEnd:
    """ORCA 6 outputs work through the CLI with any cclib."""

    def test_cli_orca6_ts(self, orca6_claisen_ts, temp_workdir, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(orca6_claisen_ts), '--both', '--nproc', '4', '--mem', '8GB'])
        assert main() == 0
        assert '1 imaginary frequencies: processed' in capsys.readouterr().out
        for suffix in ('QRCF', 'QRCR'):
            lines = (temp_workdir / f'claisen_ts_{suffix}.inp').read_text().splitlines()
            assert lines[:3] == ['! wB97X-D3 def2-SVP Opt Freq', ' %pal nprocs 4 end', ' %maxcore 2048']
            assert '* xyz 0 1' in lines
        assert (temp_workdir / 'claisen_ts_QRCF.qrc').exists()

    def test_orca6_displacement_along_imaginary_mode(self, orca6_claisen_ts, temp_workdir):
        qrc = QRCGenerator(str(orca6_claisen_ts), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        displacement = qrc.NEW_CARTESIAN - qrc.CARTESIAN
        np.testing.assert_allclose(displacement, 0.3 * qrc.DISPS[0])
        assert qrc.MW_DISTANCE > 0

    def test_orca61_large_ts(self, orca6_alkyne_ts, temp_workdir):
        """ORCA 6.1, 130 atoms, lowercase optts keyword with RIJCOSX."""
        QRCGenerator(str(orca6_alkyne_ts), 0.3, 8, '32GB', None, False, 'QRC', None, None)
        lines = (temp_workdir / 'full_alkyne_TS_QRC.inp').read_text().splitlines()
        assert lines[0] == '! Opt wb97x-d3 def2-svp rijcosx freq'
        assert ' %maxcore 4096' in lines
        assert sum(1 for line in lines if len(line.split()) == 4 and line.split()[0].isalpha()) == 130


# --- Phase 2: minima skipped, --outdir/--overwrite, from_arrays/from_ase ---

from pyqrc.pyQRC import (  # noqa: E402  pylint: disable=wrong-import-position
    QRCFileExistsError,
    vibrations_from_ase,
)

MINIMUM_LOG = datapath('g16/planar_chex_mode1.log')  # no imaginary frequencies


class TestSkipMinima:
    """Files without imaginary frequencies are skipped unless a mode is requested."""

    def test_minimum_skipped_by_default(self, temp_workdir, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(MINIMUM_LOG)])
        assert main() == 0
        out = capsys.readouterr().out
        assert 'no imaginary frequencies: skipping' in out and '--freqnum' in out
        assert not list(temp_workdir.iterdir())

    def test_minimum_with_freqnum_processed(self, temp_workdir, monkeypatch):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(MINIMUM_LOG), '--freqnum', '1', '-q'])
        assert main() == 0
        assert (temp_workdir / 'planar_chex_mode1_QRC.com').exists()

    def test_auto_still_skips_with_freqnum(self, temp_workdir, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(MINIMUM_LOG), '--auto', '--freqnum', '1'])
        assert main() == 0
        assert 'skipping' in capsys.readouterr().out
        assert not list(temp_workdir.iterdir())

    def test_library_warns_when_not_displaced(self, temp_workdir, capsys):
        qrc = QRCGenerator(str(MINIMUM_LOG), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        np.testing.assert_allclose(qrc.NEW_CARTESIAN, qrc.CARTESIAN)
        assert 'geometry was not displaced' in capsys.readouterr().out


class TestOutdir:
    """--outdir writes all files into a (new) directory."""

    def test_cli_outdir(self, g16_claisen_ts, temp_workdir, monkeypatch):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_claisen_ts), '--outdir', 'a/b', '--both'])
        assert main() == 0
        written = sorted(p.name for p in (temp_workdir / 'a' / 'b').iterdir())
        assert written == ['claisen_ts_QRCF.com', 'claisen_ts_QRCF.qrc', 'claisen_ts_QRCR.com', 'claisen_ts_QRCR.qrc']
        assert sorted(p.name for p in temp_workdir.iterdir()) == ['a']
        # The checkpoint is named for the job, which runs wherever it is submitted
        assert (temp_workdir / 'a/b/claisen_ts_QRCF.com').read_text().startswith('%chk=claisen_ts_QRCF.chk\n')

    def test_output_paths_match_written_files(self, orca_claisen_ts, tmp_path):
        qrc = QRCGenerator(str(orca_claisen_ts), 0.3, 1, '4GB', None, True, 'X', None, None,
                           write=False, outdir=str(tmp_path / 'o'))
        paths = qrc.output_paths()
        assert [p.name for p in paths] == ['claisen_ts_X.inp', 'claisen_ts_X.qrc']
        qrc.write_files()
        assert all(p.exists() for p in paths)
        assert sorted(p.name for p in (tmp_path / 'o').iterdir()) == sorted(p.name for p in paths)

    def test_output_paths_xyz(self, g16_claisen_ts, tmp_path):
        qrc = QRCGenerator(str(g16_claisen_ts), 0.3, 1, '4GB', None, False, 'QRC', None, None,
                           write=False, xyz=True, outdir=str(tmp_path))
        assert [p.name for p in qrc.output_paths()] == ['claisen_ts_QRC.xyz']


class TestOverwrite:
    """The CLI never replaces existing files unless --overwrite is given."""

    def test_second_run_refused(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_claisen_ts), '-q'])
        assert main() == 0
        target = temp_workdir / 'claisen_ts_QRC.com'
        target.write_text('edited by hand')
        assert main() == 1
        assert 'already exists' in capsys.readouterr().out
        assert target.read_text() == 'edited by hand'

    def test_both_writes_nothing_if_one_exists(self, g16_claisen_ts, temp_workdir, monkeypatch):
        (temp_workdir / 'claisen_ts_QRCR.com').write_text('keep')
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_claisen_ts), '--both'])
        assert main() == 1
        assert sorted(p.name for p in temp_workdir.iterdir()) == ['claisen_ts_QRCR.com']

    def test_existing_qrc_summary_also_protected(self, g16_claisen_ts, temp_workdir, monkeypatch):
        (temp_workdir / 'claisen_ts_QRC.qrc').write_text('keep')
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_claisen_ts)])
        assert main() == 1
        assert not (temp_workdir / 'claisen_ts_QRC.com').exists()

    def test_overwrite_flag(self, g16_claisen_ts, temp_workdir, monkeypatch):
        target = temp_workdir / 'claisen_ts_QRC.com'
        target.write_text('old')
        monkeypatch.setattr('sys.argv', ['pyqrc', str(g16_claisen_ts), '--overwrite', '-q'])
        assert main() == 0
        assert target.read_text().startswith('%chk=')

    def test_library_default_overwrites(self, g16_claisen_ts, temp_workdir):
        (temp_workdir / 'claisen_ts_QRC.com').write_text('old')
        QRCGenerator(str(g16_claisen_ts), 0.3, 1, '4GB', None, False, 'QRC', None, None)
        assert (temp_workdir / 'claisen_ts_QRC.com').read_text().startswith('%chk=')

    def test_library_overwrite_false_raises_before_writing(self, g16_claisen_ts, temp_workdir):
        (temp_workdir / 'claisen_ts_QRC.com').write_text('old')
        with pytest.raises(QRCFileExistsError):
            QRCGenerator(str(g16_claisen_ts), 0.3, 1, '4GB', None, True, 'QRC', None, None,
                         overwrite=False)
        assert not (temp_workdir / 'claisen_ts_QRC.qrc').exists()
        assert (temp_workdir / 'claisen_ts_QRC.com').read_text() == 'old'


class TestFromArrays:
    """QRCGenerator.from_arrays gives the same result as reading an output file."""

    @pytest.fixture
    def orca_data(self, orca_claisen_ts):
        return parse_output(str(orca_claisen_ts))

    def test_matches_file_based_generator(self, orca_claisen_ts, orca_data, temp_workdir):
        from_file = QRCGenerator(str(orca_claisen_ts), 0.3, 1, '4GB', None, False, 'QRC', None, None,
                                 write=False)
        from_arrays = QRCGenerator.from_arrays(
            orca_data.atomnos, orca_data.atomcoords[-1], orca_data.vibfreqs, orca_data.vibdisps)
        # ORCA prints unit-norm modes to 6 decimals
        np.testing.assert_allclose(from_arrays.NEW_CARTESIAN, from_file.NEW_CARTESIAN, atol=1e-5)
        assert from_arrays._target_modes == from_file._target_modes == {0}
        assert from_arrays.MW_DISTANCE == pytest.approx(from_file.MW_DISTANCE, rel=1e-4)
        assert not list(temp_workdir.iterdir()), "from_arrays must not write by default"

    def test_modes_are_normalized(self, orca_data):
        base = QRCGenerator.from_arrays(orca_data.atomnos, orca_data.atomcoords[-1],
                                        orca_data.vibfreqs, orca_data.vibdisps)
        scaled = QRCGenerator.from_arrays(orca_data.atomnos, orca_data.atomcoords[-1],
                                          orca_data.vibfreqs, 25.0 * orca_data.vibdisps)
        np.testing.assert_allclose(scaled.NEW_CARTESIAN, base.NEW_CARTESIAN)

    def test_mode_selection_and_negative_amplitude(self, orca_data):
        args = (orca_data.atomnos, orca_data.atomcoords[-1], orca_data.vibfreqs, orca_data.vibdisps)
        fwd = QRCGenerator.from_arrays(*args, amplitude=0.3, num=5)
        rev = QRCGenerator.from_arrays(*args, amplitude=-0.3, val=orca_data.vibfreqs[4])
        assert fwd._target_modes == rev._target_modes == {4}
        np.testing.assert_allclose(fwd.NEW_CARTESIAN - fwd.CARTESIAN, rev.CARTESIAN - rev.NEW_CARTESIAN)

    def test_write_xyz(self, orca_data, tmp_path):
        qrc = QRCGenerator.from_arrays(orca_data.atomnos, orca_data.atomcoords[-1], orca_data.vibfreqs,
                                       orca_data.vibdisps, name='mlip_ts', outdir=str(tmp_path), write=True)
        lines = (tmp_path / 'mlip_ts_QRC.xyz').read_text().splitlines()
        assert int(lines[0]) == 14 and len(lines) == 16
        np.testing.assert_allclose([float(x) for x in lines[2].split()[1:]], qrc.NEW_CARTESIAN[0], atol=1e-7)

    def test_write_gaussian_with_route(self, orca_data, tmp_path):
        QRCGenerator.from_arrays(orca_data.atomnos, orca_data.atomcoords[-1], orca_data.vibfreqs,
                                 orca_data.vibdisps, name='ts', program='Gaussian', charge=-1, mult=2,
                                 route='opt=(ts,calcfc) b3lyp/6-31g(d)', outdir=str(tmp_path), write=True)
        lines = (tmp_path / 'ts_QRC.com').read_text().splitlines()
        assert lines[3] == '# opt=(ts,calcfc) b3lyp/6-31g(d)'  # used as given
        assert lines[7] == '-1 2'

    def test_write_orca_with_route(self, orca_data, tmp_path):
        QRCGenerator.from_arrays(orca_data.atomnos, orca_data.atomcoords[-1], orca_data.vibfreqs,
                                 orca_data.vibdisps, name='ts', program='ORCA', route='r2SCAN-3c Opt',
                                 nproc=4, mem='8GB', outdir=str(tmp_path), write=True)
        lines = (tmp_path / 'ts_QRC.inp').read_text().splitlines()
        assert lines[:3] == ['! r2SCAN-3c Opt', ' %pal nprocs 4 end', ' %maxcore 2048']

    @pytest.mark.parametrize('change,match', [
        (dict(coords=np.zeros((13, 3))), 'coords must have shape'),
        (dict(modes=np.ones((3, 14, 3))), 'modes must have shape'),
        (dict(atomnos=[0] * 14), 'atomic numbers'),
        (dict(program='QChem', route='x'), 'program must be'),
        (dict(program='Gaussian'), 'a route is needed'),
        (dict(modes='zero'), 'all-zero'),
    ])
    def test_validation(self, orca_data, change, match):
        kwargs = dict(atomnos=orca_data.atomnos, coords=orca_data.atomcoords[-1],
                      freqs=orca_data.vibfreqs, modes=orca_data.vibdisps)
        kwargs.update(change)
        if isinstance(kwargs['modes'], str):
            kwargs['modes'] = np.zeros_like(orca_data.vibdisps)
        with pytest.raises(ValueError, match=match):
            QRCGenerator.from_arrays(**kwargs)


class FakeAtoms:
    """Just enough of ase.Atoms for from_ase."""

    def get_atomic_numbers(self):
        return np.array([8, 1, 1])

    def get_positions(self):
        return np.array([[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]])

    def get_initial_charges(self):
        return np.array([-1.0, 0.0, 0.0])


class FakeVibrations:
    """Just enough of ase.vibrations.Vibrations for from_ase: 9 modes, 6 trans/rot."""

    atoms = FakeAtoms()

    def get_energies(self):
        # eV; 0.062j ~ 500i cm-1; tiny values are trans/rot noise
        return np.array([1e-5j, 2e-6j, 1e-6, 2e-6, 3e-6, 4e-6, 0.062j, 0.1984, 0.4587])

    def get_mode(self, i):
        mode = np.zeros((3, 3))
        mode[i % 3, i % 3] = 2.0 + i
        return mode


class TestFromAse:
    """from_ase works on an ASE Vibrations object (ASE is not a dependency)."""

    def test_vibrations_from_ase(self):
        freqs, modes = vibrations_from_ase(FakeVibrations())
        np.testing.assert_allclose(freqs, [-500.06, 1600.2, 3699.7], atol=0.1)
        assert modes.shape == (3, 3, 3)

    def test_from_ase(self):
        qrc = QRCGenerator.from_ase(FakeVibrations(), amplitude=0.3)
        assert qrc.CHARGE == -1 and qrc.MULT == 1
        assert qrc._target_modes == {0}
        expected = FakeAtoms().get_positions()
        expected[0, 0] += 0.3  # mode 6 is a unit x-displacement of atom 0
        np.testing.assert_allclose(qrc.NEW_CARTESIAN, expected)

    def test_from_ase_kwargs(self):
        qrc = QRCGenerator.from_ase(FakeVibrations(), charge=0, num=2, amplitude=-0.1)
        assert qrc.CHARGE == 0 and qrc._target_modes == {1}

    def test_matches_ase2gaussian_route(self, temp_workdir):
        """A real ASE Vibrations run gives the same geometry as the log-file bridge."""
        ase = pytest.importorskip('ase')
        from ase.calculators.emt import EMT  # pylint: disable=import-outside-toplevel
        from ase.vibrations import Vibrations  # pylint: disable=import-outside-toplevel
        import importlib.util  # pylint: disable=import-outside-toplevel
        from tests.conftest import EXAMPLES_PATH  # pylint: disable=import-outside-toplevel
        spec = importlib.util.spec_from_file_location('ase2gaussian', EXAMPLES_PATH / 'ase_mlip' / 'ase2gaussian.py')
        ase2gaussian = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ase2gaussian)

        atoms = ase.Atoms('NH3', positions=[[0, 0, 0.1], [1.0, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.87, 0]])
        atoms.calc = EMT()
        vib = Vibrations(atoms, name=str(temp_workdir / 'vib'))
        vib.run()
        freqs, modes = ase2gaussian.extract_vibrations(vib)
        ase2gaussian.write_gaussian_freq_log('bridge.log', atoms, freqs, modes)
        via_log = QRCGenerator('bridge.log', 0.3, 1, '4GB', None, False, 'QRC', None, 1, write=False)
        direct = QRCGenerator.from_ase(vib, amplitude=0.3, num=1)
        np.testing.assert_allclose(direct.FREQS, via_log.FREQS, atol=1e-3)
        # The log stores displacements to 2 decimals
        np.testing.assert_allclose(direct.NEW_CARTESIAN, via_log.NEW_CARTESIAN, atol=0.3 * 0.006)
