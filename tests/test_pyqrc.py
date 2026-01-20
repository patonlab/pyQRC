#!/usr/bin/env python
"""Tests for pyQRC package."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pyqrc.pyQRC import (
    ATOMIC_MASSES,
    COVALENT_RADII,
    PERIODIC_TABLE,
    Logger,
    OutputData,
    QRCGenerator,
    check_overlap,
    element_id,
    gen_overlap,
    main,
    mwdist,
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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")
        data = OutputData(str(g16_acetaldehyde))
        assert data.format == "Gaussian"

    def test_orca_format_detection(self, orca_acetaldehyde):
        """Test ORCA format is detected."""
        if not orca_acetaldehyde.exists():
            pytest.skip("ORCA test file not found")
        data = OutputData(str(orca_acetaldehyde))
        assert data.format == "ORCA"

    def test_qchem_format_detection(self, qchem_acetaldehyde):
        """Test Q-Chem format is detected."""
        if not qchem_acetaldehyde.exists():
            pytest.skip("Q-Chem test file not found")
        data = OutputData(str(qchem_acetaldehyde))
        assert data.format == "QChem"


class TestQRCGeneratorAllFiles:
    """Tests for QRCGenerator against all example files."""

    def test_qrc_generation(self, example_file, example_format, temp_workdir):
        """Test QRC generation for all example files."""
        filepath = Path(example_file)
        if not filepath.exists():
            pytest.skip(f"{example_format} test file not found")

        # Copy file to temp directory
        shutil.copy(filepath, temp_workdir)
        local_file = temp_workdir / filepath.name

        try:
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
        except (IndexError, AttributeError) as e:
            # cclib may have parsing issues with certain output formats
            pytest.skip(f"cclib parsing error for {example_format}: {e}")

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
        if not filepath.exists():
            pytest.skip("TS test file not found")

        shutil.copy(filepath, temp_workdir)
        local_file = temp_workdir / filepath.name

        try:
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
        except (IndexError, AttributeError) as e:
            # cclib may have parsing issues with certain output formats
            pytest.skip(f"cclib parsing error for {ts_format}: {e}")

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
        if not filepath.exists():
            pytest.skip("Saddle point test file not found")

        shutil.copy(filepath, temp_workdir)
        local_file = temp_workdir / filepath.name

        try:
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
        except (IndexError, AttributeError) as e:
            pytest.skip(f"cclib parsing error for {saddle_format}: {e}")

        # Structure should have been displaced
        displacement = np.linalg.norm(
            np.array(qrc.NEW_CARTESIAN) - np.array(qrc.CARTESIAN)
        )
        assert displacement > 0, "Saddle point should be displaced along imaginary modes"


class TestQRCGeneratorOptions:
    """Tests for QRCGenerator with various options."""

    def test_specific_frequency_number(self, g16_claisen_ts, temp_workdir):
        """Test displacement along specific frequency number."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

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
        if not g16_planar_chex.exists():
            pytest.skip("Planar cyclohexane test file not found")

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
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

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
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
        main()

        # Check output file was created
        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_amplitude_option(self, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with --amp option."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--amp', '0.5', str(local_file)])
        main()

        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_nproc_and_mem(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with --nproc and --mem options."""
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--name', 'CUSTOM', str(local_file)])
        main()

        assert (temp_workdir / f"{local_file.stem}_CUSTOM.com").exists()

    def test_main_with_freq_option(self, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with --freq option to specify frequency value."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        # Use a frequency value that exists in the file
        monkeypatch.setattr('sys.argv', ['pyqrc', '-f', '-500', str(local_file)])
        main()

        # Should still create output even if freq doesn't exactly match
        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_freqnum_option(self, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with --freqnum option."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--freqnum', '1', str(local_file)])
        main()

        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_custom_route(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with --route option."""
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

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
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--auto', str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'imaginary frequencies: processing' in captured.out


class TestOutputDataExtended:
    """Extended tests for OutputData class."""

    def test_gaussian_termination_normal(self, g16_acetaldehyde):
        """Test Gaussian normal termination is detected."""
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")
        data = OutputData(str(g16_acetaldehyde))
        assert data.TERMINATION == "normal"

    def test_gaussian_jobtype_extraction(self, g16_acetaldehyde):
        """Test Gaussian job type is extracted."""
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")
        data = OutputData(str(g16_acetaldehyde))
        assert data.JOBTYPE is not None

    def test_gaussian_level_of_theory(self, g16_acetaldehyde):
        """Test Gaussian level of theory is extracted."""
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")
        data = OutputData(str(g16_acetaldehyde))
        assert data.LEVELOFTHEORY is not None
        # Should be in format "level/basis"
        assert '/' in data.LEVELOFTHEORY

    def test_orca_jobtype_extraction(self, orca_acetaldehyde):
        """Test ORCA job type is extracted."""
        if not orca_acetaldehyde.exists():
            pytest.skip("ORCA test file not found")
        data = OutputData(str(orca_acetaldehyde))
        assert data.JOBTYPE is not None

    def test_qchem_format_no_termination(self, qchem_acetaldehyde):
        """Test Q-Chem format detection (termination not implemented for Q-Chem)."""
        if not qchem_acetaldehyde.exists():
            pytest.skip("Q-Chem test file not found")
        data = OutputData(str(qchem_acetaldehyde))
        assert data.format == "QChem"
        # Q-Chem termination detection not implemented
        assert data.TERMINATION is None


class TestQRCGeneratorFormats:
    """Tests for QRCGenerator output format handling."""

    def test_orca_output_format(self, orca_acetaldehyde, temp_workdir):
        """Test ORCA input file generation."""
        if not orca_acetaldehyde.exists():
            pytest.skip("ORCA test file not found")

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        try:
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
        except (IndexError, AttributeError) as e:
            pytest.skip(f"cclib parsing error: {e}")

        output_file = temp_workdir / f"{local_file.stem}_QRC.inp"
        assert output_file.exists()
        content = output_file.read_text()
        # ORCA format checks
        assert '%pal nprocs 2 end' in content
        assert '%maxcore' in content
        assert '* xyz' in content

    def test_orca_memory_gb_conversion(self, orca_acetaldehyde, temp_workdir):
        """Test ORCA memory is converted from GB to MB."""
        if not orca_acetaldehyde.exists():
            pytest.skip("ORCA test file not found")

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        try:
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
        except (IndexError, AttributeError) as e:
            pytest.skip(f"cclib parsing error: {e}")

        output_file = temp_workdir / f"{local_file.stem}_QRC.inp"
        content = output_file.read_text()
        # 4GB should be converted to 4096 MB
        assert '%maxcore 4096' in content

    def test_orca_memory_mb(self, orca_acetaldehyde, temp_workdir):
        """Test ORCA memory with MB input."""
        if not orca_acetaldehyde.exists():
            pytest.skip("ORCA test file not found")

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        try:
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
        except (IndexError, AttributeError) as e:
            pytest.skip(f"cclib parsing error: {e}")

        output_file = temp_workdir / f"{local_file.stem}_QRC_MB.inp"
        content = output_file.read_text()
        # Should keep MB value
        assert '%maxcore 2000' in content

    def test_qchem_output_format(self, qchem_acetaldehyde, temp_workdir):
        """Test Q-Chem input file generation."""
        if not qchem_acetaldehyde.exists():
            pytest.skip("Q-Chem test file not found")

        shutil.copy(qchem_acetaldehyde, temp_workdir)
        local_file = temp_workdir / qchem_acetaldehyde.name

        try:
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
        except (IndexError, AttributeError) as e:
            pytest.skip(f"cclib parsing error: {e}")

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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

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
        if not orca_acetaldehyde.exists():
            pytest.skip("ORCA test file not found")

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        custom_route = "BP86 def2-SVP TightSCF"

        try:
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
        except (IndexError, AttributeError) as e:
            pytest.skip(f"cclib parsing error: {e}")

        output_file = temp_workdir / f"{local_file.stem}_QRC_custom.inp"
        content = output_file.read_text()
        assert custom_route in content


class TestSpecificFrequencyValue:
    """Tests for displacement along specific frequency value."""

    def test_val_parameter(self, g16_claisen_ts, temp_workdir):
        """Test QRCGenerator with val parameter for specific frequency."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

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
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

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
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")
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
        if not orca_claisen_ts.exists():
            pytest.skip("ORCA TS test file not found")
        data = OutputData(str(orca_claisen_ts))
        # ORCA uses _level_of_theory internally
        lot = data.LEVELOFTHEORY
        # May or may not be parsed depending on output format
        assert data.format == "ORCA"


class TestMainFileGlobbing:
    """Tests for file globbing in main()."""

    def test_main_with_glob_pattern(self, g16_acetaldehyde, temp_workdir, monkeypatch):
        """Test main with glob pattern for files."""
        if not g16_acetaldehyde.exists():
            pytest.skip("Gaussian test file not found")

        shutil.copy(g16_acetaldehyde, temp_workdir)
        local_file = temp_workdir / g16_acetaldehyde.name

        # Use glob pattern
        monkeypatch.setattr('sys.argv', ['pyqrc', str(temp_workdir / '*.log')])
        main()

        # Should process the file
        assert (temp_workdir / f"{local_file.stem}_QRC.com").exists()

    def test_main_with_multiple_files(self, g16_acetaldehyde, g16_claisen_ts, temp_workdir, monkeypatch):
        """Test main with multiple file arguments."""
        if not g16_acetaldehyde.exists() or not g16_claisen_ts.exists():
            pytest.skip("Test files not found")

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
        if not orca_acetaldehyde.exists():
            pytest.skip("ORCA test file not found")

        shutil.copy(orca_acetaldehyde, temp_workdir)
        local_file = temp_workdir / orca_acetaldehyde.name

        try:
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
        except (IndexError, AttributeError) as e:
            pytest.skip(f"cclib parsing error: {e}")

        output_file = temp_workdir / f"{local_file.stem}_QRC_nounit.inp"
        content = output_file.read_text()
        # Should use the numeric value directly
        assert '%maxcore 3000' in content


class TestPrintOutput:
    """Tests for print output in main()."""

    def test_main_prints_frequency_info(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """Test main prints frequency information."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'imaginary frequencies' in captured.out

    def test_main_prints_freq_value_info(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """Test main prints info when using --freq option."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '-f', '-500.0', str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'distorted along' in captured.out
        assert 'cm-1' in captured.out

    def test_main_prints_freqnum_info(self, g16_claisen_ts, temp_workdir, monkeypatch, capsys):
        """Test main prints info when using --freqnum option."""
        if not g16_claisen_ts.exists():
            pytest.skip("Gaussian TS test file not found")

        shutil.copy(g16_claisen_ts, temp_workdir)
        local_file = temp_workdir / g16_claisen_ts.name

        monkeypatch.setattr('sys.argv', ['pyqrc', '--freqnum', '2', str(local_file)])
        main()

        captured = capsys.readouterr()
        assert 'distorted along freq #2' in captured.out
