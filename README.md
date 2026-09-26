![pyQRC](https://raw.githubusercontent.com/patonlab/pyQRC/master/pyQRC_banner.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18308862.svg)](https://doi.org/10.5281/zenodo.18308862)
[![PyPI version](https://badge.fury.io/py/pyqrc.svg)](https://badge.fury.io/py/pyqrc)
[![Python versions](https://img.shields.io/pypi/pyversions/pyqrc)](https://pypi.org/project/pyqrc/)
[![Downloads](https://img.shields.io/pypi/dm/pyqrc)](https://pypi.org/project/pyqrc/)
[![License](https://img.shields.io/pypi/l/pyqrc)](https://opensource.org/licenses/MIT)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/patonlab/pyQRC/tree/master.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/patonlab/pyQRC/tree/master)
[![codecov](https://codecov.io/gh/patonlab/pyQRC/branch/master/graph/badge.svg)](https://codecov.io/gh/patonlab/pyQRC)

## Introduction

QRC is an abbreviation of **Quick Reaction Coordinate**. This provides a quick alternative to IRC (intrinsic reaction coordinate) calculations. This was first described by Silva and Goodman.<sup>1</sup> The [original code](http://www-jmg.ch.cam.ac.uk/software/QRC/) was developed in Java for Jaguar output files. This Python version uses [cclib](https://cclib.github.io/) to process a variety of computational chemistry outputs.

The program will read a Gaussian frequency calculation and will create a new input file which has been projected from the final coordinates along the Hessian eigenvector with a negative force constant. The magnitude of displacement can be adjusted on the command line. By default the projection will be in a positive sense (in relation to the imaginary normal mode) and the level of theory in the new input file will match that of the frequency calculation. The new input is set up to optimize to a minimum: saddle-point keywords from the original job (e.g. Gaussian `opt=(ts,noeigentest)`, ORCA `OptTS`) are replaced, and everything else in the original input — basis-set and ECP sections, solvation settings, constraints — is carried over (see [New input files](#new-input-files)).

In addition to a pound-shop (dollar store) IRC calculation, a common application for pyQRC is in distorting ground state structures to remove annoying imaginary frequencies after reoptimization. This code has, in some form or other, been in use since around 2010.

pyQRC reads frequency calculations from Gaussian, ORCA, and Q-Chem. It also integrates with [ASE](https://wiki.fysik.dtu.dk/ase/) and machine-learned interatomic potentials (MLIPs such as MACE, ANI, or AIMNet2): the bridge script in [examples/ase_mlip](examples/ase_mlip/) writes ASE frequency results in a format pyQRC reads, so Hessians from an MLIP work exactly like QM output files. Runnable Jupyter notebooks demonstrating each route — [Gaussian 16](examples/g16/generate_qrc_inputs.ipynb), [ORCA 5](examples/orca5/generate_qrc_inputs.ipynb) and [6](examples/orca6/generate_qrc_inputs.ipynb), [Q-Chem](examples/qchem/generate_qrc_inputs.ipynb), and [ASE/MLIP](examples/ase_mlip/generate_qrc_inputs.ipynb) — ship in the [examples](examples/) directory.

## Quick Start

```bash
# Install
pip install pyqrc

# Basic usage - displace along imaginary frequency
python -m pyqrc my_ts.log

# Specify processors and memory for the new input file
python -m pyqrc my_ts.log --nproc 4 --mem 8GB

# QRC from a transition state: inputs displaced in both directions
python -m pyqrc my_ts.log --both --nproc 4 --mem 8GB
```

## Installation

**Via PyPI (recommended):**
```bash
pip install pyqrc
```

**Via uv:**
```bash
uv pip install pyqrc
```

**Via pixi:**
```bash
pixi add --pypi pyqrc
```

**From source:**
Clone the repository https://github.com/patonlab/pyQRC.git and add to your PYTHONPATH variable.

### ORCA 6 compatibility

ORCA 6 outputs work with a plain `pip install pyqrc`. The current cclib release (1.8.1) cannot read ORCA 6 outputs, so pyQRC reads ORCA files that cclib fails on with its own ORCA reader, which extracts the final geometry, charge, multiplicity, frequencies and normal modes. It gives identical results to cclib on ORCA 5 outputs and to cclib's development version on ORCA 6 outputs, and cclib is still used whenever it can read the file.

Then run the script as a Python module with your computational chemistry output files (the program expects `.log` or `.out` extensions, in any case) and can accept wildcard arguments.

## Usage

```bash
python -m pyqrc [options] <output_file(s)>
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--amp AMPLITUDE` | Multiplier for the imaginary normal mode vector. Increase for larger displacements; use negative values for reverse direction. | `0.3` |
| `--nproc NPROC` | Number of processors requested in the new input file. | `1` |
| `--mem MEM` | Total memory requested in the new input file, e.g. `8GB` or `4000MB` (case-insensitive). For ORCA this is divided by `--nproc` to give `%maxcore`, which is per core. | `4GB` |
| `--route ROUTE` | Route line (Gaussian) or `!` keywords (ORCA) for the new calculation, used exactly as given, e.g. `'THEORY/BASIS opt'`. Ignored for Q-Chem. | Original, changed to a minimization |
| `-q, --quiet` | Suppress verbose output (skips the `.qrc` summary file). | Verbose by default |
| `--auto` | Skip files without imaginary frequencies even when `--freq`/`--freqnum` is given. Without those options, such files are always skipped. | Disabled |
| `--name SUFFIX` | String appended to the filename for new input file(s). | `QRC` |
| `-f, --freq FREQ` | Displace along the normal mode nearest this frequency (cm⁻¹); errors if no mode is within 1 cm⁻¹. | All imaginary |
| `--freqnum FREQNUM` | Displace along frequency number N (from lowest); errors if N exceeds the number of modes. | All imaginary |
| `--both` | Write two inputs, displaced by `+AMPLITUDE` and `-AMPLITUDE`, with `F` and `R` appended to the name (`<filename>_QRCF`, `<filename>_QRCR`). | Disabled |
| `--xyz` | Write the displaced geometry as an `.xyz` file instead of an input file. | Disabled |
| `--outdir DIR` | Directory for the new files, created if needed. | Current directory |
| `--overwrite` | Replace existing files. Without it, a file whose outputs (any of them, with `--both`) already exist is skipped with an error, and nothing is written for it. | Disabled |
| `--qcoord` | **Deprecated, removal in 3.0.** Runs Gaussian single points along normal modes directly on the local machine (requires `g16` on `PATH`). Generate inputs with the default mode and submit them through your scheduler instead. | Disabled |
| `--nummodes NUMMODES` | **Deprecated, removal in 3.0.** Number of modes for `--qcoord` calculations. | `all` |

## Output Files

Files without imaginary frequencies are skipped unless `--freq` or `--freqnum` asks for a particular mode, since there is nothing to displace along. For each file processed, pyQRC generates the following in the current directory (or `--outdir`), and never replaces existing files unless `--overwrite` is given:

- **`<filename>_QRC.com`** (Gaussian) or **`<filename>_QRC.inp`** (ORCA/Q-Chem): New input file with displaced geometry ready for optimization.
- **`<filename>_QRC.xyz`**: Displaced geometry, written instead of an input file with `--xyz`, or for outputs from other programs that cclib reads (e.g. Psi4, NWChem, Turbomole) since pyQRC cannot write their inputs.
- **`<filename>_QRC.qrc`**: Summary file containing:
  - Original geometry
  - Harmonic frequencies, reduced masses, and force constants
  - Normal mode displacement vectors
  - Mass-weighted Cartesian displacement magnitude

### New input files

Unless `--route` is given, the new input repeats the original calculation but is set up to optimize to a minimum, so that a QRC from a transition state relaxes to the reactant or product instead of searching for the TS again. A note is printed whenever the route is changed:

- **Gaussian:** `ts`, `saddle=N`, `qst2`/`qst3`, `noeigentest` and `readfc` are removed from the `opt` options, and `guess=read` is removed (the new job has no checkpoint to read). `opt` is added to a frequency-only route. Input that follows the geometry — `Gen`/`GenECP` basis sets and ECPs, ModRedundant lines, `SCRF=Read` input — is not in the Gaussian output, so it is copied from the original input file, `<filename>.com` or `<filename>.gjf` next to the output, when that file has the same route. With `--route`, these sections are copied only if the new route needs them (e.g. it uses `Gen`). If the route needs such input and no matching file is found, pyQRC prints a warning so it can be added by hand.
- **ORCA:** `OptTS` becomes `Opt` (and `Opt` is added if the job had no optimization). All `!` lines and `%` blocks (`%cpcm`, `%basis`, `%scf`, `%geom`, ...) are copied from the input that ORCA echoes at the top of the output; `%pal` and `%maxcore` come from `--nproc` and `--mem`. The `%` blocks are also copied when `--route` is given.
- **Q-Chem:** an optimization followed by a frequency job, both with every `$rem` setting and extra section (`$smx`, `$solvent`, `$basis`, ...) of the original frequency job, taken from the input Q-Chem echoes in its output.

## Dependencies

- [Python](https://www.python.org/) >= 3.9
- [cclib](https://cclib.github.io/) >= 1.8.1, < 2 (ORCA 6 outputs are read by pyQRC's own ORCA reader — see "ORCA 6 compatibility" above)
- [NumPy](https://numpy.org/) >= 1.22
- One of the following computational chemistry packages:
  - [Gaussian09](https://gaussian.com/glossary/g09/) / [Gaussian16](https://gaussian.com/gaussian16/)
  - [ORCA](https://sites.google.com/site/orcainputlibrary/home/) >= 4.0
  - [Q-Chem](https://www.q-chem.com/) >= 5.4

## Examples

The input and output files for the examples below ship in the [examples](examples/) directory. Each format subdirectory ([g16](examples/g16/), [orca5](examples/orca5/), [orca6](examples/orca6/), [qchem](examples/qchem/), [ase_mlip](examples/ase_mlip/)) also contains a runnable Jupyter notebook (`generate_qrc_inputs.ipynb`) that walks through generating QRC inputs from those files.

### Example 1: Remove an Unwanted Imaginary Frequency

```bash
python -m pyqrc acetaldehyde.log --nproc 4 --mem 8GB
```

This initial optimization inadvertently produced a transition structure. The code displaces along the normal mode and creates a new input file. A subsequent optimization then fixes the problem since the imaginary frequency disappears. Note that by default this displacement occurs along all imaginary modes - if there is more than one imaginary frequency, and displacement is only desired along one of these (e.g. the lowest) then the use of `--freqnum 1` is necessary.

### Example 2: Map a Reaction Coordinate (QRC)

```bash
python -m pyqrc claisen_ts.log --nproc 4 --mem 8GB --both
```

The initial optimization located a transition structure. The quick reaction coordinate (QRC) is obtained from two optimizations, started from two points displaced along the reaction coordinate in either direction: `--both` writes `claisen_ts_QRCF.com` (`--amp 0.3`) and `claisen_ts_QRCR.com` (`--amp -0.3`). The original job was a TS optimization, `opt(ts,calcfc,noeigentest)`, so the new inputs use `opt=calcfc` to relax to the minima on either side. The same two files can be written separately with `--amp 0.3 --name QRCF` and `--amp -0.3 --name QRCR`.

### Example 3: Conformational Sampling via Normal Mode Displacement

```bash
python -m pyqrc planar_chex.log --nproc 4 --freqnum 1 --name mode1
python -m pyqrc planar_chex.log --nproc 4 --freqnum 3 --name mode3
```

In this example, the initial optimization located a (3rd order) saddle point - planar cyclohexane - with three imaginary frequencies. Two new inputs are created by displacing along (i) only the first (i.e., lowest) normal mode and (ii) only the third normal mode. This contrasts from the `--auto` function of pyQRC which displaces along all imaginary modes. Subsequent optimizations of these new inputs results in different minima, producing (i) chair-shaped cyclohexane and (ii) twist-boat cyclohexane. This example illustrates that displacement along particular normal modes could be used for e.g. conformational sampling.

### Example 4: QRC from an ASE / MLIP Frequency Calculation

When the Hessian comes from a machine-learned interatomic potential driven through [ASE](https://wiki.fysik.dtu.dk/ase/) rather than a QM package, there is no output file for pyQRC to read. `QRCGenerator.from_ase` takes the ASE `Vibrations` object directly (see [Python API](#python-api)). To use the command line instead, the helper script [`examples/ase_mlip/ase2gaussian.py`](examples/ase_mlip/ase2gaussian.py) writes an ASE `Vibrations` result as a Gaussian-format log file that cclib parses, after which pyQRC works exactly as in the examples above. The accompanying [notebook](examples/ase_mlip/generate_qrc_inputs.ipynb) walks through the full loop with [MACE-OFF](https://github.com/ACEsuit/mace): locating the planar NH₃ inversion transition state and relaxing the QRC-displaced geometry to the pyramidal minimum, then mapping the Claisen reaction coordinate of Example 2 entirely on the MLIP — the forward and reverse QRC displacements from the DFT transition state relax to 4-pentenal and allyl vinyl ether without any further QM calculations.

## Python API

The command line wraps `QRCGenerator`, which can also be used directly:

```python
from pyqrc import QRCGenerator

# From an output file; write=False computes without writing files
qrc = QRCGenerator("claisen_ts.log", amplitude=0.3, nproc=4, mem="8GB", route=None,
                   verbose=False, suffix="QRC", val=None, num=None, write=False)
print(qrc.NEW_CARTESIAN)       # displaced geometry (Angstrom)
print(qrc.output_paths())      # files write_files() would write
qrc.write_files()

# From arrays, e.g. frequencies from another program
qrc = QRCGenerator.from_arrays(atomnos, coords, freqs, modes, amplitude=0.3)

# From an ASE Vibrations run (ASE is not a pyQRC dependency)
qrc = QRCGenerator.from_ase(vib, amplitude=0.3)
atoms.positions = qrc.NEW_CARTESIAN
```

`from_arrays` takes atomic numbers, coordinates (Å), frequencies (cm⁻¹, negative for imaginary modes, without translations and rotations) and the matching Cartesian displacement vectors, which are normalized as Gaussian and ORCA print them, so amplitudes mean the same as for output files. It accepts the same `amplitude`, `num` and `val` as the file-based generator. By default it only computes; with `write=True` it writes an `.xyz` file, or a Gaussian or ORCA input when `program` and `route` are given. `from_ase` passes its keyword arguments on to `from_arrays`.

## Comparison with IRC

To benchmark QRC against intrinsic reaction coordinate (IRC) calculations, 544 transition states from the [Grambow dataset](https://www.nature.com/articles/s41597-020-0460-4) were used. Transition states were reoptimized at the wB97XD/6-31G(d) level, and QRC calculations were performed with three displacement amplitudes (0.1, 0.3, and 0.5). The resulting reactant and product identities (canonical SMILES) and energies were compared against IRC results for the same structures.

| Amplitude | Barrier MAE (kcal/mol) | Rxn Energy MAE (kcal/mol) | Reactant Match | Product Match |
|-----------|------------------------|---------------------------|----------------|---------------|
| 0.1       | 4.55                   | 5.73                      | 97.8%          | 98.2%         |
| 0.3       | 0.65                   | 1.39                      | 98.9%          | 98.9%         |
| 0.5       | 0.58                   | 3.44                      | 98.7%          | 98.2%         |

An amplitude of **0.3** gives the best overall performance, with the highest SMILES match rates for both reactants and products (98.9%) and low MAE for barriers (0.65 kcal/mol) and reaction energies (1.39 kcal/mol). While an amplitude of 0.5 gives a marginally lower barrier MAE (0.58 kcal/mol), it produces larger errors in reaction energies and lower product match rates. An amplitude of 0.1 gives insufficient displacement, leading to higher energy errors and more mismatched products. Full details and data are available in the [irc_comparison](irc_comparison/analysis/) directory.

## Development

```bash
git clone https://github.com/patonlab/pyQRC.git
cd pyQRC
pip install -e ".[dev]"
pytest            # run the test suite
pylint pyqrc      # lint (CI requires a score >= 9.0)
```

Planned work is tracked in [ROADMAP.md](ROADMAP.md).

## Citation

If you use pyQRC in your research, please cite:

Paton, R. S.; Sowndarya S. V., S.; Landis, J.; Goodfellow, A. S. *pyQRC*. [**DOI:** 10.5281/zenodo.3365476](https://doi.org/10.5281/zenodo.3365476)

## References

1. (a) Goodman, J. M.; Silva, M. A. *Tetrahedron Lett.* **2003**, *44*, 8233-8236 [**DOI:** 10.1016/j.tetlet.2003.09.074](http://dx.doi.org/10.1016/j.tetlet.2003.09.074); (b) Goodman, J. M.; Silva, M. A. *Tetrahedron Lett.* **2005**, *46*, 2067-2069 [**DOI:** 10.1016/j.tetlet.2005.01.142](http://dx.doi.org/10.1016/j.tetlet.2005.01.142)

## Contributors

- Robert Paton ([@bobbypaton](https://github.com/bobbypaton))
- Guilian Luchini ([@luchini18](https://github.com/luchini18))
- Shree Sowndarya ([@shreesowndarya](https://github.com/shreesowndarya))
- Alister Goodfellow ([@aligfellow](https://github.com/aligfellow))

---
License: [MIT](https://opensource.org/licenses/MIT)
