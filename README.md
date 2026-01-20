![pyQRC](pyQRC_banner.png)

[![DOI](https://zenodo.org/badge/138228684.svg)](https://zenodo.org/badge/latestdoi/138228684)
[![PyPI version](https://badge.fury.io/py/pyqrc.svg)](https://badge.fury.io/py/pyqrc)
[![Python versions](https://img.shields.io/pypi/pyversions/pyqrc)](https://pypi.org/project/pyqrc/)
[![Downloads](https://img.shields.io/pypi/dm/pyqrc)](https://pypi.org/project/pyqrc/)
[![License](https://img.shields.io/pypi/l/pyqrc)](https://opensource.org/licenses/MIT)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/patonlab/pyQRC/tree/master.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/patonlab/pyQRC/tree/master)
[![codecov](https://codecov.io/gh/patonlab/pyQRC/branch/master/graph/badge.svg)](https://codecov.io/gh/patonlab/pyQRC)

## Introduction

QRC is an abbreviation of **Quick Reaction Coordinate**. This provides a quick alternative to IRC (intrinsic reaction coordinate) calculations. This was first described by Silva and Goodman.<sup>1</sup> The [original code](http://www-jmg.ch.cam.ac.uk/software/QRC/) was developed in Java for Jaguar output files. This Python version uses [cclib](https://cclib.github.io/) to process a variety of computational chemistry outputs.

The program will read a Gaussian frequency calculation and will create a new input file which has been projected from the final coordinates along the Hessian eigenvector with a negative force constant. The magnitude of displacement can be adjusted on the command line. By default the projection will be in a positive sense (in relation to the imaginary normal mode) and the level of theory in the new input file will match that of the frequency calculation.

In addition to a pound-shop (dollar store) IRC calculation, a common application for pyQRC is in distorting ground state structures to remove annoying imaginary frequencies after reoptimization. This code has, in some form or other, been in use since around 2010.

## Quick Start

```bash
# Install
pip install pyqrc

# Basic usage - displace along imaginary frequency
python -m pyqrc my_ts.log

# Specify processors and memory for the new input file
python -m pyqrc my_ts.log --nproc 4 --mem 8GB
```

## Installation

**Via PyPI (recommended):**
```bash
pip install pyqrc
```

**From source:**
Clone the repository https://github.com/patonlab/pyQRC.git and add to your PYTHONPATH variable.

Then run the script as a Python module with your computational chemistry output files (the program expects `.log` or `.out` extensions) and can accept wildcard arguments.

## Usage

```bash
python -m pyqrc [options] <output_file(s)>
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--amp AMPLITUDE` | Multiplier for the imaginary normal mode vector. Increase for larger displacements; use negative values for reverse direction. | `0.2` |
| `--nproc N` | Number of processors requested in the new input file. | `1` |
| `--mem NGB` | Memory requested in the new input file. Format: `XGB` or `X000MB`. | `4GB` |
| `--route 'THEORY/BASIS'` | Route line for the new calculation. | Same as original |
| `--name SUFFIX` | String appended to the filename for new input file(s). | `QRC` |
| `-v` | Verbose output. | Enabled |
| `--auto` | Only process files with imaginary frequencies, skip others. | Disabled |
| `-f, --freq VALUE` | Displace along a specific frequency (in cm⁻¹). | All imaginary |
| `--freqnum N` | Displace along frequency number N (from lowest). | All imaginary |
| `--qcoord` | Automatic single point calculations along normal modes. | Disabled |
| `--nummodes N` | Number of modes for `--qcoord` calculations. | `all` |

## Output Files

pyQRC generates the following files:

- **`<filename>_QRC.com`** (Gaussian) or **`<filename>_QRC.inp`** (ORCA/Q-Chem): New input file with displaced geometry ready for optimization.
- **`<filename>_QRC.qrc`**: Summary file containing:
  - Original geometry
  - Harmonic frequencies, reduced masses, and force constants
  - Normal mode displacement vectors
  - Mass-weighted Cartesian displacement magnitude

## Dependencies

- [Python](https://www.python.org/) >= 3.9
- [cclib](https://cclib.github.io/)
- [NumPy](https://numpy.org/)
- One of the following computational chemistry packages:
  - [Gaussian09](https://gaussian.com/glossary/g09/) / [Gaussian16](https://gaussian.com/gaussian16/)
  - [ORCA](https://sites.google.com/site/orcainputlibrary/home/) >= 4.0
  - [Q-Chem](https://www.q-chem.com/) >= 5.4

## Examples

### Example 1: Remove an Unwanted Imaginary Frequency

```bash
python -m pyqrc acetaldehyde.log --nproc 4 --mem 8GB
```

This initial optimization inadvertently produced a transition structure. The code displaces along the normal mode and creates a new input file. A subsequent optimization then fixes the problem since the imaginary frequency disappears. Note that by default this displacement occurs along all imaginary modes - if there is more than one imaginary frequency, and displacement is only desired along one of these (e.g. the lowest) then the use of `--freqnum 1` is necessary.

### Example 2: Map a Reaction Coordinate (QRC)

```bash
python -m pyqrc claisen_ts.log --nproc 4 --mem 8GB --amp 0.3 --name QRCF
python -m pyqrc claisen_ts.log --nproc 4 --mem 8GB --amp -0.3 --name QRCR
```

The initial optimization located a transition structure. The quick reaction coordinate (QRC) is obtained from two optimizations, started from two points displaced along the reaction coordinate in either direction.

### Example 3: Conformational Sampling via Normal Mode Displacement

```bash
python -m pyqrc planar_chex.log --nproc 4 --freqnum 1 --name mode1
python -m pyqrc planar_chex.log --nproc 4 --freqnum 3 --name mode3
```

In this example, the initial optimization located a (3rd order) saddle point - planar cyclohexane - with three imaginary frequencies. Two new inputs are created by displacing along (i) only the first (i.e., lowest) normal mode and (ii) only the third normal mode. This contrasts from the `--auto` function of pyQRC which displaces along all imaginary modes. Subsequent optimizations of these new inputs results in different minima, producing (i) chair-shaped cyclohexane and (ii) twist-boat cyclohexane. This example illustrates that displacement along particular normal modes could be used for e.g. conformational sampling.

## Citation

If you use pyQRC in your research, please cite:

Paton, R. S. *pyQRC*. **2018**, https://doi.org/10.5281/zenodo.1407814

## References

1. (a) Goodman, J. M.; Silva, M. A. *Tetrahedron Lett.* **2003**, *44*, 8233-8236 [**DOI:** 10.1016/j.tetlet.2003.09.074](http://dx.doi.org/10.1016/j.tetlet.2003.09.074); (b) Goodman, J. M.; Silva, M. A. *Tetrahedron Lett.* **2005**, *46*, 2067-2069 [**DOI:** 10.1016/j.tetlet.2005.01.142](http://dx.doi.org/10.1016/j.tetlet.2005.01.142)

## Contributors

- Robert Paton ([@bobbypaton](https://github.com/bobbypaton))
- Guilian Luchini ([@luchini18](https://github.com/luchini18))
- Shree Sowndarya ([@shreesowndarya](https://github.com/shreesowndarya))
- Alister Goodfellow ([@aligfellow](https://github.com/aligfellow))

---
License: [MIT](https://opensource.org/licenses/MIT)
