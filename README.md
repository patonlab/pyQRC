![pyQRC](pyQRC_banner.png)

[![DOI](https://zenodo.org/badge/138228684.svg)](https://zenodo.org/badge/latestdoi/138228684)
[![Build Status](https://travis-ci.org/bobbypaton/pyQRC.svg?branch=master)](https://travis-ci.org/bobbypaton/pyQRC)

QRC is an abbreviation of **Quick Reaction Coordinate**. This provides a quick alternative to IRC (intrisic reaction coordinate) calculations. This was first described by Silva and Goodman.<sup>1</sup> The [original code](http://www-jmg.ch.cam.ac.uk/software/QRC/) was developed in java for Jaguar output files. This python version works for Gaussian ouput files.

The program will read a Gaussian frequency calculation and will create a new input file which has been projcted from the final coordinates along the Hessian eigenvector with a negative force constant. The magnitude of displacement can be adjusted on the command line. By default the projection will be in a positive sense (in relation to the imaginary normal mode) and the level of theory in the new input file will match that of the frequency calculation. In addition to the new input file(s) a summary is output to a text file ending in '.qrc'

In addition to a pound-shop (dollar store) IRC calculation, a common application for pyQRC is in distorting ground state structures to remove annoying imaginary frequencies after reoptimization. This code has, in some form or other, been in use since around 2010.

#### Installation
1. Clone the repository https://github.com/bobbypaton/pyQRC.git and add to your PYTHONPATH variable
2. Run the script as a python module with your Gaussian output files (the program expects log or out extensions) and can accept wildcard arguments.

**Correct Usage**

```python
python -m pyqrc [--amp AMPLITUDE] [--nproc N] [--mem NGB] [--name APPEND] [--route 'B3LYP/6-31G*'] [-v] [--auto] [--freqnum INT] <gaussian_output_file(s)>
```

*	The `--amp` multiplies the imaginary normal mode vector by this amount. It defaults to 0.2. Increase for larger displacements, and change the sign for displacement in the reverse direction.
*	The `--nproc ` option selects the number of processors requested in the new input file. It defatuls to 1.
*	The `--mem` option specifies the memory requested in the new input file. It defatuls to 4GB. The correct format of input is XGB or X000MB where X can take any integer value.
*	The `--route` option specifies the route line for the new calculation to be performed.
*	The `--name` option is appended to the existing filename to create the new input file(s). This defaults to 'QRC'.
*	The `-v` option requests verbose output to be printed.
*	The `--auto` option will only process files with an imaginary frequency. Given any number of files it will ignore those that have no imaginary frequencies.
* The `-f` or `--freq` option allows you to request motion along a particular frequency (in cm-1).
* The `--freqnum` option allows you to request motion along a particular frequency (by number from the lowest).
 
## Example 1

```python
python -m pyqrc acetaldehyde.log --nproc 4 --mem 8GB
```

This initial optimization inadvertently produced a transition structure. The code displaces along the normal mode and creates a new input file. A subsequent optimization then fixes the problem since the imaginary frequency disappears. Note that by default this displacement occurs along all imaginary modes - if there is more than one imaginary frequency, and displacement is only desired along one of these (e.g. the lowest) then the use of `--freqnum 1` is necessary.


## Example 2

```python
python -m pyqrc claisen_ts.log --nproc 4 --mem 8GB --amp 0.3 --name QRCF
python -m pyqrc claisen_ts.log --nproc 4 --mem 8GB --amp -0.3 --name QRCR
```

The initial optimization located a transition structure. The quick reaction coordinate (QRC) is obtained from two optmizations, started from twp points displaced along the reaction coordinate in either direction.


## Example 3

```python
python -m pyqrc planar_chex.log --nproc 4 --freqnum 1 --name mode1
python -m pyqrc planar_chex.log --nproc 4 --freqnum 3 --name mode3
```

In this example, the initial optimization located a (3rd order) saddle point - planar cyclohexane - with three imaginary frequencies. Two new inputs are created by displacing along (i) only the first (i.e., lowest) normal mode and (ii) only the third normal mode. This contrasts from the `auto` function of pyQRC which displaces along all imaginary modes. Subsequent optimizations of these new inputs results in different minima, producing (i) chair-shaped cyclohexane and (ii) twist-boat cyclohexane. This example illustrates that displacement along particular normal modes could be used for e.g. conformational sampling.


#### References for the underlying theory
1. (a) Goodman, J. M.; Silva, M. A. *Tetrahedron Lett.* **2003**, *44*, 8233-8236 [**DOI:** 10.1016/j.tetlet.2003.09.074](http://dx.doi.org/10.1016/j.tetlet.2003.09.074); (b) Goodman, J. M.; Silva, M. A. *Tetrahedron Lett.* **2005**, *46*, 2067-2069 [**DOI:** 10.1016/j.tetlet.2005.01.142](http://dx.doi.org/10.1016/j.tetlet.2005.01.142)

---
License: [MIT](https://opensource.org/licenses/MIT)
