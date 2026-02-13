# Transition States

## Contents

This directory contains Gaussian input files (`.com`) and output log files (`.log`) for transition state optimizations and frequency calculations.

- `ts_NNN.com` - Gaussian input files
- `ts_NNN.log` - Gaussian output log files
- `ts.xyz` - Original Grambow transition state structures used as starting geometries
- `xyz_to_gaussian.py` - Script to convert XYZ structures to Gaussian input files
- `imag_freqs.txt` - Imaginary frequencies and original Grambow `rxn_id` for each transition state

## Origin

Transition structures were randomly sampled from the [Grambow dataset](https://www.nature.com/articles/s41597-020-0460-4). The original B97-D3/def2-mSVP transition states were reoptimized at the wB97XD/6-31G(d) level of theory with Gaussian 16 C.01.

## Timing

- Files: 588
- CPU time: 5 days 19 hours 44 minutes 6.5 seconds (optimization and frequency calculations)

All calculations were performed at the wB97XD/6-31G(d) level of theory with Gaussian 16 C.01.
