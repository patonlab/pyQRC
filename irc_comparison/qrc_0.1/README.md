# QRC Displacement 0.1

## Contents

This directory contains Gaussian input files (`.com`), output log files (`.log`), and pyQRC displacement files (`.qrc`) for quick reaction coordinate (pyQRC) calculations using a displacement amplitude of 0.1.

- `ts_NNN_QRCF01.*` - Forward QRC-displaced structures
- `ts_NNN_QRCR01.*` - Reverse QRC-displaced structures
- `*_ii.*` - Re-submitted calculations where the initial optimization did not converge
- `QRC01_data.txt` - Summary of optimized energies and SMILES for each reaction
- `parse_qrc.py` - Script used to parse QRC output data

## How Inputs Were Generated

Starting from transition state geometries in `../ts/`, QRC-displaced input files were generated using pyQRC with a displacement amplitude of 0.1:

```
python -m pyqrc ts.log --amp 0.1 --append QRCF01
```

All subsequent geometry optimizations were performed at the wB97XD/6-31G(d) level of theory with Gaussian 16 C.01.

## Timing

- Files: 1261 (1122 opt+freq, 139 opt only)
- CPU time: 23 days 0 hours 39 minutes 22.9 seconds
