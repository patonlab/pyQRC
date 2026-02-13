# QRC Displacement 0.5

## Contents

This directory contains Gaussian input files (`.com`), output log files (`.log`), and pyQRC displacement files (`.qrc`) for quick reaction coordinate (pyQRC) calculations using a displacement amplitude of 0.5.

- `ts_NNN_QRCF05.*` - Forward QRC-displaced structures
- `ts_NNN_QRCR05.*` - Reverse QRC-displaced structures
- `*_ii.*` - Re-submitted calculations where the initial optimization did not converge
- `QRC05_data.txt` - Summary of optimized energies and SMILES for each reaction

## How Inputs Were Generated

Starting from transition state geometries in `../ts/`, QRC-displaced input files were generated using pyQRC with a displacement amplitude of 0.5:

```
python -m pyqrc ts.log --amp 0.5 --append QRCF05
```

All subsequent geometry optimizations were performed at the wB97XD/6-31G(d) level of theory with Gaussian 16 C.01.

## Timing

- Files: 1214 (1118 opt+freq, 96 opt only)
- CPU time: 19 days 9 hours 14 minutes 40.4 seconds
