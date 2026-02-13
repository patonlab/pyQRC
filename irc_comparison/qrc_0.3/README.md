# QRC Displacement 0.3

## Contents

This directory contains Gaussian input files (`.com`), output log files (`.log`), and pyQRC displacement files (`.qrc`) for quick reaction coordinate (pyQRC) calculations using a displacement amplitude of 0.3.

- `ts_NNN_QRCF03.*` - Forward QRC-displaced structures
- `ts_NNN_QRCR03.*` - Reverse QRC-displaced structures
- `*_ii.*` - Re-submitted calculations where the initial optimization did not converge
- `QRC03_data.txt` - Summary of optimized energies and SMILES for each reaction

## How Inputs Were Generated

Starting from transition state geometries in `../ts/`, QRC-displaced input files were generated using pyQRC with a displacement amplitude of 0.3:

```
python -m pyqrc ts.log --amp 0.3 --append QRCF03
```

All subsequent geometry optimizations were performed at the wB97XD/6-31G(d) level of theory with Gaussian 16 C.01.

## Timing

- Files: 1220 (1075 opt+freq, 145 opt only)
- CPU time: 18 days 22 hours 33 minutes 17.3 seconds
