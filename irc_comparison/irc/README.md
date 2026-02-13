# IRC Calculations

## Contents

This directory contains Gaussian input files (`.com`) and output log files (`.log`) for intrinsic reaction coordinate (IRC) calculations.

- `completed/` - IRC calculations that completed successfully
- `failed/` - IRC calculations that did not converge
- `IRC_data.txt` - Summary of optimized energies, reaction coordinates, and SMILES for each reaction
- `parse_irc.py` - Script used to parse IRC output data

## Timing

### Completed
- Files: 562
- CPU time: 81 days 6 hours 12 minutes 38.1 seconds

### Failed
- Files: 441
- CPU time: 74 days 13 hours 56 minutes 43.8 seconds

All calculations were performed at the wB97XD/6-31G(d) level of theory with Gaussian 16 C.01. Initially IRC calculations were attempted with irc(calcfc,maxpoints=200,maxcycles=200, update=10). Failures to converge were retried with more frequent Hessian updates, which resolved the convergence failure in most cases.
