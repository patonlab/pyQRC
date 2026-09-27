# Changelog

## 2.4.0 (2026-09-27)

This release makes the input files pyQRC writes run correctly without hand
editing, reads ORCA 6 outputs with a plain `pip install pyqrc`, and adds a
Python API for frequencies computed outside QM programs (e.g. ASE and
machine-learned potentials). It includes the unreleased 2.3.0 changes.

### Changes in behaviour

Each of these changes what pyQRC writes or how it exits. How to get the
previous behaviour back is given where it is possible.

- **Cloned routes now optimize to a minimum.** When `--route` is not given,
  saddle-point keywords from the original job are replaced so that a QRC
  from a transition state relaxes to the reactant or product instead of
  searching for the TS again: Gaussian `opt=(ts,calcfc,noeigentest)` becomes
  `opt=calcfc` (`ts`, `saddle=N`, `qst2`/`qst3`, `noeigentest`, `readfc` and
  `guess=read` are removed), ORCA `OptTS` becomes `Opt`, and `opt` is added
  to frequency-only jobs. A note is printed whenever the route changes.
  *Previous behaviour:* pass the original route with `--route`, which is
  always used exactly as given.
- **Default amplitude is 0.3** (was 0.2), the value that matched IRC best in
  the README benchmark. *Previous behaviour:* `--amp 0.2`.
- **`--mem` is the total memory for every program.** For ORCA, `%maxcore`
  (memory per core) is now `--mem` divided by `--nproc`; previously
  `--nproc 4 --mem 8GB` requested 32 GB. Units are case-insensitive
  (`8gb` was read as 8 MB) and fractions work (`1.5GB` was read as 1 GB).
  Invalid values are rejected.
- **Files without imaginary frequencies are skipped** unless `--freq` or
  `--freqnum` asks for a mode; previously an undisplaced copy of the
  geometry was written. `--auto` now only matters together with those
  options, where it still skips.
- **Existing files are never replaced** by the command line: a file whose
  outputs already exist fails with a message, and with `--both` nothing is
  written unless both files are new. *Previous behaviour:* `--overwrite`.
  (The Python API still overwrites by default.)
- **Outputs from programs pyQRC cannot write inputs for** (e.g. Psi4,
  NWChem, Turbomole) now give an `.xyz` file instead of a `.com` file that
  contained only coordinates.
- **Exit code 1** for an output with no frequency data (was 0), since it is
  almost always the wrong file.
- A leading `#` in a Gaussian `--route` is no longer doubled (`##p`).

### New

- **Input carried over from the original job**: Gaussian sections after the
  geometry (Gen/GenECP basis sets and ECPs, ModRedundant, SCRF=Read input)
  from the `.com`/`.gjf` next to the output when its route matches; all
  ORCA `!` lines and `%` blocks (`%cpcm`, `%basis`, `%geom`, ...) from the
  input ORCA echoes; and every Q-Chem `$rem` setting and extra section
  (`$basis`, `$ecp`, `$smx`, ...) in both the optimization and frequency
  jobs. A warning is printed when a Gaussian route needs input that could
  not be found.
- **ORCA 6 with released cclib**: cclib 1.8.1 cannot read ORCA 6 outputs,
  so pyQRC reads ORCA files cclib fails on with its own reader
  (`pyqrc/orca_reader.py`). The GitHub-master cclib workaround is no longer
  needed.
- `--both`: write inputs displaced in both directions (`_QRCF` and `_QRCR`).
- `--outdir DIR`: write the new files into a directory.
- `--overwrite`: replace existing files.
- `--xyz`: write the displaced geometry as an `.xyz` file.
- Python API: `QRCGenerator.from_arrays()` (atomic numbers, coordinates,
  frequencies and modes) and `QRCGenerator.from_ase()` (an ASE
  `Vibrations` object, without writing an intermediate file);
  `vibrations_from_ase()`; `parse_output()`; `QRCGenerator.output_paths()`;
  `overwrite=`, `outdir=` and `xyz=` options; `QRCFileExistsError`.
- `.LOG`/`.OUT` extensions are accepted in any case.
- Warnings when the displaced structure has new close contacts between
  atoms, when fewer than 3N-6 normal modes were read, and when a geometry
  was not displaced (Python API).
- Example notebooks for Gaussian 16, ORCA 5, ORCA 6, Q-Chem and ASE/MLIP
  (MACE-OFF), including a check of the bonds that form and break between
  the two QRC directions.

### Fixed

- Q-Chem with an effective core potential: cclib derives the charge from the
  electron count (+46 for iodobenzene with an iodine ECP); the charge and
  multiplicity from the input are now used.
- Q-Chem inputs no longer lose `$rem` settings such as `DFT_D`, solvation
  and SCF options, and are never written with `METHOD None`.
- A Gaussian opt+freq output cut off during the frequency step was treated
  as terminated normally; every job step must now terminate.
- Non-UTF-8 characters in an output no longer crash pyQRC, and an
  unexpected error with one file no longer stops a batch.
- The clash check ignored contacts already present before displacement,
  so nitriles (C≡N) no longer trigger it.
- Atomic masses of Kr and Rb corrected; Cu covalent radius added.

### Deprecated

- `--qcoord` and `--nummodes` (running Gaussian single points directly)
  print a deprecation warning and will be removed in 3.0.

### Development

- Tests grew from 125 (3 skipped) to 382, with no skips on released cclib: golden-file
  tests of every generated input (`pytest tests/test_golden.py
  --update-golden` to regenerate), tests of problem outputs (truncated,
  multi-step, CRLF, empty, binary), and 40 real Gaussian 16, ORCA 5/6 and
  Q-Chem 6 outputs from the GoodVibes test suite (transition states,
  open-shell and charged species, linear molecules, GenECP, solvation).
- CI runs the example notebooks with nbval, smoke-tests an ORCA 6 output
  from the built wheel, and a weekly job runs the tests on cclib master.

## 2.2.0 (2026-06-11)

- `--freq` matches the nearest mode within 1 cm⁻¹, and `--freq`/`--freqnum`
  with no matching mode exit with an error without writing a file.
- A missing input file exits with an error; option values are no longer
  mistaken for input files.
- `run_g16.sh` is included in the wheel; cclib is bounded to `>=1.8.1,<2`.
