# Real outputs from the GoodVibes test suite

These Gaussian 16, ORCA 5, ORCA 6 and Q-Chem 6 outputs, with their inputs,
are copied unchanged from the test fixtures of
[GoodVibes](https://github.com/patonlab/GoodVibes/tree/master/tests)
(commit `8f8f62e`, MIT licence, see `LICENSE.GoodVibes`). The numbers in the
file names follow GoodVibes' scheme: 01-43 standard calculations, 44-50
transition states.

They were chosen for what they exercise in pyQRC:

| Files | What they test |
| --- | --- |
| 44-48 | Transition states (SN2, Diels-Alder, H abstraction, E2, NH3 inversion) in every program |
| 04, 16 | Open-shell and charged species (benzene radical cation, superoxide) |
| 22, 16 | Linear molecules (3N-5 modes) |
| 24 (Gaussian, Q-Chem), 25 (ORCA) | GenECP / ECP basis sets carried into the new input |
| 19, 29 | SMD and CPCM solvation |
| orca6/37 | Third-order saddle point: all three imaginary modes must be kept |
| orca6/21 | xTB job without a "Total Charge" line |
| g16/46 | Output that stops before the frequency step's "Normal termination" |

`tests/test_goodvibes_fixtures.py` lists the expected atoms, modes,
imaginary modes, charge and multiplicity for each file, checked against the
programs' own output. Their generated inputs are in `tests/golden/goodvibes/`.
