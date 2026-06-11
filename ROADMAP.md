# pyQRC Roadmap

Derived from a full repository audit (June 2026). Tasks are ordered by
milestone; within a milestone they can proceed in parallel unless a dependency
is listed. Sizes: **S** < 2 h, **M** ≈ half a day, **L** = 1–2 days.

**Definition of done for this roadmap:**
- CI builds the wheel, installs it in a clean venv, and smoke-tests `pyqrc` (and the wheel contains `run_g16.sh`).
- `pyqrc missing.log` exits non-zero with a message.
- `--freq` / `--freqnum` with no matching mode exits non-zero and writes **no** input file.
- `cclib` has version bounds in `pyproject.toml`.
- `pytest` reports zero skips when run in the repo.
- README options table matches `pyqrc --help` exactly.

---

## Decisions

Proposed answers to the audit's open questions. Treat as accepted defaults
unless overridden; tasks below assume them.

| # | Question | Decision (proposed) | Rationale |
|---|----------|---------------------|-----------|
| D1 | Keep or deprecate `--qcoord`? | **Keep, fix packaging (1.1), defer robustness work (3.1)** until there is evidence of use. | Shipping `run_g16.sh` is a one-line fix; the absence of bug reports for a long-broken mode suggests low usage, so don't invest beyond that yet. |
| D2 | Ship `examples/` in the wheel? | **No — repo-only.** Remove the dead `package-data`/`MANIFEST.in` references; optionally include examples in the sdist only. | They are test fixtures/docs (~MBs of QM output); installed users don't need them, and conftest already resolves repo-relative paths. |
| D3 | cclib pinning policy | **Bound now: `cclib>=1.8.1,<2`; raise the floor to the first release containing the ORCA 6 fix when it ships**, then make the ORCA 6 example a tested fixture. | Matches the compatibility matrix the README already documents; an upper bound protects against the known master-branch Q-Chem regression pattern. |
| D4 | `--freq` matching semantics | **Nearest mode within ±1 cm⁻¹ (configurable), print the matched mode; no match → error, no file, exit 1.** Release as **v2.2.0** with a release note. | Frequencies are reported to ~0.1 cm⁻¹; exact float equality can never be the intended UX. Previously "successful" no-op runs becoming errors is the point of the fix. |
| D5 | Python 3.13 support | **Yes — add 3.13 to both CI matrices and the classifiers.** | Pure-Python package; local development already happens on 3.13; risk is in cclib, which CI will reveal. |

Note (D3): per the README "ORCA 6 compatibility" section, cclib 1.8.1 is incompatible with ORCA 6 parsing, so ORCA 6 support is excluded until a fixed cclib release is published.

---

## Milestone 0 — Safety Net

| # | Task | Files | Acceptance criteria | Size | Risk | Deps |
|---|------|-------|---------------------|------|------|------|
| 0.1 ✅ | Commit pending working-tree changes (README ORCA-6 section, test except-clauses; track `examples/orca6/full_alkyne_TS.out`) | `README.md`, `tests/test_pyqrc.py`, `examples/orca6/` | `git status` clean; CI green | S | None | — |
| 0.2 ✅ | Add wheel-build + clean-venv install + smoke test to CircleCI: build wheel, assert it contains `pyqrc/run_g16.sh`, `pip install` it in a fresh venv, run `pyqrc` on `examples/g16/acetaldehyde.log` in a tmpdir, assert `acetaldehyde_QRC.com` is created | `.circleci/config.yml` | Job exists and **fails on current master** (proving it catches the packaging bug) | S | None | — |
| 0.3 ✅ | Characterization tests for desired CLI failure modes, marked `xfail(strict=True)`: missing file → exit 1 + message; unmatched `--freq` → exit 1, no file written; out-of-range `--freqnum` → exit 1, no file written | `tests/test_pyqrc.py` | Tests xfail today; flip to pass with 1.2/1.3 | S | None | — |

## Milestone 1 — Critical Fixes

| # | Task | Files | Acceptance criteria | Size | Risk | Deps |
|---|------|-------|---------------------|------|------|------|
| 1.1 ✅ | **Fix packaging** per D1/D2: add `pyqrc = ["run_g16.sh"]` to `[tool.setuptools.package-data]`; delete the dead `recursive-include pyqrc/examples *` from `MANIFEST.in` (add root `examples/` to the sdist only if desired) | `pyproject.toml`, `MANIFEST.in` | 0.2's CI job passes; `unzip -l` of the wheel shows `pyqrc/run_g16.sh` | S | Low | 0.2 |
| 1.2 ✅ | **Fix `--freq`/`--freqnum` matching** per D4: nearest-match within tolerance computed once before the shift loop; if the user requested a specific mode and nothing matched, raise (new `QRCModeError` or `QRCParseError` subclass) **before any file is written**; move `main()`'s "processing" message after successful generation. Bump version to 2.2.0 | `pyqrc/pyQRC.py` (`_calculate_shifts`, `QRCGenerator.__init__`, `main`), `tests/test_pyqrc.py` | 0.3 xfails flip to pass; `--freq -500` vs actual −500.123 displaces along that mode and says so; `--freqnum 99` errors with no file. Update the two tests that assert the old no-op behavior (`-f -500` cases) | M | **Medium — behavior change; needs release note** | 0.3 |
| 1.3 ✅ | **Collect files from `args.files`**, not a `sys.argv` re-scan: glob each entry (literal existing paths glob to themselves); no match → `x   <entry>: no such file`, exit code 1; warn-and-skip unexpected extensions; remove the dead `except IndexError` | `pyqrc/pyQRC.py` (`main`) | `pyqrc typo.log` exits 1 with a message; `--name backup.log` no longer captures the option value as an input | S | Low | 0.3 |

## Milestone 2 — High-Leverage Improvements

| # | Task | Files | Acceptance criteria | Size | Risk | Deps |
|---|------|-------|---------------------|------|------|------|
| 2.1 ✅ | Dependency bounds per D3: `cclib>=1.8.1,<2`, `numpy>=1.22`; README support-matrix note | `pyproject.toml`, `README.md` | Fresh install resolves to a combination the suite passes on | S | Low | — |
| 2.2 ✅ | Refactor `QRCGenerator.__init__` into `_parse()` → `compute_displacement()` (pure: copies coords, never mutates cclib data) → `write_files()`; `__init__` calls all three (backwards-compatible); use `with Logger(...)` at all call sites so handles can't leak on exceptions | `pyqrc/pyQRC.py` | All existing tests pass unmodified; new unit test obtains displaced coordinates with zero files written | M | Medium | 0.3 |
| 2.3 ✅ | De-skip the suite: remove existence-checks/skips for fixtures that live in the repo; cclib parse failures on repo examples become failures or strict xfails with a tracked reason | `tests/test_pyqrc.py`, `tests/conftest.py` | `pytest` reports 0 skips in-repo | M | Low | 2.1 |
| 2.4 ✅ | Single-source the version (keep one literal; others read it, e.g. `importlib.metadata`) | `pyproject.toml`, `pyqrc/__init__.py`, `pyqrc/pyQRC.py` | Version string appears in exactly one file | S | Low | — |

## Milestone 3 — Quality & Polish

| # | Task | Files | Acceptance criteria | Size | Risk | Deps |
|---|------|-------|---------------------|------|------|------|
| 3.1 | `--qcoord` robustness (only if D1 usage evidence appears): open the RUNIRC log once per run (not per file, which truncates it); try/except around per-file work; hoist `OutputData` out of the amplitude loop | `pyqrc/pyQRC.py` | Multi-file `--qcoord` keeps one complete log; one bad file doesn't abort the batch | M | Medium | 2.2 |
| 3.2 ✅ | ~~Rename `BOHR_TO_ANGSTROM` → `ANGSTROM_TO_BOHR`~~ (✅ done June 2026); error out instead of writing `METHOD None`/`BASIS None` in Q-Chem inputs; ~~delete~~ use `TERMINATION` to warn on abnormal Gaussian termination (it is public API per conventions) | `pyqrc/pyQRC.py` | No misnamed constant; Q-Chem path errors clearly when metadata is missing | S | Low | — |
| 3.3 ✅ | CI/lint tidy-up: pylint workflow on `[push, pull_request]`; orphan `[tool.ruff.lint]` config deleted (pylint is the enforced linter); Python 3.13 added to both CI matrices and classifiers (D5) | `.github/workflows/pylint.yml`, `pyproject.toml`, `.circleci/config.yml` | PRs from forks get linted; no orphan tool config; 3.13 green | S | Low | — |
| 3.4 ✅ | Audit the README options table against `pyqrc --help` (the `-v` → `-q` fix is already done) | `README.md` | Table matches `--help` verbatim | S | None | 1.2 |

## Quick wins (all S, can be done immediately)

1. ~~Fix the phantom `-v` row in the README options table~~ ✅ done June 2026
2. ~~Commit pending working-tree changes (0.1)~~ ✅ done June 2026
3. ~~Wheel-content CI check (0.2)~~ ✅ done June 2026
4. ~~Pylint on `pull_request` (part of 3.3)~~ ✅ done June 2026

**Milestones 0, 1, and 2 are complete** (June 2026): packaging fixed and
CI-guarded, missing files and unmatched `--freq`/`--freqnum` fail loudly,
dependencies bounded, version single-sourced, `QRCGenerator` split into
parse/compute/write with a `write=False` library mode, and the suite runs
with zero skips on release cclib. **v2.2.0 was released to PyPI on
2026-06-11** via the Trusted Publishing workflow (wheel verified to
contain `run_g16.sh`).
Milestone 3 is also complete except 3.1 (`--qcoord` robustness), which
stays deferred per D1 until there is evidence of `--qcoord` usage.

## Releasing

Publishing is automated via PyPI Trusted Publishing
(`.github/workflows/publish.yml`): create a GitHub Release with tag
`v<version>` (matching `pyproject.toml`) and the workflow builds, verifies
wheel contents and the tag/version match, and uploads to PyPI.
One-time setup required on pypi.org → project `pyqrc` → Publishing →
add trusted publisher (owner `patonlab`, repo `pyQRC`, workflow
`publish.yml`, environment `pypi`).
