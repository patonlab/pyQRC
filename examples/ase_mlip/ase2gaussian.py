"""Bridge ASE frequency results to a Gaussian-style log file for pyQRC.

pyQRC (like GoodVibes and other cclib-based tools) reads QM program output
files. When the Hessian comes from an ASE calculator (e.g. a machine-learned
interatomic potential) there is no such file, so this helper writes the
geometry, frequencies, and normal modes in the Gaussian output format that
cclib parses. The result works with ``pyqrc`` exactly like a real Gaussian
frequency job.

Only the blocks cclib and pyQRC actually read are written: the route
section, charge/multiplicity, the Standard orientation table, the harmonic
frequency block (3 modes per column group, Gaussian layout), and the normal
termination line.
"""

import numpy as np

CM1_TO_MDYNE_PER_A = 5.89141e-7  # k = mu * (2*pi*c*nu)^2 in mDyne/A with amu, cm-1
EV_TO_CM1 = 8065.54429


def extract_vibrations(vib, min_freq=100.0):
    """Pull true vibrational modes out of an ase.vibrations.Vibrations run.

    ASE returns all 3N modes including translations/rotations (near-zero
    frequencies) and reports imaginary modes as complex energies. This
    converts to the Gaussian convention pyQRC expects: signed wavenumbers
    (negative = imaginary) with the trans/rot modes dropped, leaving the
    3N-6(5) true vibrations.

    Args:
        vib: A completed ase.vibrations.Vibrations object.
        min_freq: Modes with |frequency| below this (cm-1) are discarded
            as translations/rotations. Raise it if your structure is far
            from stationary; lower it for floppy systems with genuine
            low-frequency modes.

    Returns:
        (frequencies, modes): signed wavenumbers in cm-1 sorted ascending,
        and the matching (nmodes, natoms, 3) Cartesian displacements.
    """
    energies = np.asarray(vib.get_energies())  # eV, complex for imaginary
    wavenumbers = energies * EV_TO_CM1
    freqs = np.where(np.abs(wavenumbers.imag) > 1e-6,
                     -np.abs(wavenumbers.imag), wavenumbers.real)
    modes = np.array([vib.get_mode(i) for i in range(len(freqs))])
    keep = np.abs(freqs) > min_freq
    return freqs[keep], modes[keep]


def write_gaussian_freq_log(filename, atoms, frequencies, modes,
                            energy=None, route='# opt'):
    """Write an ASE frequency calculation as a cclib-parseable Gaussian log.

    Args:
        filename: Path of the .log file to create.
        atoms: ase.Atoms with the geometry the frequencies belong to.
        frequencies: Array of vibrational frequencies in cm-1. Use negative
            values for imaginary modes (Gaussian convention). Pass only the
            3N-6 true vibrations, not translations/rotations.
        modes: Array (nmodes, natoms, 3) of Cartesian mode displacements in
            the same order as ``frequencies``. Any normalization is accepted;
            each mode is re-normalized to unit Cartesian norm, matching what
            Gaussian prints.
        energy: Optional electronic energy in Hartree for an "SCF Done" line.
        route: Route line echoed into inputs that pyQRC generates from this
            log (the default ``# opt`` suits displace-then-reoptimize use).
    """
    numbers = atoms.get_atomic_numbers()
    masses = atoms.get_masses()
    coords = atoms.get_positions()
    natoms = len(atoms)

    frequencies = np.asarray(frequencies, dtype=float)
    modes = np.asarray(modes, dtype=float).reshape(len(frequencies), natoms, 3)
    # Gaussian prints displacements normalized to unit Cartesian norm
    norms = np.linalg.norm(modes.reshape(len(frequencies), -1), axis=1)
    modes = modes / norms[:, None, None]
    # Reduced mass and force constant from the normalized displacements
    red_masses = 1.0 / np.einsum('mij,i->m', modes**2, 1.0 / masses)
    frc_consts = CM1_TO_MDYNE_PER_A * frequencies**2 * red_masses

    lines = []
    out = lines.append
    out(' Entering Gaussian System, Link 0=g16')
    out(' This file was written by ase2gaussian.py (pyQRC examples), not by')
    out(' Gaussian. It mimics the output format of Gaussian, Inc. so that')
    out(' cclib-based tools can read ASE/MLIP frequency results.')
    out(' ' + '-' * 70)
    out(f' {route}')
    out(' ' + '-' * 70)
    out(f' Charge =  {atoms.get_initial_charges().sum():.0f} Multiplicity = 1')
    out(f' NAtoms=  {natoms:4d} NActive=  {natoms:4d}')
    out('                         Standard orientation:                         ')
    out(' ' + '-' * 69)
    out(' Center     Atomic      Atomic             Coordinates (Angstroms)')
    out(' Number     Number       Type             X           Y           Z')
    out(' ' + '-' * 69)
    for i in range(natoms):
        out(f'    {i + 1:3d}        {numbers[i]:3d}           0    '
            f'{coords[i][0]:12.6f}{coords[i][1]:12.6f}{coords[i][2]:12.6f}')
    out(' ' + '-' * 69)
    if energy is not None:
        out(f' SCF Done:  E(MLIP) = {energy:16.9f}     A.U. after    1 cycles')

    nimag = int((frequencies < 0).sum())
    out(f' ******  {nimag:4d} imaginary frequencies (negative Signs) ****** ')
    out(' Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering')
    out(' activities (A**4/AMU), depolarization ratios for plane and unpolarized')
    out(' incident light, reduced masses (AMU), force constants (mDyne/A),')
    out(' and normal coordinates:')
    for start in range(0, len(frequencies), 3):
        block = range(start, min(start + 3, len(frequencies)))
        out(''.join(f'{m + 1:>23d}' for m in block))
        out(''.join(f'{"A":>23}' for _ in block))
        out(' Frequencies --' + ''.join(f'{frequencies[m]:11.4f}            '
                                        for m in block).rstrip())
        out(' Red. masses --' + ''.join(f'{red_masses[m]:11.4f}            '
                                        for m in block).rstrip())
        out(' Frc consts  --' + ''.join(f'{frc_consts[m]:11.4f}            '
                                        for m in block).rstrip())
        out(' IR Inten    --' + ''.join(f'{0.0:11.4f}            '
                                        for m in block).rstrip())
        out('  Atom  AN' + '      X      Y      Z  ' * len(list(block)))
        for i in range(natoms):
            row = f'{i + 1:6d}{numbers[i]:4d}  '
            for m in block:
                row += ''.join(f'{modes[m][i][k]:7.2f}' for k in range(3)) + '  '
            out(row.rstrip())

    out(' Normal termination of Gaussian 16.')
    # cclib's Gaussian parser logs "Unexpectedly encountered end of logfile"
    # unless a blank line follows the termination line
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n\n')
