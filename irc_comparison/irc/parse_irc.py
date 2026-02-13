#!/usr/bin/env python3
"""Parse Gaussian IRC log files and report key energies along the reaction path."""

import re
import sys
import glob
from openbabel import openbabel as ob

ELEMENTS = {
    1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne',
    11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 18:'Ar',
    19:'K', 20:'Ca', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn',
    35:'Br', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 53:'I', 78:'Pt', 79:'Au',
}


def geom_to_smiles(text, pos):
    """Extract Input orientation block at pos and return canonical SMILES."""
    m = re.search(
        r"Input orientation:.*?\n\s*-+\n.*?\n.*?\n\s*-+\n(.*?)\n\s*-+",
        text[pos:], re.DOTALL
    )
    if not m:
        return ""
    atoms = []
    for line in m.group(1).strip().split('\n'):
        parts = line.split()
        if len(parts) >= 6:
            anum = int(parts[1])
            sym = ELEMENTS.get(anum, 'X')
            atoms.append(f"{sym} {parts[3]} {parts[4]} {parts[5]}")
    xyz = f"{len(atoms)}\n\n" + "\n".join(atoms) + "\n"
    conv = ob.OBConversion()
    conv.SetInFormat("xyz")
    conv.SetOutFormat("can")
    mol = ob.OBMol()
    conv.ReadString(mol, xyz)
    return conv.WriteString(mol).strip()


def get_endpoint_smiles(text):
    """Extract SMILES for the forward and reverse IRC endpoints."""
    io_positions = [m.start() for m in re.finditer(r"Input orientation:", text)]
    smiles = {"Beginning": "", "End": ""}
    try:
        fwd_pos = text.index("Calculation of FORWARD path complete")
        fwd_geom = max(p for p in io_positions if p < fwd_pos)
        smiles["End"] = geom_to_smiles(text, fwd_geom)
    except (ValueError, StopIteration):
        pass
    try:
        rev_pos = text.index("Calculation of REVERSE path complete")
        rev_geom = max(p for p in io_positions if p < rev_pos)
        smiles["Beginning"] = geom_to_smiles(text, rev_geom)
    except (ValueError, StopIteration):
        pass
    return smiles


def parse_irc_log(filename):
    with open(filename) as f:
        text = f.read()

    # Extract TS absolute energy
    match = re.search(
        r"Energies reported relative to the TS energy of\s+([-\d.]+)", text
    )
    if not match:
        print(f"  No IRC summary found in {filename}")
        return
    ts_energy = float(match.group(1))

    # Extract the reaction path table (only within the IRC summary section)
    summary_match = re.search(
        r"Summary of reaction path following\s*\n"
        r"\s*-+\s*\n"
        r"\s*Energy\s+RxCoord\s*\n"
        r"(.*?)\n\s*-{10,}",
        text, re.DOTALL
    )
    if not summary_match:
        print(f"  No reaction path table found in {filename}")
        return
    table_text = summary_match.group(1)
    rows = re.findall(
        r"^\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s*$", table_text, re.MULTILINE
    )
    if not rows:
        print(f"  No reaction path data found in {filename}")
        return

    energies_rel = [float(r[0]) for r in rows]
    rxcoords = [float(r[1]) for r in rows]

    # Find TS index (where relative energy is 0.0)
    ts_idx = energies_rel.index(0.0)

    # Beginning, TS, End
    idxs = [0, ts_idx, -1]
    labels = ["Beginning", "TS", "End"]

    abs_energies = [ts_energy + energies_rel[i] for i in idxs]
    # Relative to beginning, converted to kcal/mol
    ref = abs_energies[0]
    rel_energies = [(e - ref) * 627.5095 for e in abs_energies]
    coords = [rxcoords[i] for i in idxs]

    smiles = get_endpoint_smiles(text)

    print(f"File: {filename}")
    print(f"{'':12s} {'Abs (Eh)':>16s} {'Rel (kcal/mol)':>16s} {'RxCoord':>12s}   {'SMILES'}")
    for label, abs_e, rel_e, coord in zip(labels, abs_energies, rel_energies, coords):
        smi = smiles.get(label, "")
        print(f"{label:12s} {abs_e:16.6f} {rel_e:16.2f} {coord:12.5f}   {smi}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile(s)>")
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        files.extend(glob.glob(arg))

    if not files:
        print("No matching files found.")
        sys.exit(1)

    for f in sorted(files):
        parse_irc_log(f)
