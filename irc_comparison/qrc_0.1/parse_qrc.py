#!/usr/bin/env python3
"""Parse QRC (Quick Reaction Coordinate) log files and report key energies.

Usage:
    parse_qrc.py <ts_dir> <qrc_dir>       # e.g. parse_qrc.py ts qrc03
    parse_qrc.py <rev> <ts> <fwd>          # explicit file paths
"""

import re
import sys
import glob
import os
from openbabel import openbabel as ob

ELEMENTS = {
    1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne',
    11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 18:'Ar',
    19:'K', 20:'Ca', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn',
    35:'Br', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 53:'I', 78:'Pt', 79:'Au',
}


def get_last_scf(filename):
    """Extract the last SCF Done energy from a Gaussian log file."""
    energy = None
    with open(filename) as f:
        for line in f:
            if "SCF Done" in line:
                m = re.search(r"SCF Done:.*=\s+([-\d.]+)", line)
                if m:
                    energy = float(m.group(1))
    return energy


def normal_termination(filename):
    """Check if the log file terminated normally."""
    with open(filename) as f:
        text = f.read()
    return "Normal termination" in text


def get_smiles(filename):
    """Extract the last geometry from a Gaussian log and return canonical SMILES."""
    with open(filename) as f:
        text = f.read()
    blocks = list(re.finditer(
        r"Standard orientation:.*?\n\s*-+\n.*?\n.*?\n\s*-+\n(.*?)\n\s*-+",
        text, re.DOTALL
    ))
    if not blocks:
        return ""
    lines = blocks[-1].group(1).strip().split('\n')
    atoms = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 6:
            anum = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            sym = ELEMENTS.get(anum, 'X')
            atoms.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    xyz = f"{len(atoms)}\n\n" + "\n".join(atoms) + "\n"
    conv = ob.OBConversion()
    conv.SetInFormat("xyz")
    conv.SetOutFormat("can")
    mol = ob.OBMol()
    conv.ReadString(mol, xyz)
    return conv.WriteString(mol).strip()


def find_latest(pattern):
    """Find the latest continuation file that terminated normally."""
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    for f in reversed(candidates):
        if normal_termination(f):
            return f
    return candidates[-1]


def parse_qrc(rev_file, ts_file, fwd_file, ts_id=""):
    """Parse three QRC files and print energy summary. Returns True on success."""
    files = {"Beginning": rev_file, "TS": ts_file, "End": fwd_file}

    abs_energies = {}
    for label, fname in files.items():
        if not os.path.isfile(fname):
            print(f"  {ts_id}: File not found: {fname}")
            return False
        if not normal_termination(fname):
            print(f"  {ts_id}: Warning: {fname} did not terminate normally")
        energy = get_last_scf(fname)
        if energy is None:
            print(f"  {ts_id}: No SCF energy found in {fname}")
            return False
        abs_energies[label] = energy

    ref = abs_energies["Beginning"]
    labels = ["Beginning", "TS", "End"]

    smiles = {}
    for label in ["Beginning", "End"]:
        smiles[label] = get_smiles(files[label])

    if ts_id:
        print(f"{ts_id}")
    print(f"{'':12s} {'File':>40s} {'Abs (Eh)':>16s} {'Rel (kcal/mol)':>16s}   {'SMILES'}")
    for label in labels:
        e = abs_energies[label]
        rel = (e - ref) * 627.5095
        smi = smiles.get(label, "")
        print(f"{label:12s} {files[label]:>40s} {e:16.6f} {rel:16.2f}   {smi}")
    print()
    return True


def batch_mode(ts_dir, qrc_dir):
    """Discover all TS IDs and parse matching QRC triplets."""
    # Extract QRC number from directory name (e.g. qrc03 -> 03)
    qrc_num = re.search(r'(\d+)', os.path.basename(qrc_dir.rstrip("/")))
    if not qrc_num:
        print(f"Cannot determine QRC number from directory name: {qrc_dir}")
        sys.exit(1)
    qrc_num = qrc_num.group(1)

    # Find all TS IDs from the ts directory
    ts_logs = sorted(glob.glob(os.path.join(ts_dir, "ts_*.log")))
    if not ts_logs:
        print(f"No ts_*.log files found in {ts_dir}")
        sys.exit(1)

    success = 0
    skipped = 0
    for ts_log in ts_logs:
        basename = os.path.basename(ts_log)
        # Extract ts_id (e.g. ts_001 from ts_001.log or ts_001_ii.log)
        m = re.match(r"(ts_\d+)", basename)
        if not m:
            continue
        ts_id = m.group(1)

        ts_file = find_latest(os.path.join(ts_dir, f"{ts_id}.log")) or \
                  find_latest(os.path.join(ts_dir, f"{ts_id}_*.log"))
        rev_file = find_latest(os.path.join(qrc_dir, f"{ts_id}_QRCR{qrc_num}*.log"))
        fwd_file = find_latest(os.path.join(qrc_dir, f"{ts_id}_QRCF{qrc_num}*.log"))

        if not ts_file or not rev_file or not fwd_file:
            missing = []
            if not ts_file: missing.append("TS")
            if not rev_file: missing.append("reverse")
            if not fwd_file: missing.append("forward")
            print(f"  {ts_id}: skipping (missing {', '.join(missing)})")
            skipped += 1
            continue

        if parse_qrc(rev_file, ts_file, fwd_file, ts_id):
            success += 1
        else:
            skipped += 1

    print(f"Summary: {success} parsed, {skipped} skipped out of {len(ts_logs)} TS files")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        arg1, arg2 = sys.argv[1], sys.argv[2]
        if os.path.isdir(arg1) and os.path.isdir(arg2):
            batch_mode(arg1, arg2)
        else:
            print(f"Directories not found: {arg1}, {arg2}")
            sys.exit(1)

    elif len(sys.argv) == 4:
        parse_qrc(sys.argv[1], sys.argv[2], sys.argv[3])

    else:
        print(f"Usage: {sys.argv[0]} <ts_dir> <qrc_dir>")
        print(f"       {sys.argv[0]} <reverse.log> <ts.log> <forward.log>")
        sys.exit(1)
