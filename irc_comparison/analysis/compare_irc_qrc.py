#!/usr/bin/env python3
"""Compare IRC and QRC results: SMILES identity, barriers, and reaction energies.

Usage: compare_irc_qrc.py [--ts-dir DIR | --freq-file FILE] IRC_data.txt QRC01_data.txt [QRC03_data.txt ...]

Options:
    --ts-dir DIR      Directory of TS Gaussian log files (extracts imaginary frequencies)
    --freq-file FILE  Text file of imaginary frequencies (ts_id  frequency), as alternative to --ts-dir
"""

import glob
import re
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

HA_TO_KCAL = 627.5095


def canonical(smi):
    """Return RDKit canonical SMILES, or original string if parsing fails."""
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return smi
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def parse_irc_file(filename):
    """Parse IRC_data.txt into dict keyed by ts_id."""
    data = {}
    with open(filename) as f:
        text = f.read()

    blocks = re.split(r"\n(?=File:)", text.strip())
    for block in blocks:
        # Extract ts_id from filename
        m = re.search(r"File:.*/(ts_\d+)", block)
        if not m:
            continue
        ts_id = m.group(1)

        energies = {}
        smiles = {}
        rxcoords = {}
        for line in block.split('\n'):
            for label in ["Beginning", "TS", "End"]:
                if line.startswith(label):
                    parts = line.split()
                    energies[label] = float(parts[1])
                    # RxCoord is 4th column (index 3)
                    if len(parts) >= 4:
                        rxcoords[label] = float(parts[3])
                    # SMILES is last token (if present)
                    if label != "TS" and len(parts) >= 5:
                        smiles[label] = parts[-1]
                    break

        if "Beginning" in energies and "TS" in energies and "End" in energies:
            path_len = abs(rxcoords.get("End", 0)) + abs(rxcoords.get("Beginning", 0))
            data[ts_id] = {"energies": energies, "smiles": smiles, "path_length": path_len}

    return data


def parse_qrc_file(filename):
    """Parse QRC_data.txt into dict keyed by ts_id."""
    data = {}
    with open(filename) as f:
        text = f.read()

    blocks = re.split(r"\n(?=ts_\d+\n)", text.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        m = re.match(r"(ts_\d+)$", lines[0])
        if not m:
            continue
        ts_id = m.group(1)

        energies = {}
        smiles = {}
        for line in lines:
            for label in ["Beginning", "TS", "End"]:
                if line.startswith(label):
                    # Extract absolute energy: find pattern like -NNN.NNNNNN
                    e_match = re.search(r"(-\d+\.\d{6})", line)
                    if e_match:
                        energies[label] = float(e_match.group(1))
                    # SMILES is the last whitespace-delimited token
                    parts = line.split()
                    if label != "TS" and parts:
                        smi = parts[-1]
                        if not re.match(r"^-?\d", smi):
                            smiles[label] = smi
                    break

        if "Beginning" in energies and "TS" in energies and "End" in energies:
            data[ts_id] = {"energies": energies, "smiles": smiles}

    return data


def smiles_match(smi1, smi2):
    """Check if two SMILES represent the same molecule."""
    if not smi1 or not smi2:
        return False
    return canonical(smi1) == canonical(smi2)


def compute_energetics(energies):
    """Return (barrier, rxn_energy) in kcal/mol from absolute energies."""
    barrier = (energies["TS"] - energies["Beginning"]) * HA_TO_KCAL
    rxn_e = (energies["End"] - energies["Beginning"]) * HA_TO_KCAL
    return barrier, rxn_e


def structural_change(smi_beg, smi_end):
    """Compute structural change metrics between reactant and product SMILES.

    Returns dict with:
      tanimoto_dist: 1 - Tanimoto similarity of Morgan fingerprints
      delta_bonds: abs difference in total bond count
      delta_frags: abs difference in number of fragments
      bond_change: bonds_broken + bonds_formed (estimated from per-fragment analysis)
    Returns None if either SMILES fails to parse.
    """
    mol_beg = Chem.MolFromSmiles(smi_beg, sanitize=False)
    mol_end = Chem.MolFromSmiles(smi_end, sanitize=False)
    if mol_beg is None or mol_end is None:
        return None
    try:
        Chem.SanitizeMol(mol_beg)
        Chem.SanitizeMol(mol_end)
    except Exception:
        return None

    # Morgan fingerprints (radius 2, ~ ECFP4)
    fp_beg = AllChem.GetMorganFingerprintAsBitVect(mol_beg, 2, nBits=2048)
    fp_end = AllChem.GetMorganFingerprintAsBitVect(mol_end, 2, nBits=2048)
    tanimoto = DataStructs.TanimotoSimilarity(fp_beg, fp_end)

    n_bonds_beg = mol_beg.GetNumBonds()
    n_bonds_end = mol_end.GetNumBonds()

    frags_beg = smi_beg.count('.') + 1
    frags_end = smi_end.count('.') + 1

    return {
        "tanimoto_dist": 1.0 - tanimoto,
        "delta_bonds": abs(n_bonds_end - n_bonds_beg),
        "delta_frags": abs(frags_end - frags_beg),
    }


def align_qrc_to_irc(irc_entry, qrc_entry):
    """Align QRC direction to IRC. Returns (aligned_energies, aligned_smiles, flipped)."""
    irc_smi = irc_entry["smiles"]
    qrc_smi = qrc_entry["smiles"]
    qrc_e = qrc_entry["energies"]

    # Check direct match
    beg_match = smiles_match(irc_smi.get("Beginning", ""), qrc_smi.get("Beginning", ""))
    end_match = smiles_match(irc_smi.get("End", ""), qrc_smi.get("End", ""))

    # Check flipped match
    beg_flip = smiles_match(irc_smi.get("Beginning", ""), qrc_smi.get("End", ""))
    end_flip = smiles_match(irc_smi.get("End", ""), qrc_smi.get("Beginning", ""))

    if beg_match and end_match:
        return qrc_e, qrc_smi, False
    elif beg_flip and end_flip:
        # Flip QRC: swap Beginning and End
        flipped_e = {
            "Beginning": qrc_e["End"],
            "TS": qrc_e["TS"],
            "End": qrc_e["Beginning"],
        }
        flipped_smi = {
            "Beginning": qrc_smi.get("End", ""),
            "End": qrc_smi.get("Beginning", ""),
        }
        return flipped_e, flipped_smi, True
    elif beg_match or end_match:
        # Partial match - one endpoint agrees
        return qrc_e, qrc_smi, False
    elif beg_flip or end_flip:
        # Partial flipped match
        flipped_e = {
            "Beginning": qrc_e["End"],
            "TS": qrc_e["TS"],
            "End": qrc_e["Beginning"],
        }
        flipped_smi = {
            "Beginning": qrc_smi.get("End", ""),
            "End": qrc_smi.get("Beginning", ""),
        }
        return flipped_e, flipped_smi, True
    else:
        # No match at all
        return qrc_e, qrc_smi, False


def parse_imag_freqs(ts_dir):
    """Extract imaginary frequency from each TS log file. Returns {ts_id: freq}."""
    freqs = {}
    for logfile in sorted(glob.glob(os.path.join(ts_dir, "ts_*.log"))):
        m = re.match(r"(ts_\d+)", os.path.basename(logfile))
        if not m:
            continue
        ts_id = m.group(1)
        if ts_id in freqs:
            continue  # already got this one
        with open(logfile) as f:
            for line in f:
                if "Frequencies --" in line:
                    parts = line.split()
                    # First frequency after "--"
                    try:
                        idx = parts.index("--") + 1
                        freq = float(parts[idx])
                        if freq < 0:
                            freqs[ts_id] = freq
                    except (ValueError, IndexError):
                        pass
                    break
    return freqs


def parse_freq_file(filename):
    """Read imaginary frequencies from a text file. Format: ts_id  frequency per line."""
    freqs = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ts_id = parts[0]
                try:
                    freqs[ts_id] = float(parts[1])
                except ValueError:
                    continue
    return freqs


def main():
    args = sys.argv[1:]
    ts_dir = None
    freq_file = None
    if "--ts-dir" in args:
        idx = args.index("--ts-dir")
        ts_dir = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
    if "--freq-file" in args:
        idx = args.index("--freq-file")
        freq_file = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if len(args) < 2:
        print(f"Usage: {sys.argv[0]} [--ts-dir DIR | --freq-file FILE] IRC_data.txt QRC_data.txt [QRC_data.txt ...]")
        sys.exit(1)

    irc_file = args[0]
    qrc_files = args[1:]

    irc_data = parse_irc_file(irc_file)

    # Parse all QRC files and find the common set across ALL inputs
    all_qrc = {}
    for qrc_file in qrc_files:
        qrc_label = os.path.basename(qrc_file).replace("_data.txt", "")
        all_qrc[qrc_label] = (qrc_file, parse_qrc_file(qrc_file))

    common_all = set(irc_data)
    for qrc_label, (qrc_file, qrc_data) in all_qrc.items():
        common_all &= set(qrc_data)

    # Filter out reactions with near-zero IRC path length (failed IRCs)
    MIN_PATH_LENGTH = 1.0  # amu^1/2 bohr
    short_irc = [ts_id for ts_id in common_all
                 if irc_data[ts_id].get("path_length", 0) < MIN_PATH_LENGTH]
    if short_irc:
        common_all -= set(short_irc)
        print(f"Excluded {len(short_irc)} reactions with IRC path length < {MIN_PATH_LENGTH}: {', '.join(sorted(short_irc))}")

    common_all = sorted(common_all)

    print(f"IRC: {len(irc_data)} reactions from {irc_file}")
    for qrc_label, (qrc_file, qrc_data) in all_qrc.items():
        print(f"{qrc_label}: {len(qrc_data)} reactions from {qrc_file}")
    print(f"Common to all: {len(common_all)} reactions")

    # Parse imaginary frequencies from ts_dir or freq_file
    imag_freqs = {}
    if freq_file:
        imag_freqs = parse_freq_file(freq_file)
        n_with_freq = sum(1 for ts_id in common_all if ts_id in imag_freqs)
        print(f"Imaginary frequencies: {len(imag_freqs)} read from {freq_file}, {n_with_freq} in common set")
    elif ts_dir:
        imag_freqs = parse_imag_freqs(ts_dir)
        n_with_freq = sum(1 for ts_id in common_all if ts_id in imag_freqs)
        print(f"Imaginary frequencies: {len(imag_freqs)} parsed from {ts_dir}, {n_with_freq} in common set")

    # Compute structural change metrics from IRC SMILES
    struct_changes = {}
    for ts_id in common_all:
        smi = irc_data[ts_id]["smiles"]
        beg, end = smi.get("Beginning", ""), smi.get("End", "")
        if beg and end:
            sc = structural_change(beg, end)
            if sc is not None:
                struct_changes[ts_id] = sc
    print(f"Structural change metrics: {len(struct_changes)}/{len(common_all)} reactions")

    # Collect plot data and mismatch details
    plot_data = {}
    mismatch_details = {}  # {qrc_label: [(ts_id, status, irc_beg, irc_end, qrc_beg, qrc_end), ...]}

    for qrc_label, (qrc_file, qrc_data) in all_qrc.items():
        print(f"\n{'='*80}")
        print(f"{qrc_label} vs IRC ({len(common_all)} common reactions)")
        print(f"{'='*80}")

        common_ids = common_all

        n_same = 0
        n_flipped = 0
        n_mismatch = 0
        n_missing_smi = 0

        barrier_diffs = []
        rxn_diffs = []
        mismatch_ids = []
        mismatch_rows = []

        irc_bars = []
        qrc_bars = []
        irc_rxns = []
        qrc_rxns = []
        match_status = []
        ts_ids_list = []
        path_lengths = []
        n_beg_match = 0
        n_end_match = 0
        n_checked = 0

        print(f"\n{'ts_id':>8s} {'IRC_bar':>9s} {qrc_label+'_bar':>9s} {'d_bar':>8s}"
              f" {'IRC_rxn':>9s} {qrc_label+'_rxn':>9s} {'d_rxn':>8s}"
              f"  {'dir':>5s} {'SMILES_match'}")

        for ts_id in common_ids:
            irc_e = irc_data[ts_id]
            qrc_e = qrc_data[ts_id]

            irc_barrier, irc_rxn = compute_energetics(irc_e["energies"])

            # Check if SMILES are available
            irc_has_smi = bool(irc_e["smiles"].get("Beginning")) and bool(irc_e["smiles"].get("End"))
            qrc_has_smi = bool(qrc_e["smiles"].get("Beginning")) and bool(qrc_e["smiles"].get("End"))

            if not irc_has_smi or not qrc_has_smi:
                n_missing_smi += 1
                qrc_barrier, qrc_rxn = compute_energetics(qrc_e["energies"])
                d_bar = qrc_barrier - irc_barrier
                d_rxn = qrc_rxn - irc_rxn
                barrier_diffs.append(d_bar)
                rxn_diffs.append(d_rxn)
                irc_bars.append(irc_barrier); qrc_bars.append(qrc_barrier)
                irc_rxns.append(irc_rxn); qrc_rxns.append(qrc_rxn)
                match_status.append("missing")
                ts_ids_list.append(ts_id)
                path_lengths.append(irc_e.get("path_length", 0))
                print(f"{ts_id:>8s} {irc_barrier:9.2f} {qrc_barrier:9.2f} {d_bar:8.2f}"
                      f" {irc_rxn:9.2f} {qrc_rxn:9.2f} {d_rxn:8.2f}"
                      f"  {'?':>5s} no SMILES")
                continue

            aligned_e, aligned_smi, flipped = align_qrc_to_irc(irc_e, qrc_e)
            qrc_barrier, qrc_rxn = compute_energetics(aligned_e)

            beg_ok = smiles_match(irc_e["smiles"].get("Beginning", ""),
                                  aligned_smi.get("Beginning", ""))
            end_ok = smiles_match(irc_e["smiles"].get("End", ""),
                                  aligned_smi.get("End", ""))

            n_checked += 1
            if beg_ok:
                n_beg_match += 1
            if end_ok:
                n_end_match += 1

            d_bar = qrc_barrier - irc_barrier
            d_rxn = qrc_rxn - irc_rxn

            if beg_ok and end_ok:
                status = "both"
                if flipped:
                    n_flipped += 1
                else:
                    n_same += 1
            elif beg_ok or end_ok:
                status = "partial"
                n_mismatch += 1
                mismatch_ids.append(ts_id)
                mismatch_rows.append((ts_id, status, beg_ok, end_ok,
                    canonical(irc_e["smiles"].get("Beginning", "")),
                    canonical(irc_e["smiles"].get("End", "")),
                    canonical(aligned_smi.get("Beginning", "")),
                    canonical(aligned_smi.get("End", "")),
                    d_bar, d_rxn))
            else:
                status = "NONE"
                n_mismatch += 1
                mismatch_ids.append(ts_id)
                mismatch_rows.append((ts_id, status, beg_ok, end_ok,
                    canonical(irc_e["smiles"].get("Beginning", "")),
                    canonical(irc_e["smiles"].get("End", "")),
                    canonical(aligned_smi.get("Beginning", "")),
                    canonical(aligned_smi.get("End", "")),
                    d_bar, d_rxn))
            barrier_diffs.append(d_bar)
            rxn_diffs.append(d_rxn)

            irc_bars.append(irc_barrier); qrc_bars.append(qrc_barrier)
            irc_rxns.append(irc_rxn); qrc_rxns.append(qrc_rxn)
            match_status.append(status)
            ts_ids_list.append(ts_id)
            path_lengths.append(irc_e.get("path_length", 0))

            direction = "flip" if flipped else "same"
            print(f"{ts_id:>8s} {irc_barrier:9.2f} {qrc_barrier:9.2f} {d_bar:8.2f}"
                  f" {irc_rxn:9.2f} {qrc_rxn:9.2f} {d_rxn:8.2f}"
                  f"  {direction:>5s} {status}")

        # Summary statistics
        print(f"\n--- {qrc_label} Summary ---")
        print(f"Direction: {n_same} same, {n_flipped} flipped, {n_mismatch} mismatch, {n_missing_smi} missing SMILES")

        if barrier_diffs:
            import statistics
            abs_bar = [abs(d) for d in barrier_diffs]
            abs_rxn = [abs(d) for d in rxn_diffs]
            print(f"Barrier difference:  MAE = {statistics.mean(abs_bar):.2f}, "
                  f"Max = {max(abs_bar):.2f}, RMSE = {(sum(d**2 for d in barrier_diffs)/len(barrier_diffs))**0.5:.2f} kcal/mol")
            print(f"Rxn energy difference: MAE = {statistics.mean(abs_rxn):.2f}, "
                  f"Max = {max(abs_rxn):.2f}, RMSE = {(sum(d**2 for d in rxn_diffs)/len(rxn_diffs))**0.5:.2f} kcal/mol")

        if mismatch_ids:
            print(f"Mismatched reactions: {', '.join(mismatch_ids[:20])}")
            if len(mismatch_ids) > 20:
                print(f"  ... and {len(mismatch_ids) - 20} more")

        beg_pct = 100 * n_beg_match / n_checked if n_checked else 0
        end_pct = 100 * n_end_match / n_checked if n_checked else 0
        print(f"Reactant match: {n_beg_match}/{n_checked} ({beg_pct:.1f}%)")
        print(f"Product match:  {n_end_match}/{n_checked} ({end_pct:.1f}%)")

        mismatch_details[qrc_label] = mismatch_rows

        plot_data[qrc_label] = {
            "irc_bars": np.array(irc_bars), "qrc_bars": np.array(qrc_bars),
            "irc_rxns": np.array(irc_rxns), "qrc_rxns": np.array(qrc_rxns),
            "match_status": match_status, "ts_ids": ts_ids_list,
            "path_lengths": np.array(path_lengths),
            "bar_diffs": np.array(barrier_diffs), "rxn_diffs": np.array(rxn_diffs),
            "bar_mae": statistics.mean(abs_bar) if barrier_diffs else 0,
            "rxn_mae": statistics.mean(abs_rxn) if rxn_diffs else 0,
            "beg_pct": beg_pct, "end_pct": end_pct,
        }

    # --- Write SMILES mismatch file ---
    mismatch_file = "smiles_mismatches.txt"
    with open(mismatch_file, "w") as mf:
        for qrc_label, rows in mismatch_details.items():
            mf.write(f"{'='*100}\n")
            mf.write(f"{qrc_label} vs IRC: {len(rows)} SMILES mismatches\n")
            mf.write(f"{'='*100}\n\n")
            for ts_id, status, beg_ok, end_ok, irc_beg, irc_end, qrc_beg, qrc_end, d_bar, d_rxn in rows:
                mf.write(f"{ts_id}  (match: {status})\n")
                beg_flag = "  OK" if beg_ok else "  XX"
                end_flag = "  OK" if end_ok else "  XX"
                mf.write(f"  Reactant{beg_flag}:  IRC: {irc_beg}\n")
                mf.write(f"           {' '*4}  {qrc_label}: {qrc_beg}\n")
                mf.write(f"  Product {end_flag}:  IRC: {irc_end}\n")
                mf.write(f"           {' '*4}  {qrc_label}: {qrc_end}\n")
                mf.write(f"  Barrier error: {d_bar:+.2f} kcal/mol    Rxn energy error: {d_rxn:+.2f} kcal/mol\n")
                mf.write("\n")
    print(f"SMILES mismatches saved to {mismatch_file}")

    # --- Generate scatter plots ---
    qrc_labels = list(plot_data.keys())
    n_qrc = len(qrc_labels)
    if n_qrc == 0:
        return

    fig, axes = plt.subplots(2, n_qrc, figsize=(5 * n_qrc, 9), squeeze=False)

    for col, qlabel in enumerate(qrc_labels):
        d = plot_data[qlabel]
        status = d["match_status"]
        matched = np.array([s == "both" for s in status])
        mismatched = ~matched

        for row, (irc_vals, qrc_vals, ylabel, mae) in enumerate([
            (d["irc_bars"], d["qrc_bars"], "Barrier (kcal/mol)", d["bar_mae"]),
            (d["irc_rxns"], d["qrc_rxns"], "Reaction energy (kcal/mol)", d["rxn_mae"]),
        ]):
            ax = axes[row, col]

            # Plot matched points
            if matched.any():
                ax.scatter(irc_vals[matched], qrc_vals[matched],
                           s=12, alpha=0.5, c='#2563eb', edgecolors='none', label='match')
            # Plot mismatched points
            if mismatched.any():
                ax.scatter(irc_vals[mismatched], qrc_vals[mismatched],
                           s=12, alpha=0.5, c='#dc2626', edgecolors='none', label='mismatch')

            # y = x reference line
            all_vals = np.concatenate([irc_vals, qrc_vals])
            lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    'k--', lw=0.8, alpha=0.5)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect('equal')

            ax.set_xlabel(f"IRC {ylabel}")
            ax.set_ylabel(f"{qlabel} {ylabel}")
            if row == 0:
                ax.set_title(qlabel)
            ann = f"MAE = {mae:.2f} kcal/mol\nReactant match: {d['beg_pct']:.1f}%\nProduct match: {d['end_pct']:.1f}%"
            ax.text(0.05, 0.95, ann, transform=ax.transAxes,
                    va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', edgecolor='gray', alpha=0.8))
            if mismatched.any():
                ax.legend(fontsize=8, loc='lower right', framealpha=0.8)

    fig.tight_layout()
    outfile = "irc_vs_qrc.png"
    fig.savefig(outfile, dpi=200)
    print(f"\nScatter plot saved to {outfile}")
    plt.close(fig)

    # --- Imaginary frequency correlation plot ---
    if imag_freqs:
        fig2, axes2 = plt.subplots(2, n_qrc, figsize=(5 * n_qrc, 9), squeeze=False)

        for col, qlabel in enumerate(qrc_labels):
            d = plot_data[qlabel]
            ts_ids = d["ts_ids"]
            status = d["match_status"]
            bar_diffs = d["bar_diffs"]
            rxn_diffs_arr = d["rxn_diffs"]

            # Build arrays: |imag_freq|, |barrier_error|, |rxn_error|, match/mismatch
            abs_freq = []
            abs_bar_err = []
            abs_rxn_err = []
            is_matched = []
            for i, ts_id in enumerate(ts_ids):
                if ts_id in imag_freqs:
                    abs_freq.append(abs(imag_freqs[ts_id]))
                    abs_bar_err.append(abs(bar_diffs[i]))
                    abs_rxn_err.append(abs(rxn_diffs_arr[i]))
                    is_matched.append(status[i] == "both")

            abs_freq = np.array(abs_freq)
            abs_bar_err = np.array(abs_bar_err)
            abs_rxn_err = np.array(abs_rxn_err)
            is_matched = np.array(is_matched)

            for row, (y_vals, ylabel) in enumerate([
                (abs_bar_err, "|Barrier error| (kcal/mol)"),
                (abs_rxn_err, "|Rxn energy error| (kcal/mol)"),
            ]):
                ax = axes2[row, col]
                matched = is_matched
                mismatched = ~is_matched

                if matched.any():
                    ax.scatter(abs_freq[matched], y_vals[matched],
                               s=12, alpha=0.5, c='#2563eb', edgecolors='none', label='match')
                if mismatched.any():
                    ax.scatter(abs_freq[mismatched], y_vals[mismatched],
                               s=12, alpha=0.5, c='#dc2626', edgecolors='none', label='mismatch')

                ax.set_xlabel("|Imaginary frequency| (cm$^{-1}$)")
                ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(qlabel)

                # Compute Pearson correlation
                if len(abs_freq) > 2:
                    r = np.corrcoef(abs_freq, y_vals)[0, 1]
                    ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(abs_freq)}",
                            transform=ax.transAxes, va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', edgecolor='gray', alpha=0.8))

                if mismatched.any():
                    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

        fig2.tight_layout()
        freq_outfile = "imag_freq_vs_error.png"
        fig2.savefig(freq_outfile, dpi=200)
        print(f"Imaginary frequency correlation plot saved to {freq_outfile}")
        plt.close(fig2)

    # --- IRC path length correlation plot ---
    fig3, axes3 = plt.subplots(2, n_qrc, figsize=(5 * n_qrc, 9), squeeze=False)

    for col, qlabel in enumerate(qrc_labels):
        d = plot_data[qlabel]
        status = d["match_status"]
        plen = d["path_lengths"]
        bar_diffs_arr = d["bar_diffs"]
        rxn_diffs_arr = d["rxn_diffs"]

        matched = np.array([s == "both" for s in status])
        mismatched = ~matched

        for row, (y_vals, ylabel) in enumerate([
            (np.abs(bar_diffs_arr), "|Barrier error| (kcal/mol)"),
            (np.abs(rxn_diffs_arr), "|Rxn energy error| (kcal/mol)"),
        ]):
            ax = axes3[row, col]

            if matched.any():
                ax.scatter(plen[matched], y_vals[matched],
                           s=12, alpha=0.5, c='#2563eb', edgecolors='none', label='match')
            if mismatched.any():
                ax.scatter(plen[mismatched], y_vals[mismatched],
                           s=12, alpha=0.5, c='#dc2626', edgecolors='none', label='mismatch')

            ax.set_xlabel("IRC path length (amu$^{1/2}$ bohr)")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(qlabel)

            if len(plen) > 2:
                r = np.corrcoef(plen, y_vals)[0, 1]
                ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(plen)}",
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', edgecolor='gray', alpha=0.8))

            if mismatched.any():
                ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

    fig3.tight_layout()
    path_outfile = "path_length_vs_error.png"
    fig3.savefig(path_outfile, dpi=200)
    print(f"IRC path length correlation plot saved to {path_outfile}")
    plt.close(fig3)

    # --- Structural change correlation plots ---
    if struct_changes:
        # Tanimoto distance plot
        fig4, axes4 = plt.subplots(2, n_qrc, figsize=(5 * n_qrc, 9), squeeze=False)

        for col, qlabel in enumerate(qrc_labels):
            d = plot_data[qlabel]
            ts_ids = d["ts_ids"]
            status = d["match_status"]
            bar_diffs_arr = d["bar_diffs"]
            rxn_diffs_arr = d["rxn_diffs"]

            # Build arrays for reactions with structural change data
            tan_dist = []
            abs_bar_err = []
            abs_rxn_err = []
            is_matched = []
            for i, ts_id in enumerate(ts_ids):
                if ts_id in struct_changes:
                    tan_dist.append(struct_changes[ts_id]["tanimoto_dist"])
                    abs_bar_err.append(abs(bar_diffs_arr[i]))
                    abs_rxn_err.append(abs(rxn_diffs_arr[i]))
                    is_matched.append(status[i] == "both")

            tan_dist = np.array(tan_dist)
            abs_bar_err = np.array(abs_bar_err)
            abs_rxn_err = np.array(abs_rxn_err)
            is_matched = np.array(is_matched)

            for row, (y_vals, ylabel) in enumerate([
                (abs_bar_err, "|Barrier error| (kcal/mol)"),
                (abs_rxn_err, "|Rxn energy error| (kcal/mol)"),
            ]):
                ax = axes4[row, col]
                matched = is_matched
                mismatched = ~is_matched

                if matched.any():
                    ax.scatter(tan_dist[matched], y_vals[matched],
                               s=12, alpha=0.5, c='#2563eb', edgecolors='none', label='match')
                if mismatched.any():
                    ax.scatter(tan_dist[mismatched], y_vals[mismatched],
                               s=12, alpha=0.5, c='#dc2626', edgecolors='none', label='mismatch')

                ax.set_xlabel("Tanimoto distance (reactant vs product)")
                ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(qlabel)

                if len(tan_dist) > 2:
                    r = np.corrcoef(tan_dist, y_vals)[0, 1]
                    ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(tan_dist)}",
                            transform=ax.transAxes, va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', edgecolor='gray', alpha=0.8))

                if mismatched.any():
                    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

        fig4.tight_layout()
        struct_outfile = "struct_change_vs_error.png"
        fig4.savefig(struct_outfile, dpi=200)
        print(f"Structural change correlation plot saved to {struct_outfile}")
        plt.close(fig4)

        # Bond count change plot
        fig5, axes5 = plt.subplots(2, n_qrc, figsize=(5 * n_qrc, 9), squeeze=False)

        for col, qlabel in enumerate(qrc_labels):
            d = plot_data[qlabel]
            ts_ids = d["ts_ids"]
            status = d["match_status"]
            bar_diffs_arr = d["bar_diffs"]
            rxn_diffs_arr = d["rxn_diffs"]

            delta_bonds = []
            abs_bar_err = []
            abs_rxn_err = []
            is_matched = []
            for i, ts_id in enumerate(ts_ids):
                if ts_id in struct_changes:
                    sc = struct_changes[ts_id]
                    delta_bonds.append(sc["delta_bonds"])
                    abs_bar_err.append(abs(bar_diffs_arr[i]))
                    abs_rxn_err.append(abs(rxn_diffs_arr[i]))
                    is_matched.append(status[i] == "both")

            delta_bonds = np.array(delta_bonds)
            abs_bar_err = np.array(abs_bar_err)
            abs_rxn_err = np.array(abs_rxn_err)
            is_matched = np.array(is_matched)

            for row, (y_vals, ylabel) in enumerate([
                (abs_bar_err, "|Barrier error| (kcal/mol)"),
                (abs_rxn_err, "|Rxn energy error| (kcal/mol)"),
            ]):
                ax = axes5[row, col]
                matched = is_matched
                mismatched = ~is_matched

                # Add jitter to integer x values for visibility
                jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(delta_bonds))
                x_jittered = delta_bonds + jitter

                if matched.any():
                    ax.scatter(x_jittered[matched], y_vals[matched],
                               s=12, alpha=0.5, c='#2563eb', edgecolors='none', label='match')
                if mismatched.any():
                    ax.scatter(x_jittered[mismatched], y_vals[mismatched],
                               s=12, alpha=0.5, c='#dc2626', edgecolors='none', label='mismatch')

                ax.set_xlabel(r"$\Delta$ bond count (|bonds$_{product}$ - bonds$_{reactant}$|)")
                ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(qlabel)

                if len(delta_bonds) > 2:
                    r = np.corrcoef(delta_bonds, y_vals)[0, 1]
                    ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {len(delta_bonds)}",
                            transform=ax.transAxes, va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', edgecolor='gray', alpha=0.8))

                if mismatched.any():
                    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

        fig5.tight_layout()
        bonds_outfile = "delta_bonds_vs_error.png"
        fig5.savefig(bonds_outfile, dpi=200)
        print(f"Bond count change correlation plot saved to {bonds_outfile}")
        plt.close(fig5)


if __name__ == "__main__":
    main()
