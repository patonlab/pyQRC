#!/usr/bin/env python3
"""
Convert XYZ file with multiple structures to Gaussian input files for TS optimization
"""

import sys
import os

def read_xyz_structures(xyz_file):
    """
    Read multiple structures from an XYZ file.
    Returns a list of structures, where each structure is a dict with:
    - 'natoms': number of atoms
    - 'comment': comment line
    - 'coords': list of (element, x, y, z) tuples
    """
    structures = []
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
        
        # Read number of atoms
        try:
            natoms = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        
        # Read comment line
        if i + 1 >= len(lines):
            break
        comment = lines[i + 1].strip()
        
        # Read coordinates
        coords = []
        for j in range(i + 2, i + 2 + natoms):
            if j >= len(lines):
                break
            parts = lines[j].split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                coords.append((element, x, y, z))
        
        if len(coords) == natoms:
            structures.append({
                'natoms': natoms,
                'comment': comment,
                'coords': coords
            })
        
        i += 2 + natoms
    
    return structures

def write_gaussian_input(structure, filename, charge=0, multiplicity=1):
    """
    Write a Gaussian input file for TS optimization.
    
    Parameters:
    - structure: dictionary containing structure information
    - filename: output filename
    - charge: molecular charge (default: 0)
    - multiplicity: spin multiplicity (default: 1)
    """
    
    with open(filename, 'w') as f:
        # Write header with resource allocation
        f.write("%nprocshared=16\n")
        f.write("%mem=72GB\n")
        #f.write("%chk={}.chk\n".format(os.path.splitext(filename)[0]))
        f.write("\n")
        
        # Route section for TS optimization
        f.write("# opt=(ts,calcfc,noeigentest) freq wB97XD/6-31G*\n")
        f.write("\n")
        
        # Title section
        title = structure['comment'] if structure['comment'] else "TS optimization"
        f.write("{}\n".format(title))
        f.write("\n")
        
        # Charge and multiplicity
        f.write("{} {}\n".format(charge, multiplicity))
        
        # Coordinates
        for element, x, y, z in structure['coords']:
            f.write("{:2s}  {:12.6f}  {:12.6f}  {:12.6f}\n".format(element, x, y, z))
        
        # Blank line to terminate input
        f.write("\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python xyz_to_gaussian.py <xyz_file> [charge] [multiplicity] [max_structures]")
        print("\nExamples:")
        print("  python xyz_to_gaussian.py structures.xyz")
        print("  python xyz_to_gaussian.py structures.xyz 0 1")
        print("  python xyz_to_gaussian.py structures.xyz 0 1 500")
        print("\nArguments:")
        print("  xyz_file       : Input XYZ file with one or more structures")
        print("  charge         : Molecular charge (default: 0)")
        print("  multiplicity   : Spin multiplicity (default: 1)")
        print("  max_structures : Maximum number of structures to convert (default: all)")
        sys.exit(1)
    
    xyz_file = sys.argv[1]
    charge = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    multiplicity = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_structures = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    if not os.path.exists(xyz_file):
        print(f"Error: File '{xyz_file}' not found!")
        sys.exit(1)
    
    # Read structures from XYZ file
    print(f"Reading structures from {xyz_file}...")
    structures = read_xyz_structures(xyz_file)
    
    if not structures:
        print("No structures found in the XYZ file!")
        sys.exit(1)
    
    total_structures = len(structures)
    print(f"Found {total_structures} structure(s)")
    
    # Apply max_structures limit if specified
    if max_structures is not None:
        if max_structures < total_structures:
            structures = structures[:max_structures]
            print(f"Limiting conversion to first {max_structures} structure(s)")
        elif max_structures > total_structures:
            print(f"Note: Requested {max_structures} structures but only {total_structures} available")
    
    num_to_convert = len(structures)
    
    # Create output directory
    output_dir = "gaussian_inputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each structure to Gaussian input
    base_name = os.path.splitext(os.path.basename(xyz_file))[0]
    
    for i, structure in enumerate(structures, 1):
        if num_to_convert == 1:
            output_file = os.path.join(output_dir, f"{base_name}.com")
        else:
            output_file = os.path.join(output_dir, f"{base_name}_{i:03d}.com")
        
        write_gaussian_input(structure, output_file, charge, multiplicity)
        print(f"  Created: {output_file} ({structure['natoms']} atoms)")
    
    print(f"\nConverted {num_to_convert} structure(s) to Gaussian input files in '{output_dir}/' directory")
    print(f"Settings: wB97XD/6-31G*, 16 processors, 72GB memory")

if __name__ == "__main__":
    main()
