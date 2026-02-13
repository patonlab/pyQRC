# Analysis

## Contents

- `compare_irc_qrc.py` - Python script to compare IRC and QRC results
- `compare_irc_qrc.txt` - Full text output of the comparison
- `smiles_mismatches.txt` - Detailed listing of reactions where IRC and QRC give different SMILES (i.e., different products or reactants)
- `irc_vs_qrc.png` - Scatter plots comparing IRC vs QRC barriers and reaction energies
- `imag_freq_vs_error.png` - Correlation between imaginary frequency magnitude and QRC error
- `path_length_vs_error.png` - Correlation between IRC path length and QRC error
- `struct_change_vs_error.png` - Correlation between Tanimoto distance (reactant vs product) and QRC error
- `delta_bonds_vs_error.png` - Correlation between change in bond count and QRC error

## What the Script Does

`compare_irc_qrc.py` compares IRC (intrinsic reaction coordinate) and QRC (quick reaction coordinate) results for the same set of transition states. It:

1. Parses IRC and QRC data files to extract energies and SMILES for reactants, transition states, and products
2. Aligns the QRC direction to match the IRC (detecting whether forward/reverse are flipped)
3. Compares SMILES identity at each endpoint using RDKit canonical SMILES
4. Computes barrier height and reaction energy differences (in kcal/mol)
5. Reports summary statistics (MAE, RMSE, max error) and match rates
6. Generates scatter plots and correlation plots against structural descriptors

## Usage

```
python compare_irc_qrc.py --freq-file ../ts/imag_freqs.txt ../irc/IRC_data.txt ../qrc_0.1/QRC01_data.txt ../qrc_0.3/QRC03_data.txt ../qrc_0.5/QRC05_data.txt
```
