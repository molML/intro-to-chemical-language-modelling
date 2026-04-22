"""
evaluation — shared utilities for the CLM Course UvA evaluation framework.

This module exposes the building blocks imported by all other modules:

    load_smiles          Load SMILES from a file (.smi / .csv) or return a list as-is.
    to_mol               Convert a single SMILES string or RDKit Mol to a validated Mol.
    to_molecules         Convert a list of SMILES or Mol objects to validated Mol objects.
    compute_fingerprints Compute Morgan (ECFP4) fingerprints for a list of molecules.
    compute_scaffolds    Compute Bemis–Murcko scaffold SMILES for a list of molecules.

All higher-level functions (metrics, splitting, properties, visualization) import
from here so that conversion and fingerprint settings are consistent throughout.
"""

import warnings

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_smiles(source):
    """Load SMILES strings from a file path or return a list as-is.

    Supported input formats:

    *   **List of strings** — returned unchanged (passthrough).
    *   **.smi / .txt** — plain text, one SMILES per line.  The first
        whitespace-delimited token on each line is taken as the SMILES; any
        remaining tokens (e.g. a molecule name or ID) are ignored.  Lines
        starting with ``#`` are treated as comments and skipped.
    *   **.csv** — tabular file with a column named ``smiles`` (case-insensitive).
        Any additional columns (name, activity, etc.) are ignored.

    Parameters
    ----------
    source : str (file path) or list of str

    Returns
    -------
    list of str
        Raw SMILES strings.  No validity checking is performed here; use
        ``to_molecules`` or ``validity`` for that.

    Example
    -------
    >>> smiles = load_smiles("my_molecules.smi")
    >>> smiles = load_smiles("dataset.csv")          # must have a 'smiles' column
    >>> smiles = load_smiles(["CCO", "c1ccccc1"])    # passthrough
    """
    if isinstance(source, list):
        return source

    path = str(source)

    if path.lower().endswith(".csv"):
        import pandas as pd
        dataframe = pd.read_csv(path)
        smiles_column = next(
            (col for col in dataframe.columns if col.lower() == "smiles"), None
        )
        if smiles_column is None:
            raise ValueError("CSV file has no column named 'smiles'.")
        return dataframe[smiles_column].dropna().astype(str).tolist()

    # .smi, .txt, or any other line-based text format
    with open(path) as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    return [line.split()[0] for line in lines]


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------

def to_mol(input_molecule):
    """Convert a single SMILES string or RDKit Mol object to a validated Mol.

    This is the elementwise conversion function.  ``to_molecules`` calls this
    for every item in a list and handles the invalid-SMILES warning.

    Parameters
    ----------
    input_molecule : str or rdkit.Chem.Mol

    Returns
    -------
    rdkit.Chem.Mol or None
        None if the SMILES string could not be parsed.

    Example
    -------
    >>> to_mol("CCO")
    <rdkit.Chem.rdchem.Mol object at ...>
    >>> to_mol("not_a_smiles") is None
    True
    """
    if isinstance(input_molecule, str):
        return Chem.MolFromSmiles(input_molecule)
    return input_molecule


def to_molecules(inputs):
    """Convert a list of SMILES strings or RDKit Mol objects to validated Mol objects.

    Calls ``to_mol`` for each item.  Invalid SMILES are dropped and a warning
    is emitted.

    Parameters
    ----------
    inputs : list of str or list of rdkit.Chem.Mol

    Returns
    -------
    list of rdkit.Chem.Mol
        Only valid molecules are returned.

    Example
    -------
    >>> molecules = to_molecules(["CCO", "not_a_smiles", "c1ccccc1"])
    >>> len(molecules)
    2
    """
    if not inputs:
        return []
    molecules = [to_mol(x) for x in inputs]
    n_invalid = sum(1 for m in molecules if m is None)
    if n_invalid:
        warnings.warn(
            f"{n_invalid} / {len(inputs)} inputs could not be parsed and were dropped."
        )
    return [m for m in molecules if m is not None]


def compute_fingerprints(molecules, radius=2, n_bits=2048):
    """Compute Morgan (circular) fingerprints for a list of molecules.

    Radius 2 corresponds to ECFP4 fingerprints. RDKit's C++ backend is used
    directly, so no explicit multiprocessing is needed here.

    Parameters
    ----------
    molecules : list of rdkit.Chem.Mol
    radius : int, default 2
        Morgan radius (2 = ECFP4, 3 = ECFP6).
    n_bits : int, default 2048
        Fingerprint bit-vector length.

    Returns
    -------
    list of rdkit.DataStructs.ExplicitBitVect

    Example
    -------
    >>> from rdkit import Chem
    >>> molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    >>> fingerprints = compute_fingerprints(molecules)
    >>> len(fingerprints)
    2
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [generator.GetFingerprint(mol) for mol in molecules]


def compute_scaffolds(molecules):
    """Compute Bemis–Murcko scaffold SMILES for a list of molecules.

    Parameters
    ----------
    molecules : list of rdkit.Chem.Mol

    Returns
    -------
    list of str
        Scaffold SMILES, one per input molecule.

    Example
    -------
    >>> from rdkit import Chem
    >>> molecules = [Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")]  # Aspirin
    >>> compute_scaffolds(molecules)
    ['c1ccccc1']
    """
    scaffolds = []
    for mol in molecules:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaffold)
    return scaffolds
