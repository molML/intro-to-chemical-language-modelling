"""
evaluation.splitting — dataset splitting for molecular sets.

Framework
---------
All splitting strategies share a common core: `split_by_values` accepts a list
of items (full SMILES or scaffold SMILES, or anything else) and a parallel list
of numeric values, sorts by those values, and partitions into train / validation
/ test.  The choice of *items* and *values* determines the split strategy:

    Items           Values          → Strategy
    ------------    -----------     ------------------------------------------
    Full SMILES     Random number   Random molecular split
    Full SMILES     Any property    Property-based split (e.g., by MW)
    Full SMILES     Scaffold NND    Hardest split: distant scaffolds → test
    Scaffold SMILES Random number   Random scaffold split (group-level random)

The convenience wrappers `random_split` and `scaffold_split` implement the two
most common cases, both calling `split_by_values` internally.

Functions
---------
split_by_values     Generic core: sort by value, partition into three sets.
random_split        Random assignment at the molecule level.
scaffold_split      Random assignment at the scaffold group level.
"""

import random

from rdkit import Chem

from . import to_molecules, compute_scaffolds


def split_by_values(molecules, values, ratio=(0.8, 0.1, 0.1), high_values_in_test=True):
    """Split a list into train / validation / test by sorting on a numeric value.

    Items are sorted by `values` and partitioned into three contiguous segments
    according to `ratio`.  The `high_values_in_test` flag controls which end of
    the sorted list becomes the test set.

    This is the building block for all other splitting functions.  It is
    intentionally generic: `molecules` can be full SMILES, scaffold SMILES, or
    any other list of items — as long as `values` is a parallel list of numbers.

    Parameters
    ----------
    molecules : list
        Items to split (typically SMILES strings, but can be anything).
    values : list of float
        One numeric value per item.  Must be the same length as `molecules`.
    ratio : tuple of three floats, default (0.8, 0.1, 0.1)
        (train fraction, validation fraction, test fraction).  Must sum to 1.
    high_values_in_test : bool, default True
        If True, items with the *highest* values end up in the test set.
        If False, items with the *lowest* values end up in the test set.

    Returns
    -------
    train, validation, test : three lists

    Example
    -------
    >>> smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCC", "CCCC"]
    >>> values = [1, 5, 3, 2, 4]
    >>> # high_values_in_test=True → highest value (5, "c1ccccc1") → test
    >>> train, val, test = split_by_values(smiles, values, ratio=(0.6, 0.2, 0.2))
    """
    if len(molecules) != len(values):
        raise ValueError(
            f"molecules and values must have the same length: "
            f"got {len(molecules)} and {len(values)}."
        )
    if abs(sum(ratio) - 1.0) > 1e-9:
        raise ValueError(f"ratio must sum to 1.0: got {sum(ratio):.4f}.")

    n = len(molecules)
    if high_values_in_test:
        # sort ascending: lowest values → train end, highest values → test end
        sorted_molecules = [mol for _, mol in sorted(zip(values, molecules))]
    else:
        # sort descending: highest values → train end, lowest values → test end
        sorted_molecules = [mol for _, mol in sorted(zip(values, molecules), reverse=True)]

    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    train = sorted_molecules[:n_train]
    validation = sorted_molecules[n_train : n_train + n_val]
    test = sorted_molecules[n_train + n_val :]
    return train, validation, test


def random_split(molecules, ratio=(0.8, 0.1, 0.1), seed=42):
    """Randomly split molecules at the **molecule** level.

    Each molecule is assigned an independent random number; molecules are then
    sorted by that number and partitioned.  Train and test sets will have very
    similar property distributions.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
    ratio : tuple of three floats, default (0.8, 0.1, 0.1)
    seed : int, default 42

    Returns
    -------
    train, validation, test : three lists of SMILES strings

    Example
    -------
    >>> train, val, test = random_split(smiles_list, ratio=(0.8, 0.1, 0.1))
    """
    valid_molecules = to_molecules(molecules)
    smiles_list = [Chem.MolToSmiles(m) for m in valid_molecules]

    rng = random.Random(seed)
    random_values = [rng.random() for _ in smiles_list]
    return split_by_values(smiles_list, random_values, ratio)


def scaffold_split(molecules, ratio=(0.8, 0.1, 0.1), seed=42):
    """Randomly split molecules at the **scaffold** level.

    Molecules are grouped by their Bemis–Murcko scaffold.  Each *scaffold group*
    is assigned a single random number; all molecules in that group inherit the
    same value and therefore sort together.  This ensures that molecules sharing
    a scaffold land in the same split, preventing data leakage across partitions.

    This differs from `random_split` only in the level of randomisation:
    molecules → scaffold groups.  The underlying `split_by_values` call is
    identical.

    For a harder evaluation split, consider using `split_by_values` with
    per-molecule nearest-neighbor distances computed on scaffold fingerprints
    as the values (isolated scaffolds → test set).

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
    ratio : tuple of three floats, default (0.8, 0.1, 0.1)
    seed : int, default 42

    Returns
    -------
    train, validation, test : three lists of SMILES strings

    Example
    -------
    >>> train, val, test = scaffold_split(smiles_list, ratio=(0.8, 0.1, 0.1))
    """
    valid_molecules = to_molecules(molecules)
    smiles_list = [Chem.MolToSmiles(m) for m in valid_molecules]
    scaffolds = compute_scaffolds(valid_molecules)

    # Assign one random value per unique scaffold;
    # all molecules with the same scaffold get the same value → sort together.
    rng = random.Random(seed)
    scaffold_value = {scaffold: rng.random() for scaffold in set(scaffolds)}
    values = [scaffold_value[s] for s in scaffolds]

    return split_by_values(smiles_list, values, ratio)
