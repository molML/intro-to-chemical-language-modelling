"""
evaluation.metrics — chemical validity and distribution metrics.

Functions
---------
validity                    Fraction of inputs that are valid molecules.
uniqueness                  Fraction of unique canonical SMILES.
novelty                     Fraction not present in a reference set.
mean_pairwise_distance      Mean pairwise Tanimoto distance within a set.
scaffold_entropy            Shannon entropy over scaffold frequencies.
nearest_neighbor_distance   Mean nearest-neighbor Tanimoto distance.
fraction_passing_lipinski   Fraction satisfying Lipinski's Rule of Five.

All functions accept either a list of SMILES strings or a list of RDKit Mol
objects.  Functions that are O(n²) accept a `sample_size` argument (default
5 000) and emit a warning when subsampling occurs.
"""

import random
import warnings
from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs

from . import to_molecules, compute_fingerprints, compute_scaffolds


# ---------------------------------------------------------------------------
# Set quality metrics
# ---------------------------------------------------------------------------

def validity(molecules):
    """Fraction of inputs that parse to valid RDKit molecules.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol

    Returns
    -------
    float in [0, 1]

    Example
    -------
    >>> validity(["CCO", "not_valid", "c1ccccc1"])
    0.6666...
    """
    if not molecules:
        return 0.0
    if isinstance(molecules[0], str):
        n_valid = sum(1 for s in molecules if Chem.MolFromSmiles(s) is not None)
        return n_valid / len(molecules)
    n_valid = sum(1 for m in molecules if m is not None)
    return n_valid / len(molecules)


def uniqueness(molecules):
    """Fraction of unique canonical SMILES in the set.

    Measures how often the same molecule appears more than once.  A value of
    1.0 means every molecule is distinct; lower values indicate repeated
    structures.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol

    Returns
    -------
    float in [0, 1]

    Example
    -------
    >>> uniqueness(["CCO", "CCO", "c1ccccc1"])
    0.6666...
    """
    valid_molecules = to_molecules(molecules)
    if not valid_molecules:
        return 0.0
    canonical_smiles = [Chem.MolToSmiles(m) for m in valid_molecules]
    if len(canonical_smiles) < 1:
        return 0.0
    return len(set(canonical_smiles)) / len(canonical_smiles)


def novelty(molecules, reference):
    """Fraction of molecules not present in a reference set.

    Useful for measuring how many generated molecules are truly new relative
    to a training or database set.  Comparison is based on canonical SMILES,
    so stereochemistry and atom ordering are normalised before matching.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
        The set to evaluate (e.g., generated molecules).
    reference : list of str or list of rdkit.Chem.Mol
        The reference set (e.g., training / fine-tuning data).

    Returns
    -------
    float in [0, 1]

    Example
    -------
    >>> novelty(["CCO", "c1ccccc1", "CC"], reference=["CCO"])
    0.6666...
    """
    query_molecules = to_molecules(molecules)
    reference_molecules = to_molecules(reference)
    reference_smiles = {Chem.MolToSmiles(m) for m in reference_molecules}
    query_smiles = [Chem.MolToSmiles(m) for m in query_molecules]
    if not query_smiles:
        return 0.0
    return sum(1 for s in query_smiles if s not in reference_smiles) / len(query_smiles)


# ---------------------------------------------------------------------------
# Structural diversity metrics (distance-based)
# ---------------------------------------------------------------------------

def mean_pairwise_distance(molecules, sample_size=5000, radius=2, n_bits=2048):
    """Mean pairwise Tanimoto distance (1 − similarity) within a molecular set.

    Computes the average structural distance over all unique pairs in the set.
    Higher values indicate a more chemically diverse set; values near 0 indicate
    that all molecules are nearly identical.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
    sample_size : int, default 5000
        If the set is larger, a random subsample of this size is used and a
        warning is emitted.
    radius : int, default 2
    n_bits : int, default 2048

    Returns
    -------
    float in [0, 1]

    Example
    -------
    >>> mean_pairwise_distance(["CCO", "c1ccccc1", "CC(=O)O"])
    0.85...
    """
    valid_molecules = to_molecules(molecules)
    if len(valid_molecules) > sample_size:
        warnings.warn(
            f"Sampled {sample_size} / {len(valid_molecules)} molecules "
            f"({100 * sample_size / len(valid_molecules):.1f}%) for mean_pairwise_distance."
        )
        valid_molecules = random.sample(valid_molecules, sample_size)

    fingerprints = compute_fingerprints(valid_molecules, radius, n_bits)
    pairwise_distances = []
    for i, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[:i])
        pairwise_distances.extend(1.0 - s for s in similarities)
    return float(np.mean(pairwise_distances)) if pairwise_distances else 0.0


def scaffold_entropy(molecules):
    """Shannon entropy over the Bemis–Murcko scaffold frequency distribution.

    Higher values indicate greater scaffold diversity. A value of 0 means all
    molecules share the same scaffold.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol

    Returns
    -------
    float >= 0  (in nats)

    Example
    -------
    >>> scaffold_entropy(["c1ccccc1", "c1ccncc1", "C1CCCCC1"])
    1.09...
    """
    valid_molecules = to_molecules(molecules)
    scaffolds = compute_scaffolds(valid_molecules)
    counts = np.array(list(Counter(scaffolds).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def nearest_neighbor_distance(molecules, reference=None, sample_size=5000, radius=2, n_bits=2048):
    """Mean nearest-neighbor Tanimoto distance from each molecule to a reference set.

    For each query molecule, finds the structurally most similar molecule in
    the reference set and records the Tanimoto *distance* (1 − similarity) to it.
    Returns the mean over all query molecules.

    When `reference` is None, computes the *internal* nearest-neighbor distance:
    each molecule is compared to all other molecules in the same set (self is
    excluded).  This single function therefore covers both use cases:

        nearest_neighbor_distance(generated)              → internal distance
        nearest_neighbor_distance(generated, training)    → distance to training set

    Higher values indicate that query molecules are structurally far from the
    reference.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
        The query set.
    reference : list of str or list of rdkit.Chem.Mol, optional
        The reference set.  If None, uses `molecules` itself (internal mode).
    sample_size : int, default 5000
        Maximum number of query molecules to use.
    radius : int, default 2
    n_bits : int, default 2048

    Returns
    -------
    float in [0, 1]

    Example
    -------
    >>> nearest_neighbor_distance(["CCO", "c1ccccc1"], reference=["CC(=O)O"])
    0.72...
    """
    internal = reference is None
    query_molecules = to_molecules(molecules)
    reference_molecules = to_molecules(reference) if not internal else query_molecules

    if not query_molecules or not reference_molecules:
        return 0.0

    # Internal mode requires at least 2 molecules (each is compared to all others)
    if internal and len(query_molecules) < 2:
        return 0.0

    if len(query_molecules) > sample_size:
        warnings.warn(
            f"Sampled {sample_size} / {len(query_molecules)} molecules "
            f"({100 * sample_size / len(query_molecules):.1f}%) for nearest_neighbor_distance."
        )
        query_molecules = random.sample(query_molecules, sample_size)

    query_fingerprints = compute_fingerprints(query_molecules, radius, n_bits)
    reference_fingerprints = compute_fingerprints(reference_molecules, radius, n_bits)

    nearest_neighbor_distances = []
    for i, fingerprint in enumerate(query_fingerprints):
        similarities = list(DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints))
        if internal:
            similarities[i] = -1.0  # exclude self-similarity; safe because len >= 2
        nearest_neighbor_distances.append(1.0 - max(similarities))
    return float(np.mean(nearest_neighbor_distances))


# ---------------------------------------------------------------------------
# Drug-likeness
# ---------------------------------------------------------------------------

def fraction_passing_lipinski(molecules):
    """Fraction of molecules satisfying Lipinski's Rule of Five (Ro5).

    Ro5 criteria:
        Molecular weight ≤ 500 Da
        LogP ≤ 5
        Hydrogen-bond donors ≤ 5
        Hydrogen-bond acceptors ≤ 10

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol

    Returns
    -------
    float in [0, 1]

    Example
    -------
    >>> fraction_passing_lipinski(["CCO", "c1ccccc1"])
    1.0
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors

    def _passes_lipinski(mol):
        return (
            Descriptors.MolWt(mol) <= 500
            and Descriptors.MolLogP(mol) <= 5
            and rdMolDescriptors.CalcNumHBD(mol) <= 5
            and rdMolDescriptors.CalcNumHBA(mol) <= 10
        )

    valid_molecules = to_molecules(molecules)
    if not valid_molecules:
        return 0.0
    if not valid_molecules:
        return 0.0
    return sum(1 for m in valid_molecules if _passes_lipinski(m)) / len(valid_molecules)
