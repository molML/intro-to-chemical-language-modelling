"""
evaluation.properties — molecular property wrappers using RDKit.

Each single-molecule function accepts either a SMILES string or an RDKit Mol
object and returns a scalar value.  The batch function `compute_properties`
accepts a list and returns a pandas DataFrame, one row per molecule.

Single-molecule functions
-------------------------
molecular_weight            Molecular weight in Da.
logp                        Crippen estimated LogP.
topological_polar_surface_area  TPSA in Å².
hydrogen_bond_donors        Number of hydrogen-bond donors.
hydrogen_bond_acceptors     Number of hydrogen-bond acceptors.
rotatable_bonds             Number of rotatable bonds.
ring_count                  Total number of rings.
quantitative_estimate_of_drug_likeness  QED score in [0, 1].

Batch function
--------------
compute_properties          Compute a set of properties for a list of molecules,
                            returning a DataFrame indexed by canonical SMILES.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

from . import to_mol, to_molecules


# ---------------------------------------------------------------------------
# Single-molecule property functions
# ---------------------------------------------------------------------------

def molecular_weight(molecule):
    """Return the molecular weight of a molecule in Daltons.

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    float

    Example
    -------
    >>> molecular_weight("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    180.16
    """
    return Descriptors.MolWt(to_mol(molecule))


def logp(molecule):
    """Return the Crippen estimated partition coefficient (LogP).

    A measure of lipophilicity. Values between -1 and 5 are typical for
    orally bioavailable drugs.

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    float

    Example
    -------
    >>> logp("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    1.31
    """
    return Descriptors.MolLogP(to_mol(molecule))


def topological_polar_surface_area(molecule):
    """Return the topological polar surface area (TPSA) in Å².

    TPSA correlates with membrane permeability. Values below 140 Å² are
    associated with good oral absorption.

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    float

    Example
    -------
    >>> topological_polar_surface_area("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    63.6
    """
    return rdMolDescriptors.CalcTPSA(to_mol(molecule))


def hydrogen_bond_donors(molecule):
    """Return the number of hydrogen-bond donors (NH and OH groups).

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    int

    Example
    -------
    >>> hydrogen_bond_donors("CC(=O)Nc1ccc(O)cc1")  # Paracetamol
    2
    """
    return rdMolDescriptors.CalcNumHBD(to_mol(molecule))


def hydrogen_bond_acceptors(molecule):
    """Return the number of hydrogen-bond acceptors (N and O atoms).

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    int

    Example
    -------
    >>> hydrogen_bond_acceptors("CC(=O)Nc1ccc(O)cc1")  # Paracetamol
    3
    """
    return rdMolDescriptors.CalcNumHBA(to_mol(molecule))


def rotatable_bonds(molecule):
    """Return the number of rotatable bonds (a measure of molecular flexibility).

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    int

    Example
    -------
    >>> rotatable_bonds("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")  # Ibuprofen
    4
    """
    return rdMolDescriptors.CalcNumRotatableBonds(to_mol(molecule))


def ring_count(molecule):
    """Return the total number of rings in the molecule.

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    int

    Example
    -------
    >>> ring_count("c1ccc2ccccc2c1")  # Naphthalene
    2
    """
    return rdMolDescriptors.CalcNumRings(to_mol(molecule))


def quantitative_estimate_of_drug_likeness(molecule):
    """Return the Quantitative Estimate of Drug-likeness (QED) score.

    QED is a composite score in [0, 1] that combines eight desirable
    drug-like properties. A score above 0.6 is generally considered drug-like.

    Parameters
    ----------
    molecule : str or rdkit.Chem.Mol

    Returns
    -------
    float in [0, 1]

    Example
    -------
    >>> quantitative_estimate_of_drug_likeness("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    0.55
    """
    return QED.qed(to_mol(molecule))


# ---------------------------------------------------------------------------
# Batch function
# ---------------------------------------------------------------------------

# Registry of molecular property functions, keyed by column name.
#
# To add your own property:
#   1. Write a function that accepts a SMILES string or an RDKit Mol object
#      and returns a single numeric value.  Use to_mol() to handle both input
#      types:
#
#          from evaluation import to_mol
#
#          def my_property(molecule):
#              mol = to_mol(molecule)
#              return <rdkit computation on mol>
#
#   2. Register it here:
#
#          PROPERTY_FUNCTIONS["my_property"] = my_property
#
#   Your property will then appear automatically in compute_properties() output
#   and in all plots that use it (plot_property_panel, plot_distribution_comparison).
PROPERTY_FUNCTIONS = {
    "molecular_weight": molecular_weight,
    "logp": logp,
    "topological_polar_surface_area": topological_polar_surface_area,
    "hydrogen_bond_donors": hydrogen_bond_donors,
    "hydrogen_bond_acceptors": hydrogen_bond_acceptors,
    "rotatable_bonds": rotatable_bonds,
    "ring_count": ring_count,
    "quantitative_estimate_of_drug_likeness": quantitative_estimate_of_drug_likeness,
}


def compute_properties(molecules, properties=None):
    """Compute molecular properties for a list of molecules.

    Returns a DataFrame indexed by canonical SMILES, with one column per
    property.  This output is directly compatible with `plot_property_panel`
    and `compare_distributions`.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
    properties : list of str, optional
        Subset of property names to compute.  Defaults to all properties.
        Available names: see ``PROPERTY_FUNCTIONS``.

    Returns
    -------
    pandas.DataFrame
        Rows = molecules (indexed by canonical SMILES), columns = properties.

    Raises
    ------
    ValueError
        If any name in ``properties`` is not in ``PROPERTY_FUNCTIONS``.

    Example
    -------
    >>> df = compute_properties(["CCO", "c1ccccc1"])
    >>> df.columns.tolist()
    ['molecular_weight', 'logp', ...]
    """
    if properties is not None:
        unknown = set(properties) - set(PROPERTY_FUNCTIONS)
        if unknown:
            raise ValueError(
                f"Unknown properties: {sorted(unknown)}.\n"
                f"Available: {sorted(PROPERTY_FUNCTIONS)}."
            )

    valid_molecules = to_molecules(molecules)
    canonical_smiles = [Chem.MolToSmiles(m) for m in valid_molecules]
    selected = {
        name: func
        for name, func in PROPERTY_FUNCTIONS.items()
        if properties is None or name in properties
    }
    data = {name: [func(mol) for mol in valid_molecules] for name, func in selected.items()}
    dataframe = pd.DataFrame(data, index=canonical_smiles)
    dataframe.index.name = "smiles"
    return dataframe
