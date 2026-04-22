# Molecular Evaluation Metrics — Reference Guide

This document explains the mathematical formulation of every metric and utility
implemented in the `evaluation` package.  Each section gives the formula, an
intuition for chemists, and guidance on how to interpret the result.

All functions are imported from `evaluation/metrics.py` unless otherwise noted.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $S$ | The full input: a list of SMILES strings |
| $M$ | The set of **valid** molecules parsed from $S$ ( $|M| \leq |S|$ ) |
| $R$ | A **reference** set of molecules (e.g., training data, fine-tuning set) |
| $c(m)$ | Canonical SMILES of molecule $m$ — a unique, standardised string representation |
| $\text{fp}(m)$ | Morgan fingerprint bit-vector of molecule $m$ (ECFP4, radius 2, 2048 bits) |

---

## Fingerprints and Tanimoto Distance

Several diversity metrics rely on comparing molecular fingerprints.  A
**Morgan fingerprint** (also called ECFP4) encodes the circular chemical
neighbourhood of each atom into a fixed-length bit-vector.  Two molecules with
similar local chemical environments will have similar fingerprints.

The **Tanimoto similarity** between two fingerprints $a$ and $b$ is:

$$T(a,\, b) = \frac{|a \cap b|}{|a \cup b|}$$

where $|a \cap b|$ is the number of bits set in *both* fingerprints and
$|a \cup b|$ is the number of bits set in *either*.  $T$ ranges from 0
(nothing in common) to 1 (identical).

The corresponding **Tanimoto distance** is simply:

$$d(a,\, b) = 1 - T(a,\, b)$$

A distance of 0 means the molecules are structurally identical; a distance of
1 means they share no fingerprint bits at all.  In practice, two structurally
unrelated drug-like molecules typically have a distance of 0.8–1.0.

---

## 1. Set Quality Metrics

These metrics characterise the basic quality of a molecular set — how many
are chemically valid, how many are unique, and how many are genuinely new
relative to some reference.

---

### 1.1 Validity

**Intuition.**  A generative model produces a list of SMILES strings.  Not all
of them will correspond to a real, parseable molecule.  Validity measures what
fraction are chemically meaningful.

**Formula.**

$$\text{Validity}(S) = \frac{|M|}{|S|}$$

**Interpretation.**
- **1.0** — every output is a valid molecule.
- **< 0.9** — a notable fraction of outputs are chemical nonsense; consider
  filtering or retraining.
- For a curated input set, validity should always be 1.0.

---

### 1.2 Uniqueness

**Intuition.**  A model that generates the same molecule repeatedly is not
exploring chemical space.  Uniqueness measures what fraction of the valid
outputs are structurally distinct.

**Formula.**

$$\text{Uniqueness}(M) = \frac{\bigl|\{c(m) : m \in M\}\bigr|}{|M|}$$

where the numerator counts the number of *distinct* canonical SMILES.

**Interpretation.**
- **1.0** — every molecule is unique.
- **< 0.8** — substantial repetition; the model may be overfit or lacking diversity.

---

### 1.3 Novelty

**Intuition.**  If a model only reproduces molecules it saw during training, it
has memorised the data rather than generalised from it.  Novelty measures the
fraction of generated molecules that are genuinely new — not seen in the
reference set.

**Formula.**

$$\text{Novelty}(M,\, R) =
\frac{\bigl|\{m \in M : c(m) \notin \{c(r) : r \in R\}\}\bigr|}{|M|}$$

**Interpretation.**
- **1.0** — no generated molecule appeared in the reference set.
- **0.0** — every generated molecule was in the reference set (complete memorisation).
- In practice, values of 0.7–1.0 are expected for a well-trained generative model.
- Novelty depends on the choice of reference set.  It is common to compute it
  relative to both the fine-tuning set and a pre-training subsample.

---

## 2. Structural Diversity Metrics

These metrics quantify how structurally varied a molecular set is, using
Tanimoto distance as the measure of structural difference between molecules.

---

### 2.1 Mean Pairwise Distance

**Intuition.**  The average distance between every pair of molecules in the
set.  A library of very different scaffolds will have a high mean pairwise
distance; a focused analogue series around one core will have a low value.

**Formula.**

$$\text{MeanPairwiseDistance}(M) =
\frac{2}{|M|(|M|-1)} \sum_{i < j} d\bigl(\text{fp}(m_i),\, \text{fp}(m_j)\bigr)$$

For large sets, a random subsample of up to `sample_size` molecules is used.

**Interpretation.**
- **> 0.85** — highly diverse (typical for a broad drug-like library).
- **0.6–0.85** — moderately diverse.
- **< 0.6** — low structural diversity; the set is focused around a narrow chemical series.

---

### 2.2 Nearest-Neighbor Distance

**Intuition.**  For each molecule in the query set, find its closest structural
neighbour in a reference set and record the Tanimoto distance to that neighbour.
The mean over all query molecules tells you how structurally "far" the query
set is from the reference.

This single function covers two use cases controlled by the `reference` argument:

| Call | Meaning |
|------|---------|
| `nearest_neighbor_distance(M)` | **Internal** — how spread out is the set relative to itself? |
| `nearest_neighbor_distance(M, R)` | **External** — how far are the query molecules from the reference? |

**Formula (external).**

$$\text{NND}(M,\, R) =
\frac{1}{|M|} \sum_{m \in M}
\min_{r \in R}\, d\bigl(\text{fp}(m),\, \text{fp}(r)\bigr)$$

**Formula (internal, self-excluding).**

$$\text{NND}(M) =
\frac{1}{|M|} \sum_{m \in M}
\min_{r \in M \setminus \{m\}}\, d\bigl(\text{fp}(m),\, \text{fp}(r)\bigr)$$

**Interpretation.**
- **Low external NND** — generated molecules are structurally close to the
  reference set (less novel, but potentially more realistic).
- **High external NND** — generated molecules explore new chemical space
  (more novel, but may also be less drug-like if reference is a curated set).
- **Internal NND** gives a diversity summary focused on the nearest
  neighbour rather than the average; it is complementary to mean pairwise distance.

---

### 2.3 Scaffold Entropy

**Intuition.**  Two molecular sets can have similar average pairwise distances
yet differ greatly in the variety of ring systems (scaffolds) they contain.
Scaffold entropy measures diversity at the level of molecular scaffolds, using
the Bemis–Murcko scaffold decomposition.

A **Bemis–Murcko scaffold** is obtained by keeping the ring systems and the
linker atoms between them, while removing all side chains.  Aspirin and
ibuprofen, for example, both have a benzene scaffold despite their different
functional groups.

**Formula.**

Let $p_k$ be the fraction of molecules with scaffold $k$:

$$p_k = \frac{|\{m \in M : \text{scaffold}(m) = k\}|}{|M|}$$

$$H(M) = -\sum_{k} p_k \ln p_k$$

**Interpretation.**
- **Higher** — more scaffold diversity (many different ring systems).
- **0** — all molecules share the same scaffold.
- Values are in nats; divide by $\ln 2$ to convert to bits.
- Compare sets of similar size for a fair comparison.

---

## 3. Drug-likeness

### 3.1 Lipinski's Rule of Five

**Intuition.**  Christopher Lipinski's Rule of Five (Ro5) is a set of
physicochemical criteria associated with good oral bioavailability in humans.
The fraction of molecules in a set that satisfy all four criteria gives a quick
indication of how "drug-like" the set is.

**Criteria.**

| Property | Threshold |
|----------|-----------|
| Molecular weight | $\leq 500$ Da |
| LogP (lipophilicity) | $\leq 5$ |
| Hydrogen-bond donors | $\leq 5$ |
| Hydrogen-bond acceptors | $\leq 10$ |

**Formula.**

$$\text{Lipinski}(M) = \frac{1}{|M|} \sum_{m \in M} \mathbb{1}\bigl[\text{MW}(m) \leq 500 \;\wedge\; \text{LogP}(m) \leq 5 \;\wedge\; \text{HBD}(m) \leq 5 \;\wedge\; \text{HBA}(m) \leq 10\bigr]$$

**Interpretation.**
- **> 0.9** — the set is largely drug-like according to Ro5.
- Note: biologics (peptides, antibodies) routinely violate Ro5; this metric is
  intended for small-molecule libraries only.

---

## 4. Molecular Properties

These functions (in `evaluation/properties.py`) wrap RDKit's descriptor
calculations.  Each accepts a single molecule (SMILES or Mol) or a batch via
`compute_properties`.

| Function | Unit | Description |
|----------|------|-------------|
| `molecular_weight` | Da | Molecular weight |
| `logp` | — | Crippen estimated partition coefficient (lipophilicity) |
| `topological_polar_surface_area` | Å² | Polar surface area; correlates with membrane permeability |
| `hydrogen_bond_donors` | count | NH and OH groups |
| `hydrogen_bond_acceptors` | count | N and O atoms |
| `rotatable_bonds` | count | Molecular flexibility |
| `ring_count` | count | Total number of rings |
| `quantitative_estimate_of_drug_likeness` | [0, 1] | Composite QED score; > 0.6 is considered drug-like |

`compute_properties(molecules)` returns a **pandas DataFrame** indexed by
canonical SMILES, with one column per property.  It is directly compatible with
`plot_property_panel` and `compare_distributions`.

---

## 5. Statistical Distribution Tests

When comparing two sets of molecules (e.g., train vs. test, generated vs.
reference), it is useful to formally test whether their property distributions
differ.  Use `compare_distributions` from `evaluation/visualization.py`.

### Kolmogorov–Smirnov Test (`test='ks'`)

A non-parametric test that measures the maximum difference between the two
empirical cumulative distribution functions $F_n$ and $G_m$:

$$D = \sup_x \bigl|F_n(x) - G_m(x)\bigr|$$

Sensitive to both location (mean) and shape differences.  **Recommended for
comparing continuous molecular properties** (MW, LogP, TPSA, …).

### Mann–Whitney U Test (`test='mw'`)

A rank-based test that asks whether one sample tends to have larger values
than the other.  Sensitive primarily to differences in the median.

### Interpreting the result

`compare_distributions` returns a dictionary:

```python
{"test": "ks", "statistic": 0.21, "p_value": 0.43, "significant": False}
```

- A **p-value < 0.05** means the two distributions are *statistically*
  significantly different.
- Always look at the visual comparison alongside the p-value — a statistically
  significant difference in a large dataset may be too small to be chemically
  meaningful, and a non-significant result in a small dataset may simply reflect
  low statistical power.

---

## 6. Data Splitting

The `evaluation/splitting.py` module provides a two-component framework for
dividing a molecular dataset into training, validation, and test subsets.

### Component 1 — `split_by_values` (the core function)

```python
split_by_values(molecules, values, ratio=(0.8, 0.1, 0.1), high_values_in_test=True)
```

Accepts any list of molecules and a parallel list of numeric values.  Molecules
are sorted by their value and partitioned into three sets:

| `high_values_in_test` | Train | Validation | Test |
|-----------------------|-------|------------|------|
| `True` (default) | lowest values | middle values | **highest values** |
| `False` | highest values | middle values | **lowest values** |

The choice of values determines the split strategy:

| Values | Effect | When to use |
|--------|--------|-------------|
| Random numbers | Random split | Baseline; quick sanity check |
| 1 / scaffold frequency | Rare scaffolds → test | Realistic evaluation; tests generalisation to new scaffolds |
| Molecular weight | Heaviest molecules → test | Check if model generalises across size |
| Nearest-neighbor distance | Most distant molecules → test | Most challenging evaluation |

### Component 2 — Convenience wrappers

`random_split` and `scaffold_split` are thin wrappers that compute the
appropriate values and call `split_by_values` internally.  Students can plug
any per-molecule property or metric directly into `split_by_values` to design
their own split.

### Choosing a split strategy

A **random split** is the simplest baseline.  Property and scaffold splits
are progressively more challenging: if the test set contains scaffolds or
property ranges not well-represented in training, the model must truly
generalise rather than interpolate.  After splitting, always compare the
property distributions of train and test visually and statistically to
understand how challenging your split actually is.
