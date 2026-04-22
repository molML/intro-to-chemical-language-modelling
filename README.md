# Workshop: Fine-Tuning a Chemical Language Model

In the morning lecture you learned how chemical language models (CLMs) are
pre-trained on large molecular libraries to learn the "grammar" of SMILES
strings. This workshop is the hands-on counterpart: you will fine-tune that
pre-trained model on your own molecular dataset, and evaluate what it learned.

**We (the instructors) will be walking around throughout the session — raise
your hand or call us over any time you have a question, hit an error, or want
to discuss your results. No question is too small.**

---

## Schedule

| Block | Content |
|-------|---------|
| ~1 h | Notebooks 1–4: clean, split, inspect, and augment your data |
| **— break —** | **Notebook 5: start fine-tuning just before the break — the model trains while you are away** |
| ~1 h | Notebook 6: evaluate your model's output and discuss results |

> Aim to reach the fine-tuning step (Notebook 5) and have the training
> running before the break. The training itself takes 10–20 minutes and
> does not need supervision.

---

## The pipeline at a glance

| Step | Notebook | What happens | Time |
|------|----------|--------------|------|
| 1 | `01_cleaning.ipynb` | Remove invalid and unsupported SMILES; filter by length | ~15 min |
| 2 | `02_data_splitting.ipynb` | Split into train / val / test using your chosen strategy | ~10 min |
| 3 | `03_inspect_reference.ipynb` | Visualise property distributions and scaffold diversity | ~10 min |
| 4 | `04_data_augmentation.ipynb` | Multiply training signal with randomised SMILES | ~10 min |
| **5** | **`05_finetuning_enumeration.ipynb`** | **Fine-tune the model — start before the break** | ~5 min setup |
| 6 | `06_evaluate_output.ipynb` | Score validity, novelty, and diversity of generated molecules | ~25 min |

Run the notebooks in order. Each notebook saves its output to disk so the
next one can pick it up. The only things you need to change in most notebooks
are the file paths at the top.

---

## Your dataset

Bring a `.csv` file with a column named `SMILES`, or a `.smi` file with one
SMILES string per line. Load it in any notebook with:

```python
from evaluation import load_smiles
smiles = load_smiles("path/to/your/file.csv")
```

Aim for at least ~100 clean molecules; the model can work with fewer but
evaluation metrics become less reliable below this threshold.

---

## Quick setup check

Run the following in a terminal before starting to confirm your environment
is working:

```bash
conda activate intro-to-clm-env
python -c "
from rdkit import rdBase
import tensorflow as tf
from evaluation import load_smiles, compute_fingerprints
print('RDKit     :', rdBase.rdkitVersion)
print('TensorFlow:', tf.__version__)
print('Setup OK')
"
```

You should see two version lines followed by `Setup OK`. If you see an error,
call one of the instructors before continuing.

---

## Step-by-step recipe

---

### Step 1 — Clean your data · `01_cleaning.ipynb` · ~15 min

**Why.** The pre-trained model has a fixed vocabulary of 62 SMILES tokens.
Molecules containing rare atoms, unusual bond types, or counter-ions cannot
be represented and must be removed. SMILES length is also filtered as a proxy
for molecular size: very short fragments and very long macromolecules are not
relevant for drug discovery.

**What to do.**
1. Set `csv_path` to your file and `smiles_column` to the correct column name
2. Run all cells in order
3. Check the printed summary — note what fraction of your molecules passed

**Done when:**
- [ ] `cleaned.csv` is saved in `dataset/cleaned_dataset/`
- [ ] The "Supported (kept)" percentage is printed
- [ ] At least ~50 molecules remain

**Common issues**
- *Column not found* — check that `smiles_column` matches your CSV header exactly (case-sensitive)
- *Very few molecules pass* — call an instructor; we can help

> **Exercise.** Try tightening the length filter (`max_length = 100`) and see
> how many more molecules are removed. Does it change the character of your
> dataset?

---

### Step 2 — Split into train / val / test · `02_data_splitting.ipynb` · ~10 min

**Why.** Holding out a test set lets you measure whether the model generalises
to molecules it never saw during training. The *way* you split determines what
kind of generalisation you are testing — from easy interpolation within the
same distribution all the way to hard extrapolation to new scaffolds or property
ranges.

The core splitting function is `split_by_values`, which accepts any list of
molecules and a parallel list of numeric values, sorts by those values, and
partitions into train / val / test. The choice of values determines the strategy:

| Values passed to `split_by_values` | Test set contains | What it tests |
|-------------------------------------|-------------------|---------------|
| Random numbers (`random_split`) | A random sample | Interpolation within the same distribution |
| Scaffold group values (`scaffold_split`) | Molecules with rarer ring systems | Partial extrapolation to new scaffolds |
| Molecular weight, LogP, QED, … | Molecules at the extreme of a property | Extrapolation in property space |

```python
from evaluation.splitting import split_by_values, random_split, scaffold_split
from evaluation.properties import compute_properties

# Random — simplest baseline
train, val, test = random_split(smiles, ratio=(0.8, 0.1, 0.1))

# Scaffold — harder, chemistry-aware
train, val, test = scaffold_split(smiles, ratio=(0.8, 0.1, 0.1))

# Property-based — plug in any per-molecule value
props = compute_properties(smiles)
qed_values = props["quantitative_estimate_of_drug_likeness"].tolist()
train, val, test = split_by_values(smiles, qed_values, high_values_in_test=True)
```

**What to do.**
1. Load your cleaned SMILES from Step 1
2. Run the random split and inspect the property distributions
3. Run the scaffold split and compare the nearest-neighbour distance (NND)
   between test and train — a larger NND means a harder, more informative test set
4. Try a property-based split with a property you find interesting
5. Choose one strategy to use for fine-tuning and save the three split files

**Done when:**
- [ ] `train.csv`, `val.csv`, and `test.csv` are saved
- [ ] You can explain why the scaffold split's NND (test → train) is larger than the random split's

**Common issues**
- *Empty val or test with scaffold split* — your dataset may have too few unique scaffolds; use the random split instead

> **Discussion.** Is a harder split always better? What does it mean if a
> model scores lower on a scaffold split than on a random split?

---

### Step 3 — Inspect your training data · `03_inspect_reference.ipynb` · ~10 min

**Why.** Before fine-tuning, spend a few minutes understanding the chemical
space you are targeting. Property distributions that look unexpected here are
a signal that something went wrong upstream — better to catch it now.

**What to do.**
1. Load your training split
2. Draw a grid of structures — do they look like what you expected?
3. Run `plot_property_panel` to see distributions of MW, LogP, TPSA, QED, and more
4. Check `plot_scaffold_frequencies` — are one or two scaffolds dominant?
5. Note the Lipinski Ro5 fraction (should be near 1.0 for drug-like sets)

Useful functions from the `evaluation` package:

```python
from evaluation.properties import compute_properties
from evaluation.visualization import (
    draw_molecule_grid,
    plot_property_panel,
    plot_scaffold_frequencies,
    plot_distribution_comparison,
    compare_distributions,
)
```

**Done when:**
- [ ] Property panel and scaffold frequency chart are displayed
- [ ] You can describe your training set in one sentence (e.g. *"~250 Da, mostly benzene-core scaffolds, all drug-like"*)

> **Exercise.** Compare your training and test splits side-by-side using
> `plot_distribution_comparison`. Are the property distributions similar or
> different? What does this tell you about how hard your split is?

---

### Step 4 — Augment with SMILES enumeration · `04_data_augmentation.ipynb` · ~10 min

**Why.** The same molecule can be written as many different — but equivalent —
SMILES strings depending on which atom the traversal starts from. Showing the
model multiple representations of each molecule provides more training signal
and reduces its dependence on the canonical atom-ordering convention.

**What to do.**
1. Set `augmentation_multiple` to 10 (a good starting point)
2. Run augmentation for your **training** split and save the output
3. Repeat for your **val** and **test** splits using the same `augmentation_multiple`
4. Confirm that the augmented row count ≈ original count × `augmentation_multiple`

**Done when:**
- [ ] `train.csv`, `val.csv`, and `test.csv` saved in `dataset/augmented_set/`
- [ ] Augmented row count ≈ original count × `augmentation_multiple`

> **Important — reporting statistics.** Augmented rows are representations of
> the same molecule, not independent data points. When you compute metrics in
> Step 6, always report *n* = original molecule count, not the augmented row count.

---

### Step 5 — Fine-tune the model · `05_finetuning_enumeration.ipynb` · start before the break

> **Start this step before the break.** The model training itself takes
> 10–20 minutes and runs without supervision — you can step away once it is
> running.

**Why.** The pre-trained LSTM already knows the grammar of SMILES from the
ChEMBL pre-training (see [Pre-trained model](#pre-trained-model)). Fine-tuning
adjusts its weights toward your chemistry using a much reduced learning rate —
so it learns your molecules without forgetting the general language.

**What to do.**
1. Verify the three paths at the top of the notebook (`results_dir`, `augmentation_dir`, `saving_dir`)
2. Run the encoding steps — check the printed array shapes
3. Run `fine_tune_model()` — watch the validation loss; it should decrease over the first few epochs
4. **Leave the notebook running through the break**
5. When you return: set `temperature` and run the sampling section in this notebook

**Done when (after the break):**
- [ ] Training completed; validation loss decreased over at least a few epochs
- [ ] A file of sampled SMILES is saved in `results/finetuning/`

**Common issues**
- *`model.h5` not found in `results/pretraining/`* — call an instructor; the checkpoint should already be there
- *Loss not decreasing at all* — try increasing `augmentation_multiple` in Step 4, or call an instructor

> **Exercise.** Sample at `temperature = 0.7` and again at `temperature = 1.3`.
> Does lower temperature give more valid SMILES? Does higher temperature give
> more diverse molecules? You will be able to quantify this in Step 6.

---

### Step 6 — Evaluate the model output · `06_evaluate_output.ipynb` · ~25 min

*Run this step after the break, once fine-tuning has completed.*

**Why.** A well fine-tuned model should produce molecules that are chemically
valid, mostly novel relative to the fine-tuning data, and structurally closer
to the fine-tuning chemistry than to the held-out test set. We compare
generated molecules against two references to distinguish "the model learned
the target chemistry" from "the model generalises to unseen molecules."

| Reference | Question answered |
|-----------|------------------|
| Fine-tuning set | Did the model learn the target chemistry? |
| Held-out test set | Does the model generalise to unseen molecules? |

**What to do.**
1. Load your generated SMILES, fine-tuning set, and test set
2. Check validity and uniqueness first — below 0.9 for either suggests a problem
3. Compute novelty against both references
4. Compare property distributions (generated vs each reference)
5. Compute nearest-neighbour distances and read the summary table

**Done when:**
- [ ] Summary table with all metrics is displayed
- [ ] NND (generated → fine-tuning set) is smaller than NND (generated → test set) — this is the signature of successful fine-tuning

> **Discussion.** If novelty vs. the fine-tuning set is very low (~0), what
> does that suggest? How does changing the sampling temperature affect the
> novelty / validity trade-off?

---

## Evaluation metrics quick reference

| Metric | What it measures | Healthy range |
|--------|-----------------|---------------|
| Validity | Fraction of outputs parseable as molecules | > 0.90 |
| Uniqueness | Fraction of valid outputs that are distinct | > 0.80 |
| Novelty (vs. fine-tuning set) | Fraction not seen during training | 0.5–0.9 for a well-tuned model |
| Mean pairwise distance | Internal structural diversity | Compare to fine-tuning set baseline |
| Scaffold entropy | Variety of ring systems | Higher = more scaffold-diverse |
| NND → fine-tuning set | Closeness to training data | Should be **lower** than NND to test |
| NND → test set | Closeness to held-out data | Should be **higher** than NND to fine-tuning |
| Lipinski Ro5 | Drug-likeness | > 0.90 for drug-like sets |

Full mathematical definitions are in [`docs/metrics_reference.md`](docs/metrics_reference.md).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` for rdkit or tensorflow | Run `conda activate intro-to-clm-env`; call an instructor if it persists |
| *Column `SMILES` not found* | Check the column name in your CSV — it is case-sensitive |
| Very few molecules pass cleaning | Call an instructor |
| *`model.h5` not found in `results/pretraining/`* | Call an instructor — the checkpoint should already be there |
| Training loss not decreasing | Try increasing `augmentation_multiple` or reducing `batch_size_finetune` |
| All novelty values = 1.0 | Check that you are loading canonical (non-augmented) SMILES for the reference sets |

---

## Pre-trained model

The pre-trained LSTM backbone was trained on a cleaned subset of ChEMBL
(~2 million drug-like molecules) using randomised SMILES with an augmentation
factor of 1 per molecule — a deliberate choice to avoid biasing the
pre-training distribution toward any particular chemical series.

The model architecture and pre-training strategy are described in:

> Brinkmann H, Argante A, Ter Steege H, Grisoni F. Going beyond SMILES
> enumeration for data augmentation in generative drug discovery.
> *Digit Discov.* 2025 Aug 14;4(10):2752–2764.
> doi: [10.1039/d5dd00028a](https://doi.org/10.1039/d5dd00028a).
> PMID: 40917333.

---

## Repository structure

```
├── scripts/
│   ├── smiles_processing.py   SMILES cleaning and validation
│   ├── encoding.py            Tokenisation and one-hot encoding
│   ├── model.py               LSTM model architecture (Keras; defines CLM)
│   └── sampling.py            Temperature sampling from the fine-tuned model in model.py
│
├── evaluation/
│   ├── __init__.py            load_smiles, to_mol, compute_fingerprints, compute_scaffolds
│   ├── metrics.py             validity, uniqueness, novelty, mean_pairwise_distance, etc.
│   ├── properties.py          RDKit property wrappers + compute_properties()
│   ├── splitting.py           split_by_values, random_split, scaffold_split
│   └── visualization.py       plot_property_panel, plot_scaffold_frequencies, etc.
│
├── docs/
│   └── metrics_reference.md   Mathematical definitions of all evaluation metrics
│
├── results/
│   ├── segment2label.json     Vocabulary (62 tokens)
│   └── pretraining/
│       ├── model.h5           Pre-trained LSTM weights
│       └── combination.json   Hyperparameter configuration
│
└── env.yml                    Conda environment specification
```

---

## Design notes

- **Augmentation applies to all splits.** Train, val, and test are all
  augmented with the same `augmentation_multiple`, keeping all splits in the
  same SMILES representation space as the pre-trained model. When reporting
  statistics, always use *n* = original molecule count, not augmented row count.
- **Canonical SMILES for evaluation.** Reference sets in Notebook 6 use
  canonical deduplicated SMILES — not augmented representations.
- **Two evaluation references.** Generated molecules are compared against the
  fine-tuning set (what the model learned) and the held-out test set
  (generalisation to unseen chemistry) separately.
- **`split_by_values` is the core splitting primitive.** Both `random_split`
  and `scaffold_split` are thin wrappers around it. Passing any numeric
  per-molecule value — MW, LogP, QED, NND, or anything you compute — directly
  to `split_by_values` lets you design any splitting strategy you want.
