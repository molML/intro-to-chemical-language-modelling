"""
evaluation.visualization — plotting utilities and statistical comparison.

All plot functions accept an optional `ax` argument so they can be embedded in
any subplot layout.  When `ax` is None the current Matplotlib axes is used.

Plotting functions
------------------
draw_molecule_grid          Display molecules as a grid image (RDKit).
plot_distribution           Histogram + KDE for a single numeric series.
plot_distribution_comparison Overlay two distributions for visual comparison.
plot_property_panel         Multi-panel figure of all columns in a DataFrame.
plot_scaffold_frequencies   Bar chart of the most common scaffolds.
plot_training_history       Plot training/validation loss and accuracy curves.

Statistical testing
-------------------
compare_distributions       Test whether two distributions differ significantly.
"""

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu

from . import to_molecules, compute_scaffolds


# ---------------------------------------------------------------------------
# Molecule grid
# ---------------------------------------------------------------------------

def draw_molecule_grid(molecules, n_cols=5, legends=None, subimage_size=(300, 300)):
    """Display molecules as a grid image using RDKit.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
    n_cols : int, default 5
        Number of columns in the grid.
    legends : list of str, optional
        One label per molecule shown below each structure.
    subimage_size : tuple of (width, height) in pixels, default (300, 300)

    Returns
    -------
    PIL.Image
        Can be displayed directly in Jupyter with `display(image)`.

    Example
    -------
    >>> image = draw_molecule_grid(smiles_list, n_cols=5)
    >>> display(image)
    """
    from rdkit.Chem import Draw
    valid_molecules = to_molecules(molecules)
    if legends is not None and len(legends) != len(valid_molecules):
        raise ValueError(
            f"legends length ({len(legends)}) must match the number of valid "
            f"molecules ({len(valid_molecules)})."
        )
    return Draw.MolsToGridImage(
        valid_molecules,
        molsPerRow=n_cols,
        subImgSize=subimage_size,
        legends=legends,
    )


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_distribution(values, label, ax=None, color=None, bins=30):
    """Plot the distribution of a numeric series as a histogram with a KDE overlay.

    Parameters
    ----------
    values : array-like
        Numeric values to plot.
    label : str
        Series name — used as the x-axis label and legend entry.
    ax : matplotlib.axes.Axes, optional
        Target axes.  Uses `plt.gca()` if None.
    color : str, optional
        Colour for both the histogram and the KDE line.
    bins : int, default 30

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> plot_distribution(molecular_weights, label="molecular_weight", ax=ax)
    """
    if ax is None:
        ax = plt.gca()

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(values, bins=bins, density=True, alpha=0.4, color=color, label=label)

    if len(values) < 2 or np.allclose(values, values[0]):
        ax.set_xlabel(label.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.legend()
        return ax

    try:
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2)
    except (ValueError, np.linalg.LinAlgError):
        pass
    ax.set_xlabel(label.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def plot_distribution_comparison(values_a, values_b, labels, xlabel=None, ax=None):
    """Overlay two distributions on the same axes for visual comparison.

    Parameters
    ----------
    values_a, values_b : array-like
        The two numeric series to compare.
    labels : list of two str
        Names for each series (e.g., ["Train", "Test"]).
    xlabel : str
        Name of the value the distribution is plot for, to place as x-axis label.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> plot_distribution_comparison(train_weights, test_weights,
    ...                              labels=["Train", "Test"], ax=ax)
    """
    if ax is None:
        ax = plt.gca()
    plot_distribution(values_a, label=labels[0], ax=ax, color="steelblue")
    plot_distribution(values_b, label=labels[1], ax=ax, color="coral")
    if xlabel is None:
        xlabel = ""
    ax.set_xlabel(xlabel.replace("_", " ").title())
    return ax


def plot_property_panel(dataframe, figsize=None):
    """Plot distributions for every column in a properties DataFrame.

    Designed to work directly with the output of `compute_properties`.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        One column per molecular property.
    figsize : tuple of (width, height), optional
        Defaults to (4 × n_cols, 3 × n_rows).

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> df = compute_properties(smiles_list)
    >>> fig = plot_property_panel(df)
    >>> plt.show()
    """
    columns = dataframe.columns.tolist()
    n_properties = len(columns)
    n_cols = min(4, n_properties)
    n_rows = (n_properties + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    figure, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, column in enumerate(columns):
        plot_distribution(dataframe[column].dropna().values, label=column, ax=axes[i])

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return figure


def plot_scaffold_frequencies(molecules, top_n=15, ax=None):
    """Horizontal bar chart of the most frequent Bemis–Murcko scaffolds.

    Parameters
    ----------
    molecules : list of str or list of rdkit.Chem.Mol
    top_n : int, default 15
        Number of most-frequent scaffolds to display.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> plot_scaffold_frequencies(smiles_list, top_n=10, ax=ax)
    """
    if ax is None:
        ax = plt.gca()

    valid_molecules = to_molecules(molecules)
    if not valid_molecules:
        ax.set_title("No valid molecules to display")
        return ax

    scaffolds = compute_scaffolds(valid_molecules)
    top_scaffolds = Counter(scaffolds).most_common(top_n)

    if not top_scaffolds:
        ax.set_xlabel("Count")
        ax.set_title(f"Top {top_n} Bemis–Murcko Scaffolds")
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "No scaffolds to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax
    if not top_scaffolds:
        ax.set_title("No scaffolds found")
        return ax

    scaffold_labels, counts = zip(*top_scaffolds)
    # Truncate long SMILES for readability
    display_labels = [s[:25] + "…" if len(s) > 25 else s for s in scaffold_labels]

    ax.barh(range(len(display_labels)), counts, tick_label=display_labels)
    ax.invert_yaxis()  # most frequent at the top
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} Bemis–Murcko Scaffolds")
    return ax


def plot_training_history(history, figsize=(12, 4)):
    """Plot training and validation loss & accuracy curves over epochs.

    Parameters
    ----------
    history : dict or tensorflow.keras.callbacks.History
        Training history object from model.fit() or a dictionary containing
        keys: "loss", "val_loss", "accuracy", "val_accuracy".
    figsize : tuple of (width, height), optional
        Figure size. Default is (12, 4).

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two subplots: loss curves and accuracy curves.

    Example
    -------
    >>> model, history = trainer.fine_tune_model(xTrain, yTrain, xVal, yVal)
    >>> fig = plot_training_history(history)
    >>> plt.show()
    """
    # Extract history dictionary if passed a Keras History object
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history

    # Validate that required keys are present
    required_keys = {'loss', 'val_loss', 'accuracy', 'val_accuracy'}
    if not required_keys.issubset(history_dict.keys()):
        missing = required_keys - set(history_dict.keys())
        raise ValueError(
            f"History dictionary missing required keys: {missing}. "
            f"Available keys: {list(history_dict.keys())}"
        )

    epochs = range(1, len(history_dict['loss']) + 1)

    figure, axes = plt.subplots(1, 2, figsize=figsize)

    # ────────────────────────────────── Loss curves
    axes[0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ────────────────────────────────── Accuracy curves
    axes[1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_distributions(values_a, values_b, test="ks", significance_level=0.05):
    """Test whether two distributions are statistically significantly different.

    Two tests are available:
        'ks'  Kolmogorov–Smirnov test — non-parametric, sensitive to both
              location and shape differences.  Good for continuous properties.
        'mw'  Mann–Whitney U test — non-parametric rank test.  Sensitive to
              differences in the median / central tendency.

    Parameters
    ----------
    values_a, values_b : array-like
        The two numeric series to compare.
    test : str, default 'ks'
        Statistical test to use: 'ks' or 'mw'.
    significance_level : float, default 0.05
        Threshold p-value for declaring significance.

    Returns
    -------
    dict with keys:
        test           Name of the test used.
        statistic      Test statistic.
        p_value        Two-sided p-value.
        significant    True if p_value < significance_level.

    Example
    -------
    >>> result = compare_distributions(train_weights, test_weights, test="ks")
    >>> print(result["p_value"])   # large p-value → distributions are similar
    """
    available_tests = {
        "ks": lambda a, b: ks_2samp(a, b),
        "mw": lambda a, b: mannwhitneyu(a, b, alternative="two-sided"),
    }
    if test not in available_tests:
        raise ValueError(f"Unknown test '{test}'. Choose from: {list(available_tests.keys())}")

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    values_a = values_a[~np.isnan(values_a)]
    values_b = values_b[~np.isnan(values_b)]

    if values_a.size == 0 or values_b.size == 0:
        raise ValueError("values_a and values_b must each contain at least one non-NaN value")

    result = available_tests[test](values_a, values_b)
    return {
        "test": test,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < significance_level),
    }
