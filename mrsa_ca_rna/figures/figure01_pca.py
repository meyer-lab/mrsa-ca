"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:
    Refactor file based on new perform_PCA output.
        Now outputs a scores matrix containing metadata for easy access.
        No longer any need to generate labels within the fig setup func.

    Make a function that takes in PCA component data and automatically determines the
        appropriate size and layout of the graph. Can remove assert once
        completed.

    Figure out how to plot PCA with sns.pairplot instead of sns.scatterplot
        Seems like it would negate the need for component_pairs since it pairs
        the data itself? Can't get a good sense of it just from the documentation
        yet.
"""

import numpy as np
import pandas as pd

from mrsa_ca_rna.import_data import (
    import_mrsa_rna,
    import_ca_rna,
    import_GSE_rna,
    concat_datasets,
)
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns
import matplotlib.pyplot as plt


def genFig():
    fig_size = (12, 9)
    layout = {
        "ncols": 4,
        "nrows": 3,
    }
    ax, f, _ = setupBase(fig_size, layout)

    scores, _, _ = perform_PCA()

    # modify what components you want to compare to one another:
    component_pairs = np.array(
        [
            [1, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [3, 4],
            [3, 5],
            [4, 5],
            [4, 6],
            [5, 6],
            [5, 7],
            [6, 7],
            [7, 8],
        ],
        dtype=int,
    )

    assert (
        component_pairs.shape[0] == layout["ncols"] * layout["nrows"]
    ), "component pairs to be graphed do not match figure layout size"

    for i, (j, k) in enumerate(component_pairs):
        a = sns.scatterplot(
            data=scores.loc[:, (scores.columns[j + 1], scores.columns[k + 1])],
            x=scores.columns[j + 1],
            y=scores.columns[k + 1],
            hue=scores.loc[:, "disease"],
            ax=ax[i],
        )

        a.set_xlabel(scores.columns[j + 1])
        a.set_ylabel(scores.columns[k + 1])
        a.set_title(f"Var Comp {scores.columns[j+1]} vs {scores.columns[k+1]}")

    return f
