"""
This file is the start of analyzing the loadings of the
mrsa+ca+healthy data, based on previous scores analysis.
This file may become obsolete post scores heatmap analysis currently
planned.

To-do:
    change file to make loadings DataFrame
"""

import numpy as np
import pandas as pd

from mrsa_ca_rna.import_data import concat_datasets
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

    rna_mat = concat_datasets()
    scores, loadings, pca = perform_PCA()

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

    # reduce all readable component pair values by 1 since indexes start at 0
    component_pairs -= 1

    for i, (j, k) in enumerate(component_pairs):
        a = sns.scatterplot(
            data=loadings.loc[(loadings.index[j], loadings.index[k]), :].T,
            x=loadings.index[j],
            y=loadings.index[k],
            ax=ax[i],
        )
        a.set_xlabel(loadings.index[j])
        a.set_ylabel(loadings.index[k])
        a.set_title(f"Feature Var by {loadings.index[j]} vs {loadings.index[k]}")

    return f


"""Debug function call section"""
fig = genFig()
fig.savefig("./mrsa_ca_rna/output/fig02_Loadings_NewPCA.png")
