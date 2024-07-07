"""
Generate a figure depicting the explained variance of the PCA
analysis. We're looking to see diminishing returns after several
components.

To-do:
    Cleanup after writing more of these figure files
    Regenerate plots using recently added gse_healthy data
"""

import numpy as np
import pandas as pd
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns


from mrsa_ca_rna.pca import perform_PCA


def figure_00_setup():
    """Make and organize the data to be used in genFig"""
    _, _, pca = perform_PCA()

    components = np.arange(1, pca.n_components_ + 1, dtype=int)
    total_explained = np.cumsum(pca.explained_variance_ratio_)

    data = pd.DataFrame(components, columns=["components"])
    data["total_explained"] = total_explained

    return data


def genFig():
    """
    Start making the figure.
    """
    fig_size = (3, 3)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data = figure_00_setup()
    a = sns.lineplot(data=data, x="components", y="explained", ax=ax[0])
    a.set_xlabel("# of Components")
    a.set_ylabel("Fraction of explained variance")
    a.set_title("PCA performance")

    return f
