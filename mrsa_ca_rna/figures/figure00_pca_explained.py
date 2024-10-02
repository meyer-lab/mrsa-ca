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
import seaborn as sns

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca


def figure_00_setup():
    """Make and organize the data to be used in genFig"""

    whole_data = concat_datasets(scale=False, tpm=True)

    datasets = {
        "MRSA": whole_data[whole_data.obs["disease"] == "MRSA"].to_df(),
        "MRSA+CA+Healthy": whole_data.to_df(),
        "CA": whole_data[whole_data.obs["disease"] == "Candidemia"].to_df(),
    }

    for dataset in datasets:
        _, _, pca = perform_pca(datasets[dataset])

        components = np.arange(1, pca.n_components_ + 1, dtype=int)
        total_explained = np.cumsum(pca.explained_variance_ratio_)

        data = pd.DataFrame(components, columns=pd.Index(["components"]))
        data["total_explained"] = total_explained

        datasets[dataset] = data

    return datasets


def genFig():
    """
    Start making the figure.
    """
    fig_size = (9, 3)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    datasets = figure_00_setup()

    for i, dataset in enumerate(datasets):
        a = sns.lineplot(
            data=datasets[dataset], x="components", y="total_explained", ax=ax[i]
        )
        a.set_xlabel("# of Components")
        a.set_ylabel("Fraction of explained variance")
        a.set_title(f"PCA performance of {dataset} dataset")

    return f
