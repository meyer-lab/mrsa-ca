"""
Plots a figure of the explained variance of the PCA components for each dataset.
Datasets are MRSA, MRSA+CA, and CA.
"""

import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca


def figure_00_setup():
    """Make and organize the data to be used in genFig"""

    # list the datasets we want to compare and define total components
    datasets = ["mrsa", "ca"]
    components = 70

    # grab the combined dataset
    combined_data = concat_datasets(datasets, scale=False, tpm=True)

    # split it into the datasets we want to compare
    mrsa_data = combined_data[combined_data.obs["disease"] == "MRSA"].copy()
    ca_data = combined_data[combined_data.obs["disease"] == "Candidemia"].copy()

    # convert the datasets to pd.dataframes to hand to perform_pca
    combined_data = combined_data.to_df()
    mrsa_data = mrsa_data.to_df()
    ca_data = ca_data.to_df()

    # perform PCA on the datasets
    _, _, combined_pca = perform_pca(combined_data, components)
    _, _, mrsa_pca = perform_pca(mrsa_data, components)
    _, _, ca_pca = perform_pca(ca_data, components)

    # get the cumulative explained variance for each dataset
    combined_explained = np.cumsum(combined_pca.explained_variance_ratio_)
    mrsa_explained = np.cumsum(mrsa_pca.explained_variance_ratio_)
    ca_explained = np.cumsum(ca_pca.explained_variance_ratio_)

    # create a dataframe to hold the data (1-70, explained variance)
    variance_explained = pd.DataFrame(
        np.arange(1, components + 1, dtype=int), columns=pd.Index(["components"])
    )
    variance_explained["combined"] = combined_explained
    variance_explained["mrsa"] = mrsa_explained
    variance_explained["ca"] = ca_explained

    return variance_explained


def genFig():
    """
    Start making the figure.
    """
    fig_size = (3, 3)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    explained_variance = figure_00_setup()

    # convert the data to long form for seaborn
    explained_variance = explained_variance.melt(
        id_vars="components", var_name="dataset", value_name="explained_variance"
    )

    a = sns.lineplot(
        data=explained_variance,
        x="components",
        y="explained_variance",
        hue="dataset",
        ax=ax[0],
    )
    a.set_title("Comparison of PCA Explained Variance across datasets")
    a.set_xlabel("Components")
    a.set_ylabel("Explained Variance")
    a.legend(title="Dataset")

    return f
