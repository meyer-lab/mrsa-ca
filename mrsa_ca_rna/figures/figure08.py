"""
Code copied from figure07 but plotting components of MRSA data against patient metadata
Now, we use anndata to easily access the metadata and RNA data for MRSA patients
"""

from mrsa_ca_rna.regression import perform_PLSR
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.figures.base import setupBase

import pandas as pd
import seaborn as sns
import numpy as np


def figure08_setup():
    whole_data = concat_datasets()

    mrsa_df = whole_data[whole_data.obs["disease"] == "MRSA"].to_df()
    ca_df = whole_data[whole_data.obs["disease"] == "Candidemia"].to_df()
    # mrsa_whole = whole_data.loc["MRSA", :]
    # ca_whole = whole_data.loc["Candidemia", :]

    X_data = mrsa_df.T
    y_data = ca_df.T

    scores, loadings, model = perform_PLSR(X_data, y_data, 10)

    """
    Set up two dfs to pass to genFig(), one for the loadings of the components
    and one for the MRSA metadata
    """

    mrsa_loadings = loadings["X"]
    mrsa_meta = whole_data.obs.loc[
        whole_data.obs["disease"] == "MRSA", ["gender", "age", "status"]
    ]

    # # start by plotting mrsa (x) loadings with status metadata to find components associated with outcome
    # mrsa_loadings :pd.DataFrame = pd.concat([mrsa_whole["meta"]["status"], loadings["X"]], axis=1)

    # # order mrsa_laodings dataframe by status to better visualize the components
    # mrsa_loadings = mrsa_loadings.sort_values(by="status")

    return mrsa_loadings, mrsa_meta


def genFig():
    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data, meta = figure08_setup()

    meta.loc[:, :] = meta.to_numpy(dtype=float)

    gender_colors = [(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)]
    status_colors = [(0.5, 0.5, 0.1), (0.9, 0.9, 0.1)]

    a = sns.heatmap(meta.loc[:, ["age"]].to_numpy(dtype=float), ax=ax[0])
    a = sns.heatmap(
        meta.loc[:, ["status"]].to_numpy(dtype=float), cmap=status_colors, ax=ax[1]
    )
    colorbar = a.collections[0].colorbar
    M = meta["status"].max()
    colorbar.set_ticks([0, M])
    colorbar.set_ticklabels(["Negative", "Positive"])

    a = sns.heatmap(data, ax=ax[2])
    a.set_title("PLSR components of MRSA (X) data")
    # a.set_xlabel("Components")
    # a.set_ylabel("Patient outcomes")

    return f
