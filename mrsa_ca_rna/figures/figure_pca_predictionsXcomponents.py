"""
This figure will plot the nested cross-validation performance of logistic regression
on PCA-transformed MRSA, Combined, and CA datasets as a function of the number of
components used in the PCA transformation.
"""

import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    """Create a dataFrame of regression performance over component #"""
    components = 15

    # collect the mrsa and ca data
    datasets = ["mrsa", "ca"]
    diseases = ["MRSA", "Candidemia"]
    whole_data = concat_datasets(
        datasets,
        diseases,
        scale=False,
    )

    # split the data into the datasets we want to compare
    mrsa_split = whole_data[whole_data.obs["disease"] == "MRSA"].copy()
    # ca_split = whole_data[whole_data.obs["disease"] == "Candidemia"].copy()
    combined = whole_data.copy()

    # convert the datasets to pd.dataframes to hand to perform_pca
    mrsa_df = mrsa_split.to_df()
    # ca_df = ca_split.to_df()
    combined_df = combined.to_df()

    # get the MRSA outcome data to regress against
    y_true = whole_data.obs.loc[whole_data.obs["disease"] == "MRSA", "status"].astype(
        int
    )

    """We only have MRSA outcome data, so we have to leave out the CA data
    and truncate the combined data to MRSA data before performing regression."""

    # set up a dataframe to hold the performance data
    total_performance = pd.DataFrame(
        np.arange(1, components + 1, dtype=int), columns=pd.Index(["components"])
    )

    # perform PCA on the datasets @ full component count
    mrsa_pc, _, _ = perform_pca(mrsa_df)
    # ca_pc, _, _ = perform_pca(ca_df)
    combined_pc, _, _ = perform_pca(combined_df)

    # truncate the combined dataset to MRSA data
    mrsa_index = whole_data.obs["disease"] == "MRSA"
    combined_pc = combined_pc.loc[mrsa_index, :]

    # perform regression on the datasets in component subsets
    mrsa_performance = [
        perform_LR(mrsa_pc.iloc[:, : i + 1], y_true)[0] for i in range(components)
    ]
    # ca_performance, _, _ = [
    # perform_LR(ca_pc.iloc[:, : i + 1], y_true) for i in range(components)
    # ]
    combined_performance = [
        perform_LR(combined_pc.iloc[:, : i + 1], y_true)[0] for i in range(components)
    ]

    # add the performance data to the dataframe
    total_performance["MRSA"] = mrsa_performance
    # total_performance["CA"] = ca_performance
    total_performance["Combined"] = combined_performance

    return total_performance


def genFig():
    fig_size = (3, 3)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    performance_data = figure_setup()

    # melt the data for seaborn lineplot
    melted = performance_data.melt(
        id_vars=["components"], var_name="Dataset", value_name="Performance"
    )

    a = sns.lineplot(
        data=melted, x="components", y="Performance", hue="Dataset", ax=ax[0]
    )
    a.set_title("Regression Performance by Component Count")
    a.set_xlabel("Component Count")
    a.set_ylabel("Regression Performance (Balanced Accuracy)")

    return f
