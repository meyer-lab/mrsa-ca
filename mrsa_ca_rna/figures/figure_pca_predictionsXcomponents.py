"""
This figure will plot the nested cross-validation performance of logistic regression
on PCA-transformed MRSA, Combined, and CA datasets as a function of the number of
components used in the PCA transformation.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import prepare_data, prepare_mrsa_ca


def figure_setup():
    """Create a dataFrame of regression performance over component #"""
    components = 15

    combined = prepare_data(filter_threshold=-1)

    # Get the MRSA and CA data, grab persistance labels
    mrsa_split, _, combined = prepare_mrsa_ca(combined)

    # convert the datasets to pd.dataframes to hand to perform_pca
    mrsa_df = mrsa_split.to_df()
    combined_df = combined.to_df()

    # Z-score the now independent mrsa dataset
    mrsa_df.loc[:, :] = StandardScaler().fit_transform(mrsa_df.to_numpy())

    # Get status targets for regression
    y_true = mrsa_split.obs["status"].astype(int)

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
    mrsa_index = combined.obs["disease"] == "MRSA"
    combined_pc = combined_pc.loc[mrsa_index, :].copy()

    # perform regression on the datasets in component subsets
    mrsa_performance = [
        perform_LR(mrsa_pc.iloc[:, : i + 1], y_true)[0] for i in range(components)
    ]
    combined_performance = [
        perform_LR(combined_pc.iloc[:, : i + 1], y_true)[0] for i in range(components)
    ]

    # add the performance data to the dataframe
    total_performance["MRSA"] = mrsa_performance
    total_performance["Combined"] = combined_performance

    return total_performance


def genFig():
    fig_size = (4, 4)
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
    a.set_ylabel("Regression Performance (CV Balanced Accuracy)")

    return f
