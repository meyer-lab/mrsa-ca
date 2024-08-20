"""
Graphing a heatmap of candidemia patient RNA signals prior to scaling, after scaling,
and after weighting via regression coef output.
"""

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import seaborn as sns

from mrsa_ca_rna.regression import perform_PC_LR
from mrsa_ca_rna.regression import concat_datasets
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase


def figure04_setup():
    """Organize data for plotting"""

    # push adata to df for compatibility with previously written code
    whole_dataset = concat_datasets()
    df = whole_dataset.to_df()

    scores, _, _ = perform_PCA(df)

    # another compatibility line so I don't have to rewrite everything
    rna = df

    desired_components = pd.IndexSlice["components", scores["components"].columns]
    component_data = scores.loc[:, desired_components]

    nested_accuracy, model = perform_PC_LR(
        scores.loc["MRSA", desired_components],
        scores.loc["MRSA", ("meta", "status")],
        whole_dataset[whole_dataset.obs["disease" == "MRSA"]],
        whole_dataset[whole_dataset.obs.loc["disease" == "MRSA", "status"]],
    )
    weights = model.coef_

    weighted_rna = rna.copy()
    for pat in range(len(weighted_rna.index.get_level_values(1))):
        weighted_rna.iloc[pat, :] = weighted_rna.iloc[pat, :].values * weights
    totals = []
    for col in weighted_rna.columns:
        totals.append(weighted_rna[col].sum())
    total_df = pd.DataFrame(
        np.reshape(totals, (1, -1)), index=["Total"], columns=weighted_rna.columns
    )
    largest_3 = total_df.T.nlargest(3, total_df.index, keep="all")
    smallest_3 = total_df.T.nsmallest(3, total_df.index, keep="all")
    largest_smallest = pd.concat([largest_3.T, smallest_3.T], axis=1)

    weighted_data = pd.concat(
        [largest_smallest, weighted_rna.droplevel(0)], axis=0, join="inner"
    )

    return (component_data, weighted_data, nested_accuracy)


def genFig():
    fig_size = (8, 4)
    layout = {"ncols": 2, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    component_data, weighted_data, nested_score = figure04_setup()

    a = sns.heatmap(component_data, cmap="viridis", center=0, ax=ax[0])
    a.set_title("Component importance per patient")

    a = sns.heatmap(weighted_data, cmap="viridis", center=0, ax=ax[1])
    a.set_title(
        f"Most correlated and anticorrelated gene expressions to MRSA outcome\nScore {nested_score}"
    )

    return f
