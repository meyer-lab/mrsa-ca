"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

"""

import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import prepare_mrsa_ca


def setup_figure() -> tuple[pd.DataFrame, float]:
    """
    Collect data for plotting the PCA scores. Prior analysis
    reveals that the first 5 components are the most important, and that
    PC2 and 3 might be the most interesting to look at.
    """

    # Prepare the MRSA and CA data
    _, _, combined = prepare_mrsa_ca()

    scores, _, _ = perform_pca(combined.to_df(), components=5)

    # subset the scores to just MRSA
    mrsa_idxs = combined[combined.obs["disease"] == "MRSA"].obs_names
    scores_mrsa = scores.loc[mrsa_idxs].copy()
    y_mrsa = combined.obs.loc[mrsa_idxs, "status"].astype(int)

    accuracy, _, model = perform_LR(scores_mrsa, y_mrsa)
    betas = model.coef_

    # weigh the whole scores by the betas
    data = pd.DataFrame(scores @ np.diag(betas.reshape(-1)), index=scores.index)

    # relabel the columns after the matrix multiplication
    data.columns = data.columns.map({0: "PC1", 1: "PC2", 2: "PC3", 3: "PC4", 4: "PC5"})
    data["disease"] = combined.obs.loc[:, "disease"]
    data["status"] = combined.obs.loc[:, "status"]

    return data, accuracy


def genFig():
    figure_size = (12, 8)
    layout = {"ncols": 3, "nrows": 2}
    ax, f, _ = setupBase(figure_size, layout)

    # bring in the rna anndata objects and push them to dataframes for perform_pca()

    data, accuracy = setup_figure()

    n_cats = len(data.loc[:, "status"].unique())
    sns.set_palette("turbo", n_cats)

    # plot PC1 and PC2
    a = sns.scatterplot(data, x="PC1", y="PC2", hue="status", style="disease", ax=ax[0])
    a.set_title(f"PC1 vs PC2 weighted by LR betas\nAccuracy: {accuracy:.2f}")
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")

    # plot PC2 and PC3
    a = sns.scatterplot(data, x="PC2", y="PC3", hue="status", style="disease", ax=ax[1])
    a.set_title(f"PC2 vs PC3 weighted by LR betas\nAccuracy: {accuracy:.2f}")
    a.set_xlabel("PC2")
    a.set_ylabel("PC3")

    # plot PC3 and PC4
    a = sns.scatterplot(data, x="PC3", y="PC4", hue="status", style="disease", ax=ax[2])
    a.set_title(f"PC3 vs PC4 weighted by LR betas\nAccuracy: {accuracy:.2f}")
    a.set_xlabel("PC3")
    a.set_ylabel("PC4")

    # plot PC2 and PC4
    a = sns.scatterplot(data, x="PC2", y="PC4", hue="status", style="disease", ax=ax[3])
    a.set_title(f"PC2 vs PC4 weighted by LR betas\nAccuracy: {accuracy:.2f}")
    a.set_xlabel("PC2")
    a.set_ylabel("PC4")

    data = data.melt(id_vars=["disease", "status"], var_name="PC", value_name="Score")

    a = sns.stripplot(data, x="PC", y="Score", hue="status", dodge=True, ax=ax[4])
    a.set_title("Stripplot of PC1-PC5 weighted by LR betas")
    a.set_xlabel("PC")
    a.set_ylabel("Score")

    return f
