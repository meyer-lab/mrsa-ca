"""
This figure shows the ROC results of the combined logistic regression model,
the top 5 components ranked by predictive ability, and their associated comp_genes.
"""

# imports
from typing import TYPE_CHECKING

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets, gene_filter

if TYPE_CHECKING:
    import pandas as pd


def make_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """Make the ROC curve"""

    # convert y's to integer
    y_true = y_true.astype(int)

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba[:, 1])
    data = {"FPR": fpr, "TPR": tpr}

    auc = roc_auc_score(y_true, y_proba[:, 1])

    return data, auc


# setup figure
def figure_setup():
    top_n = 5

    # get the data
    datasets = ["mrsa", "ca"]
    diseases = ["MRSA", "Candidemia"]
    combined_ad = concat_datasets(
        datasets,
        diseases,
        scale=False,
    )
    y_true = combined_ad.obs.loc[combined_ad.obs["disease"] == "MRSA", "status"].astype(
        int
    )

    # perform PCA on the combined data
    combined_df = combined_ad.to_df()
    patient_comps, gene_comps, _ = perform_pca(combined_df)

    # truncate the combined components to MRSA data and only the first 5
    mrsa_index = combined_ad.obs["disease"] == "MRSA"
    patient_comps = patient_comps.loc[mrsa_index, :"PC10"].copy()

    gene_comps = gene_comps.loc[:"PC10", :].copy()

    # optionally print out the gene components to a csv
    # gene_comps.T.to_csv("output/pca_genes.csv")

    # perform logistic regression on the combined data
    _, y_proba, model = perform_LR(patient_comps, y_true, splits=10)

    # get the beta coefficients from the model and associate them with the components
    weights: np.ndarray = model.coef_[0]
    weights_dict = {f"PC{i + 1}": weights[i] for i in range(len(weights))}

    ## TODO: this move is too slick, refactor
    # transform into a series of dataframes with the top genes for each component
    top_genes: pd.DataFrame = gene_comps.apply(  # type: ignore
        lambda x: gene_filter(x.to_frame().T, threshold=0, method="mean", top_n=top_n),
        axis=1,
    )

    return y_proba, y_true, weights_dict, top_genes


def genFig():
    fig_size = (12, 15)
    layout = {"ncols": 3, "nrows": 5}
    ax, f, _ = setupBase(fig_size, layout)

    y_proba, y_true, weights, comp_genes = figure_setup()

    data, auc = make_roc_curve(y_true, y_proba)

    a = sns.lineplot(data, x="FPR", y="TPR", ax=ax[0])
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title(
        "Classification of MRSA outcomes using a 10 component\n"
        "PCA decomposition of Combined data\n"
        f"AUC: {auc:.3f}"
    )

    # plot the top 10 components and their weights
    a = sns.barplot(x=list(weights.keys()), y=list(weights.values()), ax=ax[1])
    a.set_xlabel("Component")
    a.set_ylabel("Weight")
    a.set_title("Beta Coefficients of the PCA Components")

    for i, comp in enumerate(comp_genes):
        a = sns.barplot(data=comp, ax=ax[i + 2])
        a.set_xlabel("Gene")
        a.set_ylabel("Weight")
        a.set_title(f"Top 5 genes for Component component {i + 1}")

    return f


if __name__ == "__main__":
    genFig()
