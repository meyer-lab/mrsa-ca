"""
This figure shows the ROC results of the combined logistic regression model,
the top 5 components ranked by predictive ability, and their associated comp_genes.
"""

# imports

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import prepare_data, prepare_mrsa_ca


def make_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """Make the ROC curve"""

    # convert y's to integer
    y_true = y_true.astype(int)

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba[:, 1])
    data = {"FPR": fpr, "TPR": tpr}

    auc = roc_auc_score(y_true, y_proba[:, 1])

    return data, auc


def figure_setup():
    top_n = 5

    # Get the MRSA and CA data, grab persistance labels
    combined_ad = prepare_data(filter_threshold=-1)
    _, _, combined_ad = prepare_mrsa_ca(combined_ad)

    # Get the labels for MRSA samples
    y_true = combined_ad.obs.loc[combined_ad.obs["disease"] == "MRSA", "status"].astype(
        int
    )

    # Perform PCA on the combined data
    combined_df = combined_ad.to_df()
    patient_comps, gene_comps, _ = perform_pca(combined_df)

    # Can only use MRSA data for regression, so truncate the combined PC data
    mrsa_index = combined_ad.obs["disease"] == "MRSA"
    patient_comps = patient_comps.loc[mrsa_index, :"PC7"].copy()
    gene_comps = gene_comps.loc[:"PC7", :].copy()

    # Perform logistic regression on the PCA components
    _, y_proba, model = perform_LR(patient_comps, y_true, splits=10)

    # get the beta coefficients from the model and associate them with the components
    weights: np.ndarray = model.coef_[0]
    weights_dict = {f"PC{i + 1}": weights[i] for i in range(len(weights))}

    # Get top genes for each component individually
    top_genes_per_component = {}
    for pc in gene_comps.index:
        # Find top genes for this component based on absolute contribution
        component_top_genes = (
            gene_comps.loc[pc].abs().sort_values(ascending=False).head(top_n)
        )
        top_genes_per_component[pc] = component_top_genes.index.tolist()

    # Create a unique set of all top genes across components
    all_top_genes = list(
        set([gene for genes in top_genes_per_component.values() for gene in genes])
    )

    # Filter gene_comps to only include these top genes
    gene_comps = gene_comps.loc[:, all_top_genes]

    return y_proba, y_true, weights_dict, gene_comps, top_genes_per_component


def genFig():
    fig_size = (12, 16)
    layout = {"ncols": 3, "nrows": 4}
    ax, f, _ = setupBase(fig_size, layout)

    # TODO: split up functions to not be returning so much data
    y_proba, y_true, weights, comp_genes, top_genes_per_component = figure_setup()

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

    # plot the top 10 components and their associated gene values
    for i, pc in enumerate(comp_genes.index[:10]):
        pc_top_genes = top_genes_per_component[pc]
        component_data = comp_genes.loc[pc, pc_top_genes]

        b = sns.barplot(
            x=component_data.index,
            y=component_data.values,
            order=component_data.index,
            ax=ax[i + 2],
        )
        b.set(title=f"Gene Contributions to {pc}", xlabel="Gene", ylabel="Contribution")
        # Rotate x-tick labels for better readability
        b.set_xticklabels(
            b.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

    return f
