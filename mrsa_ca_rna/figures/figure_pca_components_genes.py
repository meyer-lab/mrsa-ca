"""
This figure shows the ROC results of the combined logistic regression model,
the top 6 components tanked by predictive ability, and their associated genes.
"""

# imports
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets, gene_converter
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR


def make_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """Make the ROC curve"""

    # convert y's to integer
    y_true = y_true.astype(int)

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba[:, 1])
    data = {"FPR": fpr, "TPR": tpr}

    auc = roc_auc_score(y_true, y_proba[:, 1])

    return data, auc


# setup figure
def figure03a_setup():
    """Collect the data required for figure03a"""
    top_n = 6

    # get the data
    datasets = ["mrsa", "ca"]
    combined_ad = concat_datasets(datasets, scale=False, tpm=True)
    y_true = combined_ad.obs.loc[combined_ad.obs["disease"] == "MRSA", "status"]

    # perform PCA on the combined data
    combined_df = combined_ad.to_df()
    patient_comps, gene_comps, _ = perform_pca(combined_df)

    # truncate the combined components to MRSA data
    mrsa_index = combined_ad.obs["disease"] == "MRSA"
    patient_comps = patient_comps.loc[mrsa_index, :]

    # perform logistic regression on the combined data
    _, y_proba, model = perform_LR(patient_comps, y_true, splits=20)

    # get the beta coefficients from the model
    weights: np.ndarray = model.coef_[0]

    # get the location of the top 6 most important components
    # (absolute value to capture both directions)
    top_weights_locs = np.absolute(weights).argsort()[-top_n:]

    # make new dataframe containing only the top 5 components
    top_comps = gene_comps.iloc[top_weights_locs].copy()

    # get the weights of the top 5 components
    top_comp_weights = dict(
        zip(top_comps.index, weights[top_weights_locs], strict=False)
    )

    # get the top 100 genes for each of the top 5 components
    top_comp_genes = {}
    for comp in top_comps.index:
        # find the top 100 genes for current component
        gene_locs = top_comps.loc[comp].abs().nlargest(100).index

        # go back to the top component dataframe and select the genes
        genes_series = top_comps.loc[comp, gene_locs]

        # create a dataframe of the top 100 genes and their weights
        top_comp_genes[comp] = pd.DataFrame(
            {"Gene": genes_series.index, "Weight": genes_series.values}
        )

        # convert EnsemblGeneID to Symbol
        top_comp_genes[comp] = gene_converter(
            top_comp_genes[comp], "EnsemblGeneID", "Symbol"
        )

        ## save the top 100 genes to a csv file
        # top_comp_genes[comp].loc[:, "Gene"].to_csv(
        # f"mrsa_ca_rna/figures/figure03a_top_genes_{comp}.csv",
        # index=False,
        # header=False
        # )

    return y_proba, y_true, top_comp_weights, top_comp_genes


def genFig():
    fig_size = (8, 12)
    layout = {"ncols": 2, "nrows": 4}
    ax, f, _ = setupBase(fig_size, layout)

    y_proba, y_true, top_comp_weights, top_comp_genes = figure03a_setup()

    data, auc = make_roc_curve(y_true, y_proba)

    a = sns.lineplot(data, x="FPR", y="TPR", ax=ax[0])
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title(
        "Classification of MRSA outcomes using 70 component\n"
        "PCA decomposition of Combined data\n"
        f"AUC: {auc:.3f}"
    )

    # plot the top 5 components and their weights
    a = sns.barplot(
        x=list(top_comp_weights.keys()), y=list(top_comp_weights.values()), ax=ax[1]
    )
    a.set_xlabel("Component")
    a.set_ylabel("Weight")
    a.set_title("Top 5 Components determining prediction accuracy")

    # plot the top N genes for each of the top_n components
    N = 10
    for i, comp in enumerate(top_comp_genes):
        a = sns.barplot(
            data=top_comp_genes[comp].loc[:N, :], x="Weight", y="Gene", ax=ax[i + 2]
        )
        a.set_xlabel("Gene")
        a.set_ylabel("Weight")
        a.set_title(f"Top 5 Genes for Component {comp}")

    return f
