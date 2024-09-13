"""
This file will first perform PCA on CA data, then use the trained PCA model to project MRSA data onto the same space.
Then, it will perform a logistic regression on the transformed MRSA data to predict MRSA outcome.
It will then plot the regression performance as a scatter plot of the predicted vs actual MRSA outcome.
Then, using the logistic regression model, it will barplot the most important components predicting MRSA outcome.
From there, it will barplot the most important genes that map to the most important components.
This will help tie the regression model back to the original data.
"""

# imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.regression import perform_PC_LR
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets, gene_converter

import numpy as np
import pandas as pd
import seaborn as sns

skf = StratifiedKFold(n_splits=10)


# setup figure
def figure03a_setup():
    """Collect the data required for figure03a"""
    orig_data = concat_datasets(scale=False, tpm=True)
    mrsa_data = orig_data[orig_data.obs["disease"] == "MRSA"]
    ca_data = orig_data[orig_data.obs["disease"] == "Candidemia"]

    # scale MRSA data prior to use
    mrsa_data.X = StandardScaler().fit_transform(mrsa_data.X)

    # perform PCA on CA data. Scaling is done in PCA function
    _, ca_loadings, ca_pca = perform_PCA(ca_data.to_df())

    # transform MRSA data using CA's PCA model
    mrsa_xform = ca_pca.transform(mrsa_data.to_df())

    # perform logistic regression on transformed MRSA data
    _, model = perform_PC_LR(mrsa_xform, mrsa_data.obs["status"])
    # y_proba = model.predict_proba(mrsa_xform)
    y_proba = cross_val_predict(
        model, X=mrsa_xform, y=mrsa_data.obs["status"], cv=skf, method="predict_proba"
    )

    # get the beta coefficients from the model
    weights: np.ndarray = model.coef_[0]

    # get the location of the top 5 most important components (absolute value to capture both directions)
    top_weights_locs = np.absolute(weights).argsort()[-5:]

    # get the top 5 components from the loadings matrix using the locations
    top_comps: pd.DataFrame = ca_loadings.iloc[top_weights_locs]

    # get the weights of the top 5 components
    top_comp_weights = dict(zip(top_comps.index, weights[top_weights_locs]))

    # get the top 100 genes for each of the top 5 components
    top_comp_genes = {}
    for comp in top_comps.index:
        gene_locs = top_comps.loc[comp].abs().nlargest(100).index
        genes_series = top_comps.loc[comp, gene_locs]
        top_comp_genes[comp] = pd.DataFrame(
            {"Gene": genes_series.index, "Weight": genes_series.values}
        )

        # convert EnsemblGeneID to Symbol, then print to csv
        top_comp_genes[comp] = gene_converter(
            top_comp_genes[comp], "EnsemblGeneID", "Symbol"
        )
        # top_comp_genes[comp].loc[:, "Gene"].to_csv(f"mrsa_ca_rna/figures/figure03a_top_genes_{comp}.csv", index=False, header=False)

    return y_proba, top_comp_weights, top_comp_genes


def genFig():
    fig_size = (8, 16)
    layout = {"ncols": 2, "nrows": 4}
    ax, f, _ = setupBase(fig_size, layout)

    whole_data = concat_datasets(scale=False, tpm=True)
    mrsa_data = whole_data[whole_data.obs["disease"] == "MRSA"]
    y_true = mrsa_data.obs["status"].values.astype(int)

    y_proba, top_comp_weights, top_comp_genes = figure03a_setup()

    # plot the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    data = {"FPR": fpr, "TPR": tpr}
    a = sns.lineplot(data, x="FPR", y="TPR", ax=ax[0])
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title(f"ROC Curve\nAUC: {roc_auc_score(y_true, y_proba[:, 1])}")

    # plot the top 5 components and their weights
    a = sns.barplot(
        x=list(top_comp_weights.keys()), y=list(top_comp_weights.values()), ax=ax[1]
    )
    a.set_xlabel("Component")
    a.set_ylabel("Weight")
    a.set_title("Top 5 Components and Weights")

    # plot the top 5 genes for each of the top 5 components
    for i, comp in enumerate(top_comp_genes):
        a = sns.barplot(
            data=top_comp_genes[comp].loc[:5, :], x="Gene", y="Weight", ax=ax[i + 2]
        )
        a.set_xlabel("Gene")
        a.set_ylabel("Weight")
        a.set_title(f"Top 5 Genes for Component {comp}")

    return f
