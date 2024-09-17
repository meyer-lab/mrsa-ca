"""This file plots the ROC curve of the MRSA (X) PLSR data against MRSA outcomes.
It will also plot the beta coefficients of the logistic regression model for each component.
Then, it will barplot the most important genes that map to the most important components."""

from mrsa_ca_rna.import_data import concat_datasets, gene_converter
from mrsa_ca_rna.regression import perform_PLSR, perform_PC_LR
from mrsa_ca_rna.figures.base import setupBase

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import pandas as pd
import numpy as np
import seaborn as sns

skf = StratifiedKFold(n_splits=10)


def figure08b_setup():
    """Organize data for plotting"""

    # bring in whole dataset then split into MRSA (X, y) and CA (Y) sets
    whole_data = concat_datasets(scale=False, tpm=True)

    # trim out RBC genes

    mrsa_X = whole_data[whole_data.obs["disease"] == "MRSA"].to_df()
    mrsa_y = whole_data.obs.loc[whole_data.obs["disease"] == "MRSA", "status"]
    ca_Y = whole_data[whole_data.obs["disease"] == "Candidemia"].to_df()

    # independently scale, using StandardScaler, the two datasets to avoid data leakage
    scaler = StandardScaler()
    mrsa_X.loc[:, :] = scaler.fit_transform(mrsa_X.values)
    ca_Y.loc[:, :] = scaler.fit_transform(ca_Y.values)

    # perform PLSR on MRSA (X) and CA (Y) data
    X_data = mrsa_X.T
    Y_data = ca_Y.T

    components = 3
    scores, loadings, pls = perform_PLSR(X_data, Y_data, components)

    mrsa_loadings = loadings["X"]
    mrsa_scores = scores["X"]

    # perform logistic regression on mrsa_loadings data
    _, model = perform_PC_LR(mrsa_loadings, mrsa_y)
    y_proba = cross_val_predict(
        model, X=mrsa_loadings, y=mrsa_y, cv=skf, method="predict_proba"
    )

    # get the beta coefficients from the model, arrange them by absolute value, then tie them back to the components
    weights: np.ndarray = model.coef_[0]
    sorted_weights = np.absolute(weights).argsort()
    sorted_components = mrsa_loadings.iloc[:, sorted_weights]
    weighted_components = dict(zip(sorted_components.columns, weights[sorted_weights]))

    # get the top 100 genes for each of the components using the mrsa_scores
    top_genes = {}
    for comp in sorted_components.columns:
        # get the location of the top 100 genes for each component by absolute value
        gene_locs = mrsa_scores.loc[:, comp].abs().nlargest(100).index
        # grab the genes and their real values
        genes_series = mrsa_scores.loc[gene_locs, comp]
        top_genes[comp] = pd.DataFrame(
            {"Gene": genes_series.index, "Weight": genes_series.values}
        )

        # convert EnsemblGeneID to Symbol, then print to csv
        top_genes[comp] = gene_converter(top_genes[comp], "EnsemblGeneID", "Symbol")
        # top_genes[comp].loc[:, "Gene"].to_csv(
        #     f"mrsa_ca_rna/figures/figure08b_top_genes_{comp}.csv",
        #     index=False,
        #     header=False,
        # )

    return y_proba, weighted_components, top_genes


def genFig():
    fig_size = (8, 12)
    layout = {"ncols": 2, "nrows": 3}
    ax, f, _ = setupBase(fig_size, layout)

    whole_data = concat_datasets(scale=False, tpm=True)
    y_true = (
        whole_data[whole_data.obs["disease"] == "MRSA"].obs["status"].values.astype(int)
    )

    y_proba, weighted_components, top_genes = figure08b_setup()

    # plot the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    a = sns.lineplot(x=fpr, y=tpr, ax=ax[0])
    a.set_title(
        f"Covariance maximization between MRSA and CA data\npredicts MRSA outcome using 3 components\n(AUC = {roc_auc:.3f})"
    )
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")

    # plot the weighted components
    a = sns.barplot(
        x=list(weighted_components.keys()),
        y=list(weighted_components.values()),
        ax=ax[1],
    )
    a.set_title("Component relevance to prediction")
    a.set_xlabel("Component")
    a.set_ylabel("Weight")

    # plot the top 10 genes for each component
    for i, comp in enumerate(top_genes):
        a = sns.barplot(
            x="Weight", y="Gene", data=top_genes[comp].loc[:10, :], ax=ax[2 + i]
        )
        a.set_title(f"Top 10 Gene Factors for Component {comp}")
        a.set_xlabel("Weight")
        a.set_ylabel("Gene")

    return f
