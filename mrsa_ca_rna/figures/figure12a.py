"""This file will plot the prediction performance of a logistic regression model
that predicts what disease category a patient belongs to based on """
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.import_data import (
    import_healthy,
    import_breast_cancer,
    concat_datasets,
    concat_general,
)
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.regression import perform_PC_LR

import pandas as pd
import numpy as np
import seaborn as sns

def figure12a_setup(rank=20):
    """Generate the disease factor matrix using the PARAFAC2 algorithm
    and perform logistic regression on a per-rank basis to predict disease category"""

    old_data = concat_datasets(scale=False, tpm=True)
    bc_data = import_breast_cancer(tpm=True)
    healthy_data = import_healthy(tpm=True)
    disease_data = concat_general(
        [old_data, healthy_data, bc_data], shrink=True, scale=True, tpm=True
    )

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, _ = perform_parafac2(disease_xr, rank=rank)
    disease_factors = tensor_decomp[1][0] # only the first factor matrix is needed

    # organize disease factors into a pandas dataframe
    disease_df = pd.DataFrame(
        disease_factors,
        index=disease_data.obs["disease"].unique(),
        columns=range(1, rank + 1),
        )
    
    # transpose the dataframe and reset index to perform rank-based logistic regression
    disease_df = disease_df.T.reset_index(names="Rank")

    # melt the dataframe to have 3 columns: Rank, Disease, Factor Value
    melted = disease_df.melt(
        id_vars=["Rank"],
        value_vars=["MRSA", "Candidemia", "Healthy", "BreastCancer"],
        var_name="Disease",
        value_name="Factor Value")

    # make mapping between disease and string integers, then remap the Rank column
    mapping = {"MRSA": "1", "Candidemia": "2", "Healthy": "3", "BreastCancer": "4"}
    melted["Rank"] = melted["Rank"].astype(str) + "_" + melted["Disease"].map(mapping)

    x_data = melted.loc[:, ["Factor Value"]]
    y_data = melted["Disease"].values.ravel()

    # perform logistic regression on the melted dataframe
    score, model = perform_PC_LR(x_data, y_data)

    return melted, score, model

def genFig():
    """Generate the figure comparing the prediction performance of logistic regression
    on the disease factor matrix for each rank"""

    fig_size = (8, 4)
    layout = {"ncols": 2, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    for i, rank in enumerate([20, 50]):
        melted, _, model = figure12a_setup(rank=rank)

        # plot the confusion matrix for the model overall
        y_pred = cross_val_predict(model, X=melted.loc[:, ["Factor Value"]], y=melted["Disease"], cv=5)
        
        conf_matrix = confusion_matrix(melted["Disease"], y_pred, labels=["MRSA", "Candidemia", "Healthy", "BreastCancer"])

        true_labels = ["True MRSA", "True Candidemia", "True Healthy", "True BreastCancer"]
        pred_labels = ["Predicted MRSA", "Predicted Candidemia", "Predicted Healthy", "Predicted BreastCancer"]

        a = sns.heatmap(conf_matrix, xticklabels=pred_labels, yticklabels=true_labels, annot=True, ax=ax[i])
        a.set_title(f"Using Rank {rank}")

    return f