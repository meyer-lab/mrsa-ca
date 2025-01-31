"""This file plots the results logistic regression of the MRSA (X) PLSR data
against MRSA outcomes."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.regression import perform_LR, perform_PLSR


def figure08a_setup():
    """Organize data for plotting"""

    # bring in whole dataset then split into MRSA (X, y) and CA (Y) sets
    whole_data = concat_datasets(scale=False, tpm=True)

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

    components = 10
    scores, loadings, pls = perform_PLSR(X_data, Y_data, components)

    # make a transformed mrsa_X using CA PLSR scores,
    # then scale since scores are not unit variance
    mrsa_Xform = mrsa_X.values @ scores["Y"].values
    mrsa_Xform = pd.DataFrame(
        mrsa_Xform, index=mrsa_X.index, columns=range(1, components + 1)
    )
    mrsa_Xform.loc[:, :] = scaler.fit_transform(mrsa_Xform.values)

    datasets = {"MRSA": loadings["X"], "Xform": mrsa_Xform}

    # perform logistic regression on mrsa_loadings data,
    # with increasing components against MRSA outcomes
    accuracies: dict = {
        "MRSA (X)": [],
        "MRSA (X) transformed by CA (Y) scores": [],
    }
    for list, data in zip(accuracies, datasets, strict=False):
        for i in range(components):
            nested_accuracy = perform_LR(
                X_data=datasets[data].iloc[:, : i + 1], y_data=mrsa_y
            )[0]
            accuracies[list].append(nested_accuracy)

    return accuracies


def genFig():
    fig_size = (8, 4)
    layout = {"ncols": 2, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    accuracies = figure08a_setup()
    for i, data in enumerate(accuracies):
        a = sns.lineplot(
            x=np.arange(1, len(accuracies[data]) + 1), y=accuracies[data], ax=ax[0 + i]
        )

        a.set_title(
            f"Predicting MRSA Outcomes\nusing {data} PLSR components\n"
            "PLSR performed with: MRSA(X) CA(Y)"
        )
        a.set_xlabel(f"Components of {data} PLSR")
        a.set_ylabel("Balanced Accuracy")

    return f
