"""Regressing against time"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from mrsa_ca_rna.import_data import extract_time_data
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.regression import perform_linear_regression, perform_elastic_regression
from mrsa_ca_rna.figures.base import setupBase


def figure05_setup(components: int = 60):
    # for compatibility, I'm just going to remake the df from the adata object
    scores, _, _ = perform_PCA()
    time_adata = extract_time_data(scale=True, tpm=True)

    time_meta = time_adata.obs.loc[:, ["subject_id", "time"]]
    # time_meta = time_data.loc[:, ("meta", ["subject_id", "time"])]

    time_scores: pd.DataFrame = pd.concat(
        [time_meta, scores],
        axis=1,
        keys=["meta", "components"],
        join="inner",
    )

    linear_scores = []
    eNet_scores = []
    for i in range(1, components + 1):
        desired_components = pd.IndexSlice[
            "components", time_scores["components"].columns[0:i]
        ]
        linear_performance, _ = perform_linear_regression(
            time_scores.loc[:, desired_components], time_scores.loc[:, ("meta", "time")]
        )
        eNet_performance, eNet_model = perform_elastic_regression(
            time_scores.loc[:, desired_components], time_scores.loc[:, ("meta", "time")]
        )

        linear_scores.append(linear_performance)
        eNet_scores.append(eNet_performance)
    linear_performance = pd.DataFrame(linear_scores, columns=["Nested Accuracy"])
    eNet_performance = pd.DataFrame(eNet_scores, columns=["Nested Accuracy"])

    time_components = time_scores.drop(columns=("meta", "subject_id"))
    ordered_time: pd.DataFrame = time_components.sort_values(by=[("meta", "time")])

    weighted_time = ordered_time.copy()
    terminal_components = pd.IndexSlice[
        "components", weighted_time["components"].columns[:components]
    ]
    for sample in weighted_time.index:
        weighted_time.loc[sample, terminal_components] = (
            ordered_time.loc[sample, terminal_components].values * eNet_model.coef_
        )

    return linear_performance, eNet_performance, weighted_time


def genFig():
    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    components = 60

    # runs = list(range(1))
    # data_list = list()
    # for run in runs:
    #     data = figure05_setup(components=components)
    #     data.rename(
    #         columns={"Nested Accuracy": f"Nested Accuracy, run: {run+1}"},
    #         inplace=True,
    #     )
    #     data_list.append(data)

    linear_scores, eNet_scores, weighted_time = figure05_setup(
        components=components
    )
    model_scores = {"linear": linear_scores, "eNet": eNet_scores}

    for i, model in enumerate(model_scores):
        model_scores[model].insert(
            0, column="components", value=np.arange(1, components + 1)
        )

        a = sns.lineplot(
            data=model_scores[model], x="components", y="Nested Accuracy", ax=ax[i]
        )

        a.set_xlabel("# of components")
        a.set_ylabel("R2 Score")
        a.set_title(
            f"{model} regression on PCA data w.r.t. time\nNested cross validated"
        )

    a = sns.heatmap(
        weighted_time["components"],
        cmap="viridis",
        center=0,
        yticklabels=weighted_time[("meta", "time")],
        ax=ax[2],
    )
    a.set_title("PCs weighted by eNet time regression coefficients")

    return f
