"""plotting time dependent patients"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from mrsa_ca_rna.import_data import extract_time_data
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.regression import perform_linear_regression
from mrsa_ca_rna.figures.base import setupBase


def figure05_setup(components: int=60):
    
    scores, _, _ = perform_PCA()
    time_meta = extract_time_data()
    time_meta = time_meta.loc[:, ("meta", ["subject_id", "time"])]

    time_scores:pd.DataFrame = pd.concat([time_meta["meta"], scores.loc["Candidemia", "components"]], axis=1, keys=["meta", "components"], join="inner")

    scores_train = []
    for i in range(1, components+1):
        desired_components = pd.IndexSlice["components", time_scores["components"].columns[0:i]]
        nested_performance, _ = perform_linear_regression(time_scores.loc[:, desired_components], time_scores.loc[:, ("meta", "time")])
        scores_train.append(nested_performance)
    performance = pd.DataFrame(scores_train, columns=["Nested Accuracy"])

    return performance

def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    components = 60
    runs = list(range(1))
    data_list = list()

    for run in runs:
        data = figure05_setup(components=components)
        data.rename(
            columns={"Nested Accuracy": f"Nested Accuracy, run: {run+1}"},
            inplace=True,
        )
        data_list.append(data)

    data = pd.concat(data_list, axis=1)
    data.insert(0, column="components", value=np.arange(1, components + 1))

    data_melt = pd.melt(data, ["components"])  # convert wide df to tall df for sns.

    a = sns.lineplot(
        data=data_melt, x="components", y="value", hue="variable", ax=ax[0]
    )

    a.set_xlabel("# of components")
    a.set_ylabel("Score")
    a.set_title(
        "Linear Regression nested cross validation\nR2 score"
    )

    return f