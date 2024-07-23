"""
This figure will show regression performance of the
MRSA outcome data in the context of the full PCA'd data
containing MRSA, CA, and Healthy patients.
"""

import pandas as pd
import numpy as np
import seaborn as sns

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.regression import perform_PC_LR
from mrsa_ca_rna.figures.base import setupBase


def figure_03_setup(components: int = 60):
    """Create a dataFrame of regression performance over component #"""
    scores_train = []
    failures = []

    pca_rna, _, _ = perform_PCA()

    for i in range(1, components + 1):
        train_performance, failed, _ = perform_PC_LR(pca_rna, components=i)
        scores_train.append(train_performance)
        failures.append(failed)

    performance = pd.DataFrame(scores_train, columns=["Training performance"])

    return performance, failures


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    components = 60
    runs = list(range(3))
    linewidths = [1.3, 0.9, 0.5]
    colors = ["red", "blue", "green"]
    data_list = list()
    failures_list = list()

    for run in runs:
        data, failures = figure_03_setup(components=components)
        data.rename(
            columns={"Training performance": f"Training performance, run: {run+1}"},
            inplace=True,
        )
        data_list.append(data)
        failures_list.append(failures)

    data = pd.concat(data_list, axis=1)
    data.insert(0, column="components", value=np.arange(1, components + 1))

    data_melt = pd.melt(data, ["components"])  # convert wide df to tall df for sns.

    a = sns.lineplot(
        data=data_melt, x="components", y="value", hue="variable", ax=ax[0]
    )

    for failures, color, lw in zip(failures_list, colors, linewidths):
        for failure in failures:
            if failure != 0:
                a.axvline(x=failure, linewidth=lw, color=color, linestyle="--")

    a.set_xlabel("# of components")
    a.set_ylabel("Score")
    a.set_title(
        "Performance of Regression of MRSA outcome:\nPCA(MRSA+CA), max_iter=6k, scaled prior to PCA, 'saga' solver"
    )

    return f
genFig()