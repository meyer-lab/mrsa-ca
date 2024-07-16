"""
This figure will show regression performance of the
MRSA outcome data in the context of the full PCA'd data
containing MRSA, CA, and Healthy patients.
"""

import pandas as pd
import numpy as np
import seaborn as sns

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.figures.base import setupBase


def figure_03_setup():
    """Create a dataFrame of regression performance over component #"""

    components = 60
    performance = pd.DataFrame(np.arange(1, components + 1), columns=["components"])
    scores_train = []
    failures = []

    pca_rna, _, _ = perform_PCA()

    for i in performance["components"]:
        train_performance, failed, _ = perform_LR(pca_rna, components=i)
        scores_train.append(train_performance)
        failures.append(failed)

    performance["Training performance"] = scores_train

    return performance, failures


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data, failures = figure_03_setup()
    data_melt = pd.melt(data, ["components"])  # convert wide df to tall df for sns.

    a = sns.lineplot(
        data=data_melt, x="components", y="value", hue="variable", ax=ax[0]
    )
    for failure in failures:
        a.axvline(x=failure, linewidth=0.37, color="red")

    a.set_xlabel("# of components")
    a.set_ylabel("Score")
    a.set_title(
        "Performance of Regression of MRSA outcome:\nPCA(MRSA+CA), max_iter=10k, scaled prior to PCA, all components"
    )

    return f
