"""
This figure will show regression performance of the
MRSA outcome data in the context of the full PCA'd data
containing MRSA, CA, and Healthy patients.
"""

import pandas as pd
import numpy as np
import seaborn as sns

from mrsa_ca_rna.pca import perform_PCA, perform_PCA_validation
from mrsa_ca_rna.regression import perform_mrsa_LR
from mrsa_ca_rna.figures.base import setupBase

def figure_03_setup():
    """Create a dataFrame of regression performance over component #"""

    components = 60
    performance = pd.DataFrame(np.arange(1,components+1), columns=["components"])
    scores_train = []
    scores_test = []

    whole_scores, _, _ = perform_PCA()
    mrsa_scores = whole_scores.loc[whole_scores["disease"] == "mrsa"]
    mrsa_data = mrsa_scores.loc[~(mrsa_scores["status"] == "Unknown")]

    val_scores, _, _ = perform_PCA_validation()
    mrsa_val = val_scores.loc[val_scores["disease"] == "mrsa"]

    for i in performance["components"]:

        train_performance, test_performance, _ = perform_mrsa_LR(mrsa_data, mrsa_val, components=i)
        scores_train.append(train_performance)
        scores_test.append(test_performance)
    
    performance["Training performance"] = scores_train
    performance["Testing performance"] = scores_test

    return performance


def genFig():

    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data= figure_03_setup()
    data_melt = pd.melt(data, ["components"])
    
    a = sns.lineplot(data=data_melt, x="components", y="value", hue="variable", ax=ax[0])
    a.set_xlabel("# of components")
    a.set_ylabel("Accuracy")
    a.set_title("Performance of Regression of MRSA outcome data given PCA across MRSA, CA, and Healthy data")

    return f
