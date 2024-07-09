"""
This figure will show regression performance of the
MRSA outcome data in the context of the full PCA'd data
containing MRSA, CA, and Healthy patients.
"""

import pandas as pd
import numpy as np
import seaborn as sns

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.regression import perform_mrsa_LR
from mrsa_ca_rna.figures.base import setupBase

def figure_03_setup():
    """Create a dataFrame of regression performance over component #"""

    components = 12
    performance = pd.DataFrame(np.arange(1,components+1), columns=["components"])
    scores = []

    whole_scores, _, _ = perform_PCA()
    mrsa_scores = whole_scores.loc[whole_scores["disease"] == "mrsa"]
    mrsa_data = mrsa_scores.loc[~(mrsa_scores["status"] == "Unknown")]

    for i in performance["components"]:

        i_performance, _ = perform_mrsa_LR(mrsa_data, components=i)
        scores.append(i_performance)
    
    performance["scores"] = scores

    return performance

def genFig():

    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data = figure_03_setup()

    a = sns.lineplot(data=data, x="components", y="scores", ax=ax[0])
    a.set_xlabel("# of components")
    a.set_ylabel("Accuracy")
    a.set_title("Performance of Regression of MRSA outcome data given PCA across MRSA, CA, and Healthy data")

    return f

"""debug until I look at make file"""
genFig().savefig("./mrsa_ca_rna/output/fig03_RegressionScore.png")