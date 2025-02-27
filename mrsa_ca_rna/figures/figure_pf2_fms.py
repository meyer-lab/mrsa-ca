"""
This file explores the stability of the parafac2 factor matrices with changing ranks
and L1 strengths.

TODO: Add a loop to test different ranks and L1 strengths.
    Add genFig() to plot FMS vs. rank and L1 strength.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tlviz.factor_tools import factor_match_score

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    """Collect and organize data for plotting. This function will load the data and
    perform resampling, then pf2 factorization with different ranks and L1 strengths.

    TODO: Make a loop to test different ranks and L1 strengths.
    """

    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    disease_data = concat_datasets(disease_list, scale=False, tpm=True)

    """Using values from wandb exploration, we will test the stability
    of the factor matrices
    The rank and l1 strength are a combination of good fms score >.70
    and low error <.4"""
    rank = 30
    l1 = 2e-5

    fms_list = []
    R2X_diff_list = []
    trials = 30
    for _i in range(trials):
        # store the original data and resampled data
        X = disease_data.copy()
        X.X = StandardScaler().fit_transform(X.X)

        X_resampled = disease_data.copy()
        X_resampled = X_resampled[
            np.random.choice(X_resampled.shape[0], X_resampled.shape[0], replace=True),
            :,
        ]
        X_resampled.X = StandardScaler().fit_transform(X_resampled.X)

        weights_true, factors_true, _, R2X_true = perform_parafac2(
            X,
            condition_name="disease",
            rank=rank,
            l1=l1,
        )

        # perform the parafac2 factorization on the resampled data
        weights_resampled, factors_resampled, _, R2X_resampled = perform_parafac2(
            X_resampled, condition_name="disease", rank=rank, l1=l1
        )

        # convert the factors to cp_tensors
        factors_true = (weights_true, factors_true)
        factors_resampled = (weights_resampled, factors_resampled)

        # calculate the factor match score
        factor_match = factor_match_score(
            factors_true, factors_resampled, consider_weights=False, skip_mode=1
        )
        R2X_diff_percent = (R2X_resampled - R2X_true) / R2X_true
        R2X_diff_list.append(R2X_diff_percent)
        fms_list.append(factor_match)

    metrics = {"fms": fms_list, "R2X_diff": R2X_diff_list}
    metrics = pd.DataFrame(
        metrics, index=range(trials), columns=pd.Index(["fms", "R2X_diff"])
    )

    return metrics


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data = figure_setup()

    a = sns.scatterplot(data=data, x="fms", y="R2X_diff", ax=ax[0])
    a.set_xlabel("Factor Match Score")
    a.set_ylabel("R2X Difference (%)")
    a.set_title(
        "FMS and R2X percent difference of PF2 factor matrices"
        "over 30 trials"
        )

    return f
