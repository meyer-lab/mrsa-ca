"""
This file bootstraps the factor match score and R2X difference of the PF2 factor
matrices over 30 trials. The rank and l1 values are determined from wandb exploration.

"""

import anndata as ad
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tlviz.factor_tools import factor_match_score

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets, resample_adata


def factorize(X_in: ad.AnnData, rank: int):
    X_in.X = StandardScaler().fit_transform(X_in.X)

    X, r2x = perform_parafac2(X_in, slice_col="disease", rank=rank)

    factors = list([X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]])
    cp_factors = (X.uns["Pf2_weights"], factors)

    return cp_factors, r2x


def bootstrap_fms(X, rank, target_trials=30):
    fms_list = []
    R2X_diff_list = []

    for _ in range(target_trials):
        # factorize the original and resampled data
        factors_true, R2X_true = factorize(X, rank)
        factors_resampled, R2X_resampled = factorize(resample_adata(X), rank)

        # calculate the factor match score
        factor_match = factor_match_score(
            factors_true, factors_resampled, consider_weights=False, skip_mode=1
        )

        # calculate the relative difference in R2X
        R2X_diff_percent = (R2X_resampled - R2X_true) / R2X_true

        # collect the metrics
        R2X_diff_list.append(R2X_diff_percent)
        fms_list.append(factor_match)

    return fms_list, R2X_diff_list


def get_data(rank=5, trails=30):
    disease_data = concat_datasets()

    fms_list, r2x_list = bootstrap_fms(
        disease_data.copy(), rank=rank, target_trials=trails
    )

    metrics = {"fms": fms_list, "R2X_diff": r2x_list}
    metrics = pd.DataFrame(
        metrics, index=range(len(fms_list)), columns=pd.Index(list(metrics.keys()))
    )

    return metrics


def genFig():
    fig_size = (4, 8)
    layout = {"ncols": 1, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    trails = 10

    # Generate data for different ranks
    rank_5 = get_data(5, trails)
    rank_10 = get_data(10, trails)
    rank_50 = get_data(50, trails)
    rank_80 = get_data(80, trails)

    # Create scatter plot and KDE plot
    a = sns.scatterplot(data=rank_5, x="fms", y="R2X_diff", ax=ax[0])
    a = sns.scatterplot(data=rank_10, x="fms", y="R2X_diff", ax=ax[0])
    a = sns.scatterplot(data=rank_50, x="fms", y="R2X_diff", ax=ax[0])
    a = sns.scatterplot(data=rank_80, x="fms", y="R2X_diff", ax=ax[0])
    a.set_xlabel("Factor Match Score")
    a.set_ylabel("R2X Difference (%)")
    a.set_title("FMS and R2X percent difference of PF2 factor matrices")
    a.legend(
        title="Rank",
        labels=["5", "10", "50", "80"],
    )

    b = sns.kdeplot(data=rank_5, x="fms", clip=(0, 1), ax=ax[1])
    b = sns.kdeplot(data=rank_10, x="fms", clip=(0, 1), ax=ax[1])
    b = sns.kdeplot(data=rank_50, x="fms", clip=(0, 1), ax=ax[1])
    b = sns.kdeplot(data=rank_80, x="fms", clip=(0, 1), ax=ax[1])
    b.set_xlabel("Factor Match Score")
    b.set_ylabel("Density")
    b.set_xlim(0, 1)
    b.set_title("Distribution of Factor Match Scores")
    b.legend(
        title="Rank",
        labels=["5", "10", "50", "80"],
    )

    return f
