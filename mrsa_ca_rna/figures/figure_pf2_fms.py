"""
This file bootstraps the factor match score and R2X difference of the PF2 factor
matrices over 30 trials. The rank and l1 values are determined from wandb exploration.

"""

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tlviz.factor_tools import factor_match_score

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets, resample_adata


def factorize(X_in: ad.AnnData, rank: int, random_state=None):
    X_in.X = StandardScaler().fit_transform(X_in.X)

    weights, factors, _, R2X = perform_parafac2(
        X_in, condition_name="disease", rank=rank, rnd_seed=random_state
    )
    factors = (weights, factors)

    return factors, R2X


def bootstrap_fms(X, rank, target_trials=30):
    fms_list = []
    R2X_diff_list = []

    # Track successful and failed trials
    successful_trials = 0
    failed_trials = 0
    target_trials = 50

    seeds = np.random.randint(0, 1000, size=(target_trials * 2,))

    # continue until we have enough successful trials
    while successful_trials < target_trials:
        try:
            # uniquely set the seed for the current trial
            seed = seeds[successful_trials + failed_trials]

            # factorize the original and resampled data
            factors_true, R2X_true = factorize(X, rank, random_state=seed)
            factors_resampled, R2X_resampled = factorize(
                resample_adata(X), rank, random_state=seed
            )

            # calculate the factor match score
            factor_match = factor_match_score(
                factors_true, factors_resampled, consider_weights=False, skip_mode=1
            )

            # calculate the relative difference in R2X
            R2X_diff_percent = (R2X_resampled - R2X_true) / R2X_true

            # collect the metrics
            R2X_diff_list.append(R2X_diff_percent)
            fms_list.append(factor_match)

            successful_trials += 1
            print(f"Successful trial {successful_trials}/{target_trials}")

        except Exception as e:
            failed_trials += 1
            print(f"Trial failed: {e}")
            continue

    return fms_list, R2X_diff_list, successful_trials, failed_trials


def figure_setup():
    disease_list = "all"
    disease_data = concat_datasets(
        disease_list,
        scale=False,
    )

    ranks = [10, 20]

    fms_0, R2X_0, s_0, f_0 = bootstrap_fms(
        disease_data.copy(), rank=ranks[0], target_trials=30
    )
    fms_1, R2X_1, s_1, f_1 = bootstrap_fms(
        disease_data.copy(), rank=ranks[1], target_trials=30
    )

    # combined the matrics from the 20 and 30 rank trials
    fms_list = fms_0 + fms_1
    R2X_diff_list = R2X_0 + R2X_1
    s_trials = s_0 + s_1
    f_trials = f_0 + f_1
    ranks = [ranks[0]] * len(fms_0) + [ranks[1]] * len(fms_1)

    metrics = {"rank": ranks, "fms": fms_list, "R2X_diff": R2X_diff_list}
    metrics = pd.DataFrame(
        metrics, index=range(len(fms_list)), columns=pd.Index(list(metrics.keys()))
    )

    return metrics, s_trials, f_trials


def genFig():
    fig_size = (4, 8)
    layout = {"ncols": 1, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    data, successes, failures = figure_setup()

    a = sns.scatterplot(data=data, x="fms", y="R2X_diff", hue="rank", ax=ax[0])
    a.set_xlabel("Factor Match Score")
    a.set_ylabel("R2X Difference (%)")
    a.set_title(
        "FMS and R2X percent difference of PF2 factor matrices\n"
        f"Successes: {successes}, Failures: {failures}\n"
        f"Rank: {data["rank"].unique()}"
    )

    a = sns.kdeplot(data=data, x="fms", hue="rank", clip=(0, 1), ax=ax[1])
    a.set_xlabel("Factor Match Score")
    a.set_ylabel("Density")
    a.set_xlim(0, 1)
    a.set_title("Distribution of Factor Match Scores")

    return f
