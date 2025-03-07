"""
This file bootstraps the factor match score and R2X difference of the PF2 factor
matrices over 30 trials. The rank and l1 values are determined from wandb exploration.

"""

import anndata as ad
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tlviz.factor_tools import factor_match_score

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    disease_data = concat_datasets(disease_list, scale=False, tpm=True)

    # rank and l1 values determined from wandb exploration
    rank = 30
    l1 = 1.5e-5

    # scale and perform the parafac2 factorization on original data
    X = disease_data.copy()
    X.X = StandardScaler().fit_transform(X.X)
    weights_true, factors_true, _, R2X_true = perform_parafac2(
        X, condition_name="disease", rank=rank, l1=l1, gpu_id=1
    )
    # convert the factors to cp_tensors
    factors_true = (weights_true, factors_true)

    # convert to pd.DataFrame to keep obs ordered and unique
    df = disease_data.to_df()
    df.insert(0, "disease", X.obs["disease"].values)

    fms_list = []
    R2X_diff_list = []
    trials = 30
    for _i in range(trials):
        # resample the data
        df_resampled: pd.DataFrame = resample(df, replace=True)  # type: ignore

        # make a unique index
        df_resampled = df_resampled.reset_index(drop=True)

        # convert back to AnnData and scale
        df_resampled.index = df_resampled.index.astype(str)
        X_resampled = ad.AnnData(df_resampled.loc[:, df_resampled.columns != "disease"])
        X_resampled.obs["disease"] = df_resampled["disease"].to_numpy()
        X_resampled.X = StandardScaler().fit_transform(X_resampled.X)

        # perform the parafac2 factorization on the resampled data
        weights_resampled, factors_resampled, _, R2X_resampled = perform_parafac2(
            X_resampled, condition_name="disease", rank=rank, l1=l1
        )
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

    return metrics, rank, l1


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data, rank, l1 = figure_setup()

    a = sns.scatterplot(data=data, x="fms", y="R2X_diff", ax=ax[0])
    a.set_xlabel("Factor Match Score")
    a.set_ylabel("R2X Difference (%)")
    a.set_title(
        "FMS and R2X percent difference of PF2 factor matrices"
        f"over {data.shape[0]} trials.\n"
        f"Rank: {rank}, L1: {l1}"
    )

    return f
