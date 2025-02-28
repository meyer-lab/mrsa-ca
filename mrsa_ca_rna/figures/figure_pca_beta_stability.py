"""
This file will plot PCA component beta weights for the MRSA data across
boostrapped iterations to determine the stability of the components.
Breifly, the raw patient data is resampled, PCA is performed, and components
are standardized with linear sum assignment to match across iterations. Then,
the components are used in a logistic regression classifier to determine beta
weights, which are averaged across iterations and plotted.
"""

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample
from tqdm import tqdm

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets


def component_stability_test(data: ad.AnnData, iterations: int = 10):
    """
    Tests the stability of the PCA components across bootstrapped iterations
    of the combined dataset regressed against the MRSA status.

    Parameters:
        data (pd.DataFrame): the combined dataset
        iterations (int): the number of bootstrapped iterations

    Returns:
        beta_weights (pd.DataFrame): the beta weights across each iteration.
    """
    # check for data formatted as anndata
    if not isinstance(data, ad.AnnData):
        raise TypeError("data must be anndata")
    # check for "status" column in obs
    if "status" not in data.obs.columns:
        raise ValueError("data must have a 'status' column in obs")

    # extract the true y values and add them to the combined dataset for resampling
    combined_data = data.copy().to_df()
    y_true = data.obs.loc[data.obs["status"] != "Unknown", "status"]
    combined_data["status"] = y_true

    hperf_coef_list = []
    lperf_coef_list = []
    hperf_pats = []
    lperf_pats = []
    for i in tqdm(range(iterations)):
        # resample the data and pop-out the y values
        resampled_data = resample(combined_data)
        y_resampled = resampled_data.pop("status")

        # perform PCA on the combined dataset, perform_pca scales the data
        scores, _, _ = perform_pca(resampled_data, components=5)

        # perform the linear sum assignment to match components
        _, col_ind = linear_sum_assignment(scores.abs(), maximize=True)

        # standardize the components
        scores: pd.DataFrame = scores.iloc[:, col_ind]

        # pop in the resampled y values and drop rows with NaN to get just the MRSA data
        scores["status"] = y_resampled
        scores = scores.dropna(axis=0)

        # split the data into X and y
        y_resampled = scores.pop("status")

        # perform logistic regression and collect beta coefficient data
        nested_score, _, model = perform_LR(scores, y_resampled, splits=10)

        # collect the high performance beta coefficients
        if nested_score >= 0.7:
            hperf_coef_list.append(
                pd.DataFrame(model.coef_, index=[i], columns=scores.columns)
            )
            hperf_pats.append(scores.index.to_numpy())
        else:
            lperf_coef_list.append(
                pd.DataFrame(model.coef_, index=[i], columns=scores.columns)
            )
            lperf_pats.append(scores.index.to_numpy())

    hperf_coef_data = pd.concat(hperf_coef_list).T
    lperf_coef_data = pd.concat(lperf_coef_list).T

    h_pats = np.concatenate(hperf_pats)
    l_pats = np.concatenate(lperf_pats)

    h_pats = pd.Series(h_pats).value_counts()
    h_pats = pd.concat([h_pats, y_true], axis=1).dropna(axis=0)
    h_pats = h_pats.reset_index(drop=False, names=["patient"])
    h_pats["percent"] = 100 * h_pats["count"] / h_pats["count"].sum()

    l_pats = pd.Series(l_pats).value_counts()
    l_pats = pd.concat([l_pats, y_true], axis=1).dropna(axis=0)
    l_pats = l_pats.reset_index(drop=False, names=["patient"])
    l_pats["percent"] = 100 * l_pats["count"] / l_pats["count"].sum()

    # order the rows by the mean beta weight, then by smallest standard deviation
    hperf_coef_data["Mean"] = hperf_coef_data.mean(axis=1).abs()
    hperf_coef_data["Std"] = hperf_coef_data.std(axis=1)
    hperf_coef_data = hperf_coef_data.sort_values(
        by=["Mean", "Std"], ascending=[False, True]
    )
    hperf_coef_data.drop(columns=["Mean", "Std"], inplace=True)

    # order the rows by the mean beta weight, then by smallest standard deviation
    lperf_coef_data["Mean"] = lperf_coef_data.mean(axis=1).abs()
    lperf_coef_data["Std"] = lperf_coef_data.std(axis=1)
    lperf_coef_data = lperf_coef_data.sort_values(
        by=["Mean", "Std"], ascending=[False, True]
    )
    lperf_coef_data.drop(columns=["Mean", "Std"], inplace=True)

    return hperf_coef_data, lperf_coef_data, h_pats, l_pats


def setup_figure():
    """
    collect the data necessary to plot
    """
    # do not scale the data prior to resampling
    data = concat_datasets(["mrsa", "ca"], scale=False, tpm=True)
    results = component_stability_test(data, iterations=100)

    return results


def genFig():
    """
    Generate the figure for the PCA component stability test
    """

    fig_size = (10, 16)
    layout = {"ncols": 1, "nrows": 4}
    ax, f, _ = setupBase(fig_size, layout)

    results = setup_figure()

    h_results = results[0]
    l_results = results[1]
    h_pats = results[2]
    l_pats = results[3]

    n_high = h_results.shape[1]
    n_low = l_results.shape[1]

    # convert the results to long form
    h_results.reset_index(drop=False, inplace=True, names=["Component"])
    h_results = h_results.melt(
        id_vars="Component", var_name="Iteration", value_name="Beta Weight"
    )
    l_results.reset_index(drop=False, inplace=True, names=["Component"])
    l_results = l_results.melt(
        id_vars="Component", var_name="Iteration", value_name="Beta Weight"
    )

    a = sns.boxplot(x="Component", y="Beta Weight", data=h_results, ax=ax[0])
    a.set_xlabel("Component")
    a.set_ylabel("Beta Weight")
    a.set_title(f"High Performance Components ({n_high} > 0.7 BA)")

    a = sns.boxplot(x="Component", y="Beta Weight", data=l_results, ax=ax[1])
    a.set_xlabel("Component")
    a.set_ylabel("Beta Weight")
    a.set_title(f"Low Performance Components ({n_low} < 0.7 BA)")

    a = sns.barplot(
        data=h_pats,
        x="patient",
        y="percent",
        hue="status",
        hue_order=["1", "0"],
        palette="pastel",
        ax=ax[2],
    )
    a.set_title("High Performance Patients")
    a.set_xlabel("Patient")
    a.set_ylabel("Resample Incidences (Percent)")
    a.set_xticklabels(a.get_xticklabels(), rotation=90)

    a = sns.barplot(
        data=l_pats,
        x="patient",
        y="percent",
        hue="status",
        hue_order=["1", "0"],
        palette="pastel",
        ax=ax[3],
    )
    a.set_title("Low Performance Patients")
    a.set_xlabel("Patient")
    a.set_ylabel("Resample Incidences (Percent)")
    a.set_xticklabels(a.get_xticklabels(), rotation=90)

    return f
