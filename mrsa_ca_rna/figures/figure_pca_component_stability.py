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
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR


def stability_insights(data: ad.AnnData, iterations: int = 5):
    """
    Tests the stability of the PCA components across bootstrapped iterations
    of the combined dataset regressed against the MRSA status.

    Parameters:
        data (ad.AnnData): the combined dataset
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


    combined_data = data.copy().to_df()
    y_true = data.obs.loc[data.obs["status"] != "Unknown", "status"]
    combined_data["status"] = y_true

    scores = None
    U = None
    sigma = None
    V = None
    coef_list = []
    for i in range(iterations):

        if U is not None:
            old_scores = scores
            old_U = U
            old_sigma = sigma
            old_V = V

        # resample the data and pop-out the y values
        resampled_data = resample(combined_data)
        y_resampled = resampled_data.pop("status")

        # perform PCA on the combined dataset
        scores, V_T, pca = perform_pca(resampled_data, components=10)
        V = V_T.T

        # separate the scores into U and sigma
        U: pd.DataFrame = (scores/(pca.singular_values_))
        sigma = pca.singular_values_

        # # check U and V for orthogonality across components
        # print(np.allclose(U.T @ U, np.eye(U.shape[1])))
        # print(np.allclose(V.T @ V, np.eye(V.shape[1])))

        # # perform the linear sum assignment to match components
        # _, col_ind = linear_sum_assignment(U.abs(), maximize=True)

        # # standardize the components
        # U: pd.DataFrame = U.iloc[:, col_ind]

        # calculate the mean squared error between the old and new U and sigma
        if i > 0:
            mse_scores = mean_squared_error(scores, old_scores)
            mse_U = mean_squared_error(U, old_U)
            mse_sigma = mean_squared_error(sigma, old_sigma)
            mse_V = mean_squared_error(V, old_V)
            print(f"Iteration {i}:\nScores MSE: {mse_scores}, U MSE: {mse_U}, Sigma MSE: {mse_sigma}, V MSE: {mse_V}")


        # pop in the resampled y values and drop rows with NaN to get just the MRSA data
        U["status"] = y_resampled
        U_mrsa = U.dropna(axis=0)
        y_mrsa = U_mrsa.pop("status")

        # drop the status column from U for mse calculation later
        U.drop(columns=["status"], inplace=True)

        # perform logistic regression and collect beta coefficient data
        _, _, model = perform_LR(U_mrsa, y_mrsa, splits=10)

        coef_list.append(pd.DataFrame(model.coef_, index=[i], columns=U_mrsa.columns))


    coef_list = pd.concat(coef_list).T

    # order the rows by the mean beta weight, then by smallest standard deviation
    coef_list["Mean"] = coef_list.mean(axis=1).abs()
    coef_list["Std"] = coef_list.std(axis=1)
    coef_list = coef_list.sort_values(by=["Mean", "Std"], ascending=[False, True])
    coef_list.drop(columns=["Mean", "Std"], inplace=True)

    return coef_list

def component_stability_test(data: ad.AnnData, iterations: int = 5):
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
    for i in range(iterations):
        # resample the data and pop-out the y values
        resampled_data = resample(combined_data)
        y_resampled = resampled_data.pop("status")

        # perform PCA on the combined dataset
        scores, _, _ = perform_pca(resampled_data, components=4)

        # # perform the linear sum assignment to match components
        # _, col_ind = linear_sum_assignment(scores.abs(), maximize=True)

        # # standardize the components
        # scores: pd.DataFrame = scores.iloc[:, col_ind]

        # pop in the resampled y values and drop rows with NaN to get just the MRSA data
        scores["status"] = y_resampled
        scores = scores.dropna(axis=0)

        # split the data into X and y
        y_true = scores.pop("status")

        # perform logistic regression and collect beta coefficient data
        nested_score, _, model = perform_LR(scores, y_true, splits=10)

        # collect the high performance beta coefficients
        if nested_score >= 0.7:
            hperf_coef_list.append(pd.DataFrame(model.coef_, index=[i], columns=scores.columns))
        else:
            lperf_coef_list.append(pd.DataFrame(model.coef_, index=[i], columns=scores.columns))


    hperf_coef_data = pd.concat(hperf_coef_list).T
    lperf_coef_data = pd.concat(lperf_coef_list).T

    # order the rows by the mean beta weight, then by smallest standard deviation
    hperf_coef_data["Mean"] = hperf_coef_data.mean(axis=1).abs()
    hperf_coef_data["Std"] = hperf_coef_data.std(axis=1)
    hperf_coef_data = hperf_coef_data.sort_values(by=["Mean", "Std"], ascending=[False, True])
    hperf_coef_data.drop(columns=["Mean", "Std"], inplace=True)

    # order the rows by the mean beta weight, then by smallest standard deviation
    lperf_coef_data["Mean"] = lperf_coef_data.mean(axis=1).abs()
    lperf_coef_data["Std"] = lperf_coef_data.std(axis=1)
    lperf_coef_data = lperf_coef_data.sort_values(by=["Mean", "Std"], ascending=[False, True])
    lperf_coef_data.drop(columns=["Mean", "Std"], inplace=True)

    return hperf_coef_data, lperf_coef_data


def setup_figure():
    """
    collect the data necessary to plot
    """
    data = concat_datasets(["mrsa", "ca"], scale=True, tpm=True)
    results = component_stability_test(data, iterations=100)
    # results = stability_insights(data, iterations=10)

    return results


def genFig():
    """
    Generate the figure for the PCA component stability test
    """

    fig_size = (8, 8)
    layout = {"ncols": 1, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    results = setup_figure()
    # standard_dev = statistics["Std"].to_numpy()

    # # convert the results to long form
    # results.reset_index(drop=False, inplace=True, names=["Component"])
    # results = results.melt(
    #         id_vars="Component", var_name="Iteration", value_name="Beta Weight"
    #     )
    
    # a = sns.boxplot(x="Component", y="Beta Weight", data=results, ax=ax[0])
    # a.set_xlabel("Component")
    # a.set_ylabel("Beta Weight")
    # a.set_title("Component Stability, using U (Scores/Sigma)")

    h_results = results[0]
    l_results = results[1]

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

    return f