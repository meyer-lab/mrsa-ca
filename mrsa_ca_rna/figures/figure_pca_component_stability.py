"""
This file will plot PCA component beta weights for the MRSA data across
boostrapped iterations to determine the stability of the components.
Breifly, the raw patient data is resampled, PCA is performed, and components
are standardized with linear sum assignment to match across iterations. Then,
the components are used in a logistic regression classifier to determine beta
weights, which are averaged across iterations and plotted.
"""

import anndata as ad
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR


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

    coef_list = []
    for i in range(iterations):
        # resample the data and pop-out the y values
        resampled_data = resample(combined_data)
        y_true = resampled_data.pop("status")

        # perform PCA on the combined dataset
        scores, _, _ = perform_pca(resampled_data, components=20)

        # perform the linear sum assignment to match components
        _, col_ind = linear_sum_assignment(scores.abs(), maximize=True)

        # standardize the components
        scores: pd.DataFrame = scores.iloc[:, col_ind]

        # pop in the true y values and drop rows with NaN to get just the MRSA data
        scores["status"] = y_true
        scores = scores.dropna(axis=0)

        # split the data into X and y
        y_true = scores.pop("status")

        # perform logistic regression and collect beta coefficient data
        nested_score, _, model = perform_LR(scores, y_true, splits=10)
        coef_list.append(pd.DataFrame(model.coef_, index=[i], columns=scores.columns))

    coef_data = pd.concat(coef_list).T

    # convert the results to long form
    coef_data.reset_index(drop=False, inplace=True, names=["Component"])
    coef_data = coef_data.melt(
        id_vars="Component", var_name="Iteration", value_name="Beta Weight"
    )

    return coef_data


def setup_figure():
    """
    collect the data necessary to plot
    """
    data = concat_datasets(["mrsa", "ca"], scale=True, tpm=True)
    results = component_stability_test(data, iterations=200)

    return results


def genFig():
    """
    Generate the figure for the PCA component stability test
    """

    fig_size = (12, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    results = setup_figure()
    # standard_dev = statistics["Std"].to_numpy()

    a = sns.boxplot(x="Component", y="Beta Weight", data=results, ax=ax[0])
    a.set_xlabel("Component")
    a.set_ylabel("Beta Weight")
    # a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment='right')

    return f
