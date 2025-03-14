"""This file will plot the norms of the PCA components for the MRSA CA RNA data to
assess the variability of the components across runs with resampling."""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import spatial
from sklearn.utils import resample

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.utils import concat_datasets


def matrix_cosines(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """
    Compute the column-wsie cosine similarity between two matrices
    which are organized as dataframes.
    """
    return np.array(
        [spatial.distance.cosine(a.iloc[:, i], b.iloc[:, i]) for i in range(a.shape[1])]
    )


def figure_setup():
    # import and convert the data to pandas for resample
    mrsa_ca = concat_datasets(["mrsa", "ca"], scale=True).to_df()


    n_comp = 15
    # start with a pca decomposition of the true data
    _, loadings_true, _ = perform_pca(mrsa_ca, components=n_comp)

    # resample the data
    n_resamples = 100
    resampled_data: list[pd.DataFrame] = [
        resample(mrsa_ca, replace=True)
        for _ in range(n_resamples)  # type: ignore
    ]

    # set up dataframes
    pc_index = pd.Index([f"{i}" for i in range(1, n_comp + 1)])
    pc_columns = pd.Index([f"Resample {i+1}" for i in range(n_resamples)])

    pca_singular_values = pd.DataFrame(
        np.zeros((n_comp, n_resamples)), columns=pc_columns, index=pc_index
    )
    pca_diff = pd.DataFrame(
        np.zeros((n_comp, n_resamples)), columns=pc_columns, index=pc_index
    )

    # perform PCA on each resampled dataset, storing the metrics of interest
    # into the dataframes
    for i, data in enumerate(resampled_data):
        _, loadings, pca = perform_pca(data, components=n_comp)
        pca_singular_values.iloc[:, i] = pca.singular_values_
        pca_diff.iloc[:, i] = matrix_cosines(loadings_true.T, loadings.T)

    return pca_singular_values, pca_diff


def genFig():
    fig_size = (6, 8)
    layout = {"ncols": 1, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    pca_singular_values, pca_diff = figure_setup()

    # convert to long form for plotting

    pca_singular_values = pca_singular_values.reset_index(names=["Component"]).melt(
        id_vars="Component", var_name="Resample", value_name="Singular Values"
    )

    pca_diff = pca_diff.reset_index(names=["Component"]).melt(
        id_vars="Component", var_name="Resample", value_name="Cosine Distance"
    )

    a = sns.boxplot(
        data=pca_singular_values, x="Component", y="Singular Values", ax=ax[0]
    )
    a.set_xlabel("PCA Component")
    a.set_ylabel("Singular values")
    a.set_title("Singular values of PCA Components across resampling")

    a = sns.boxplot(data=pca_diff, x="Component", y="Cosine Distance", ax=ax[1])
    a.set_xlabel("PCA Component")
    a.set_ylabel("Cosine Distance")
    a.set_title("Cosine Distance of PCA Components across resampling")

    return f
