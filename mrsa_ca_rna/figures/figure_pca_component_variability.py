"""This file will plot the norms of the PCA components for the MRSA CA RNA data to
assess the variability of the components across runs with resampling."""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tlviz.factor_tools import factor_match_score
from tqdm import tqdm

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


def matrix_fms(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """
    For two matrices, compute the factor match score between the two matrices
    for each number of columns from 1 to the number of columns in the matrices.

    Basically, we are computing the factor match score between the matrices if
    just the first column is used, then the first two columns, and so on.
    """

    fms_scores = []
    for i in range(1, a.shape[1] + 1):
        f_a = [
            np.ones((a.shape[0], i)),
            np.ones((a.shape[0], i)),
            a.iloc[:, :i].to_numpy(),
        ]
        f_b = [
            np.ones((b.shape[0], i)),
            np.ones((b.shape[0], i)),
            b.iloc[:, :i].to_numpy(),
        ]
        cp_a = (None, f_a)
        cp_b = (None, f_b)
        fms_scores.append(factor_match_score(cp_a, cp_b, consider_weights=False))
    return np.array(fms_scores)


def figure_setup():
    # import and convert the data to pandas for resample
    datasets = ["mrsa", "ca"]
    mrsa_ca = concat_datasets(datasets).to_df()

    n_comp = 15
    # start with a pca decomposition of the true data
    _, loadings_true, _ = perform_pca(mrsa_ca, components=n_comp)

    # resample the data
    n_resamples = 100
    resampled_data: list[pd.DataFrame] = [
        resample(mrsa_ca, replace=True)
        for _ in range(n_resamples)  # type: ignore
    ]

    # Z-score each resampled dataset
    resampled_data = [
        pd.DataFrame(
            StandardScaler().fit_transform(data.to_numpy()),
            index=data.index,
            columns=data.columns,
        )
        for data in resampled_data
    ]

    # set up dataframes
    pc_index = pd.Index([f"{i}" for i in range(1, n_comp + 1)])
    pc_columns = pd.Index([f"Resample {i + 1}" for i in range(n_resamples)])

    pca_singular_values = pd.DataFrame(
        np.zeros((n_comp, n_resamples)), columns=pc_columns, index=pc_index
    )
    pca_diff = pd.DataFrame(
        np.zeros((n_comp, n_resamples)), columns=pc_columns, index=pc_index
    )
    pca_fms = pd.DataFrame(
        np.zeros((n_comp, n_resamples)), columns=pc_columns, index=pc_index
    )

    # perform PCA on each resampled dataset, storing the metrics of interest
    # into the dataframes
    for i, data in tqdm(
        enumerate(resampled_data),
        total=len(resampled_data),
        desc="Resampling PCA",
        leave=True,
    ):
        _, loadings, pca = perform_pca(data, components=n_comp)
        pca_singular_values.iloc[:, i] = pca.singular_values_
        pca_diff.iloc[:, i] = matrix_cosines(loadings_true.T, loadings.T)
        pca_fms.iloc[:, i] = matrix_fms(loadings_true.T, loadings.T)

    return pca_singular_values, pca_diff, pca_fms


def genFig():
    fig_size = (6, 12)
    layout = {"ncols": 1, "nrows": 3}
    ax, f, _ = setupBase(fig_size, layout)

    pca_singular_values, pca_diff, pca_fms = figure_setup()

    # Convert to long form for plotting
    pca_singular_values = pca_singular_values.reset_index(names=["Component"]).melt(
        id_vars="Component", var_name="Resample", value_name="Singular Values"
    )

    pca_diff = pca_diff.reset_index(names=["Component"]).melt(
        id_vars="Component", var_name="Resample", value_name="Cosine Distance"
    )

    pca_fms = pca_fms.reset_index(names=["Component"]).melt(
        id_vars="Component", var_name="Resample", value_name="Factor Match Score"
    )

    # singular values
    a = sns.lineplot(
        data=pca_singular_values,
        x="Component",
        y="Singular Values",
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        ax=ax[0],
    )
    a.set_xlabel("PCA Component")
    a.set_ylabel("Singular values")
    a.set_title("Average Singular values of PCA Components with 95% CI")

    # cosine distances
    a = sns.lineplot(
        data=pca_diff,
        x="Component",
        y="Cosine Distance",
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        ax=ax[1],
    )
    a.set_xlabel("PCA Component")
    a.set_ylabel("Cosine Distance")
    a.set_title("Average Cosine Distance of PCA Components with 95% CI")

    a = sns.lineplot(
        data=pca_fms,
        x="Component",
        y="Factor Match Score",
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        ax=ax[2],
    )
    a.set_xlabel("PCA Component Inclusion")
    a.set_ylabel("Factor Match Score")
    a.set_title("Average Factor Match Score of PCA Components with 95% CI")

    return f
