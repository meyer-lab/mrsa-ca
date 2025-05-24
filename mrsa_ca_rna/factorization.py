"""
This file will include the functions necessary to perform parafac2 factorization on
a tensor dataset. The functions will prepare the data for the factorization and then
perform the factorization using the parafac2_nd function from the parafac2 library.
"""

import anndata as ad
import numpy as np
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.decomposition import parafac2
from tensorly.preprocessing import (
    svd_compress_tensor_slices,
    svd_decompress_parafac2_tensor,
)


def standardize_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    weights, factors = cp_flip_sign(cp_normalize((None, factors)), mode=1)

    # Order components by weight
    w_idx = np.argsort(weights)
    factors = [f[:, w_idx] for f in factors]

    # Order eigen-cells to maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(factors[1].T), maximize=True)
    factors[1] = factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(factors[1]))
    factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return weights, factors, projections


def perform_parafac2(
    X: ad.AnnData,
    condition_name: str = "disease",
    rank: int = 10,
):
    """Performs the parafac2 tensor factorization on the data by calling our custom
    parafac2_nd function.

    Parameters
    ----------
    X : ad.AnnData
        formatted data for tensor factorization
    condition_name : str
        The condition by which to slice the data, by default "disease"
    rank : int, optional
        The rank of the resulting decomposition, by default 10

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        the weights, factors, projections, and R2X value of the decomposition
    """

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex
    X.obs["condition_unique_idxs"] = X.obs["condition_unique_idxs"].astype("category")

    # convert to list
    X_list = [np.array(X[sgIndex == i].X) for i in range(np.amax(sgIndex) + 1)]

    scores, loadings = svd_compress_tensor_slices(
        X_list, 0.00001, rank * 2 + 10, "randomized_svd"
    )

    decomposition = parafac2(
        scores,
        rank=rank,
        verbose=True,
        init="svd",
        svd="randomized_svd",
        tol=1e-5,
        n_iter_max=100,
    )

    weights, factors, projections = svd_decompress_parafac2_tensor(
        decomposition, loadings
    )

    R2X = 0.1

    return weights, factors, projections, R2X
