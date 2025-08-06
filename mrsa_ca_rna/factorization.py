"""
This file will include the functions necessary to perform parafac2 factorization on
a tensor dataset. The functions will prepare the data for the factorization and then
perform the factorization using the parafac2_nd function from the parafac2 library.
"""

import anndata as ad
import cupy as cp
import numpy as np
import tensorly as tl
from pacmap import PaCMAP
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.decomposition import parafac2


def store_pf2(
    X: ad.AnnData,
    weights: np.ndarray,
    factors: list[np.ndarray],
    projections: list[np.ndarray],
) -> ad.AnnData:
    """Store the parafac2 factors and projections in the AnnData object.
    Parameters
    ----------
    X : ad.AnnData
        The AnnData object to store the factors and projections in.
    weights : np.ndarray
        The weights from the parafac2 decomposition.
    factors : list[np.ndarray]
        The factor matrices from the parafac2 decomposition.
    projections : list[np.ndarray]
        The projection matrices from the parafac2 decomposition.
    """

    unique_idxs = X.obs["disease_unique_idxs"]

    # Store the unstructured weights
    X.uns["Pf2_weights"] = np.asarray(weights)

    # Store the factor matrices. Pf2_C lines up with genes
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = [np.asarray(f) for f in factors]

    # Set up empty projections matrix in the obsm slot to store the projections
    pf2_proj: np.ndarray = np.zeros(
        (X.shape[0], len(X.uns["Pf2_weights"])), dtype=np.float64
    )

    # Go through each unique index and store the projections
    for i, proj in enumerate(projections):
        # Get the number of samples for this disease in the original data
        mask = unique_idxs == i
        n_samples = mask.sum()

        # Take only the relevant rows from the projection matrix
        # (discarding any rows that correspond to padding)
        proj_to_store = np.asarray(proj[:n_samples, :])
        pf2_proj[mask, :] = proj_to_store

    # Store the projections in the obsm slot
    X.obsm["Pf2_projections"] = pf2_proj

    X.obsm["weighted_Pf2_projections"] = np.asarray(
        X.obsm["Pf2_projections"] @ X.uns["Pf2_B"]
    )

    return X


def standardize_pf2(
    weights: np.ndarray, factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    weights, factors = cp_flip_sign(cp_normalize((weights, factors)), mode=1)

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
    slice_col: str = "disease",
    rank: int = 10,
) -> tuple[ad.AnnData, float]:
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
    _, sgIndex = np.unique(X.obs_vector(slice_col), return_inverse=True)
    X.obs["disease_unique_idxs"] = sgIndex
    X.obs["disease_unique_idxs"] = X.obs["disease_unique_idxs"].astype("category")

    # convert to list
    X_list = [cp.array(X[sgIndex == i].X) for i in range(np.amax(sgIndex) + 1)]

    # Check if any arrays are smaller than the requested rank
    for i, arr in enumerate(X_list):
        if arr.shape[0] < rank:
            # Calculate padding needed
            padding_rows = rank - arr.shape[0]
            # Create zero padding with same number of columns
            zero_padding = cp.zeros((padding_rows, arr.shape[1]), dtype=arr.dtype)
            # Add padding to the array
            X_list[i] = cp.vstack([arr, zero_padding])
            print(f"Padded array {i} from {arr.shape[0]} to {rank} rows")

    tl.set_backend("cupy")

    pf2, errors = parafac2(
        X_list,
        rank=rank,
        verbose=True,
        init="svd",
        tol=1e-6,
        n_iter_max=1000,
        return_errors=True,
        normalize_factors=False,
    )

    # calculate R2X
    R2X = 1.0 - errors[-1] ** 2
    R2X = cp.asnumpy(R2X.get())

    tl.set_backend("numpy")

    weights = cp.asnumpy(pf2[0].get())
    factors = [cp.asnumpy(f.get()) for f in pf2[1]]
    projections = [cp.asnumpy(p.get()) for p in pf2[2]]

    # Standardize the factors and projections
    weights, factors, projections = standardize_pf2(weights, factors, projections)
    X = store_pf2(X, weights, factors, projections)

    if rank > 1:
        pcm = PaCMAP()
        X.obsm["Pf2_PaCMAP"] = np.asarray(pcm.fit_transform(X.obsm["Pf2_projections"]))
    else:
        print("Rank is 1, skipping PaCMAP projection.")
        X.obsm["Pf2_PaCMAP"] = np.zeros((X.shape[0], 2))

    return X, R2X