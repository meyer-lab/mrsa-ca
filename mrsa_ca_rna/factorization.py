"""
This file will include the functions necessary to perform parafac2 factorization on
a tensor dataset. The functions will prepare the data for the factorization and then
perform the factorization using the parafac2_nd function from the parafac2 library.
"""

import anndata as ad
import cupy as cp
import numpy as np
from parafac2 import parafac2_nd

# set the gpu id to use for the factorization
cp.cuda.Device(1).use()


# prepare the data to form a numpy list using xarray to pass to tensorly's parafac2
def prepare_data(X: ad.AnnData, expansion_dim: str = "None"):
    """Prepares the data for tensor factorization by creating a unique index for the
    condition by which we slice the tensor and pre-calculating the gene means.

    Parameters
    ----------
    X : ad.AnnData
        The data to be prepared for tensor factorization
    expansion_dim : str
        The dimension by which to expand the data

    Returns
    -------
    ad.AnnData
        Data prepared for tensor factorization via parafac2_nd
    """

    assert expansion_dim != "None", (
        "Please provide the expansion dimension for the data"
    )

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(expansion_dim), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex
    X.obs["condition_unique_idxs"] = X.obs["condition_unique_idxs"].astype("category")

    # Pre-calculate gene means
    means = np.mean(X.X, axis=0)  # type: ignore
    X.var["means"] = means

    return X


def perform_parafac2(
    X: ad.AnnData,
    condition_name: str = "disease",
    rank: int = 10,
    rnd_seed: int | None = None,
    callback=None,
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
    gpu_id : int, options: 0 or 1
        the GPU target to run the factorization on, by default 1
    rnd_seed : int, optional
        specify a random state for the factorization, by default None
    callback : func, optional
        for interior value extraction during wandb experiments, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        the weights, factors, projections, and R2X value of the decomposition
    """

    # Prepare the data for the tensor factorization
    X = prepare_data(X, expansion_dim=condition_name)

    decomposition, R2X = parafac2_nd(
        X_in=X,
        rank=rank,
        n_iter_max=100,
        tol=1e-12,
        random_state=rnd_seed,
        callback=callback,
    )
    weights = decomposition[0]
    factors = decomposition[1]
    projections = decomposition[2]

    return weights, factors, projections, R2X
