"""
This file will include parafac2 tensor factoization methods.
We will use the tensorly library to perform the factorization.

We will also introduce the xarray packages and the dataset class to
hold our data in an all-in-one format to ease with manipulation.
"""

import os

import anndata as ad
import numpy as np
from parafac2 import parafac2_nd


# prepare the data to form a numpy list using xarray to pass to tensorly's parafac2
def prepare_data(X: ad.AnnData, expansion_dim: str = "None"):
    """
    Prepare data for parafac2_nd tensor factorization by creating a new index
    based on the expansion dimension and calculating the gene means.

    Parameters:
        data_ad (anndata.AnnData): The anndata object to convert to an xarray dataset
        expansion_dim (str): The dimension to index on. Default is "None"

    Returns:
        X (ad.AnnData): The anndata object with the new index and gene means
    """

    assert (
        expansion_dim != "None"
    ), "Please provide the expansion dimension for the data"

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
    l1: float = 0.0,
    gpu_id: int = 0,
    rnd_seed: int = None,
    callback=None,
):
    """
    Perform the parafac2 tensor factorization on the data.

    Parameters:
    X (ad.AnnData): The anndata object to perform the factorization on
    condition_name (str): The name of the condition to expand the data on
    rank (int): The rank of the factorization
    l1 (float): The L1 regularization parameter

    Returns:
    factors (list): The factor matrices of the decomposition
    projections (list): The projections of the data
    R2X (float): The R2X value of the decomposition
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Prepare the data for the tensor factorization
    X = prepare_data(X, expansion_dim=condition_name)

    decomposition, R2X = parafac2_nd(
        X_in=X,
        rank=rank,
        n_iter_max=100,
        l1=l1,
        random_state=rnd_seed,
        callback=callback,
    )
    weights = decomposition[0]
    factors = decomposition[1]
    projections = decomposition[2]

    return weights, factors, projections, R2X
