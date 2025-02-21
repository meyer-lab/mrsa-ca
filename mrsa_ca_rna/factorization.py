"""
This file will include parafac2 tensor factoization methods.
We will use the tensorly library to perform the factorization.

We will also introduce the xarray packages and the dataset class to
hold our data in an all-in-one format to ease with manipulation.
"""

import anndata as ad
import matcouply as mcp
import numpy as np
import xarray as xr
from matcouply.penalties import L1Penalty, NonNegativity
from pacmap import PaCMAP
from parafac2 import parafac2_nd


# prepare the data to form a numpy list using xarray to pass to tensorly's parafac2
def prepare_data(data_ad: ad.AnnData, expansion_dim: str = "None"):
    """
    Prepare data for parafac2 tensor factorization by pushing the anndata object
    into an xarray dataset. Takes an expansion dimension to split the data into
    DataArrays for each expansion label.

    Parameters:
        data_ad (anndata.AnnData): The anndata object to convert to an xarray dataset
        expansion_dim (str): The dimension to split the data
            into DataArrays | default="None"

    Returns:
        data_xr (xarray.Dataset): The xarray dataset of the rna data
    """

    assert (
        expansion_dim != "None"
    ), "Please provide the expansion dimension for the data"

    # manually form DataArrays for each expansion label
    # then combine them into a dataset, aligned by genes
    expansion_labels = data_ad.obs[expansion_dim].unique()

    data_arrays = []
    for label in expansion_labels:
        data = data_ad[data_ad.obs[expansion_dim] == label]
        samples = data.obs.index
        genes = data.var.index
        data_ar = xr.DataArray(
            data.X, coords=[("sample_" + str(label), samples), ("gene", genes)]
        )
        data_arrays.append(data_ar)
    data_xr = xr.Dataset(
        {
            label: data_ar
            for label, data_ar in zip(expansion_labels, data_arrays, strict=False)
        }
    )

    return data_xr


def new_parafac2(X: ad.AnnData, condition_name: str = "disease", rank: int = 10, l1: float = 0.0):


    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex
    X.obs["condition_unique_idxs"] = X.obs["condition_unique_idxs"].astype("category")

    # Pre-calculate gene means
    means = np.mean(X.X, axis=0)  # type: ignore
    X.var["means"] = means

    decomposition, R2X = parafac2_nd(X_in=X, rank=rank, n_iter_max=100, l1=l1)
    factors = decomposition[1]
    projections = decomposition[2]

    return factors, projections, R2X




def perform_parafac2(
    data: xr.Dataset, rank: int = 10, l1: float = 0.0, normalize: bool = False
):
    """
    Perform the parafac2 tensor factorization on passed xarray dataset data,
    with a specified rank. The data should be in the form of a dataset with
    DataArrays for each expansion label, chosen during data preparation
    in the prepare_data method.

    Parameters:
        data (xarray.Dataset): The xarray dataset of the rna data
        rank (int): The rank of the tensor factorization | default=10
        l1 (float): The L1 regularization penalty

    Returns:
        tuple of:
            weights (np.ndarray): The weights of the factorization
            factors (list): The list of factor matrices, ordered by slices, rows,
                            and columns w.r.t. rank (R). The unaligned dimension is
                            replaced with eigenvalues (lambda) of the factorization
                            ex. rows unaligned: (slices*rows*columns) =>
                                (slices*R), (lambda*R), (columns*R)
            projections (list): The list of projection matrices
        mapped_p (np.ndarray): The mapped projection matrices
        rec_errors (list): The list of reconstruction errors at each iteration
    """

    # convert the xarray dataset to a numpy list
    data_list = [data[slc].values for slc in data.data_vars]

    # pad with zeros trick where needed to make Pf2 work even with few rows
    # this is because TensorLy is using the reduced SVD, when a full SVD is needed
    for ii in range(len(data_list)):
        cur_rows = data_list[ii].shape[0]

        if cur_rows < rank:
            data_list[ii] = np.pad(data_list[ii], ((0, rank - cur_rows), (0, 0)))

    # check if L1 regularization is needed
    out = mcp.decomposition.parafac2_aoadmm(
        matrices=data_list,
        rank=rank,
        regs=[[NonNegativity()], [], [L1Penalty(reg_strength=l1)]],
        n_iter_max=500,
        init="svd",
        svd="randomized_svd",
        inner_n_iter_max=10,
        return_errors=True,
        verbose=False,
    )
    (weights, factors), diag = out
    projections = factors[1]
    rec_errors = diag.rec_errors[-1]

    # FIXME: Right now the projections are weighted projections, and the B
    # matrix is empty. We can fix this when we see whether the regularization
    # is helping.
    factors[1] = np.eye(rank, rank)

    # normalize the factors
    # FIXME: as of now, normalization skips the B matrix
    if normalize:
        AC_factors, weights = normalize_factors([factors[0], factors[2]])
        factors[0], factors[2] = AC_factors

    # FIXME: reimplement when we have B matrix and projection sorted
    # define a pacmap object for us to use, then fit_transform the data
    pacmap = PaCMAP(n_components=2, n_neighbors=10)
    mapped_p = pacmap.fit_transform(np.concatenate(projections, axis=0))

    # include code to make dataframes for the mapped matrices
    # for ease of plotting later and to add disease labels?

    return (
        (weights, factors, projections),
        mapped_p,
        rec_errors,
    )


def normalize_factors(factors: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Normalize the factor matrices of a tensor factorization

    Parameters:
        factors (list): list of factor matrices

    Returns:
        factors (list): list of normalized factor matrices
        weights (np.ndarray): the weights of the factorization"""

    weights = np.ones(factors[0].shape[1])

    for i, factor in enumerate(factors):
        scales = np.linalg.norm(factor, axis=0)
        scales_non_zero = np.where(scales == 0, np.ones(scales.shape), scales)
        weights *= scales
        factors[i] = factor / scales_non_zero

    return factors, weights
