"""
This file will include parafac2 tensor factoization methods.
We will use the tensorly library to perform the factorization.

We will also introduce the xarray packages and the dataset class to
hold our data in an all-in-one format to ease with manipulation.
"""

import anndata as ad
import cupy as cp
import numpy as np
import tensorly as tl
import xarray as xr


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


def perform_parafac2(data: xr.Dataset, rank: int = 10):
    """
    Perform the parafac2 tensor factorization on passed xarray dataset data,
    with a specified rank. The data should be in the form of a dataset with
    DataArrays for each expansion label, chosen during data preparation
    in the prepare_data method.

    Parameters:
        data (xarray.Dataset): The xarray dataset of the rna data
        rank (int): The rank of the tensor factorization | default=10

    Returns:
        tuple of:
        weights (np.ndarray): The weights of the factorization
        factors (list): The list of factor matrices, ordered by slices, rows,
                        and columns w.r.t. rank (R). The unaligned dimension is
                        replaced with eigenvalues (lambda) of the factorization
                        ex. rows unaligned: (slices*rows*columns) =>
                            (slices*R), (lambda*R), (columns*R)
        projections (list): The list of projection matrices

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

    tl.set_backend("cupy")

    # perform the factorization
    (weights, factors, projections), rec_errors = tl.decomposition.parafac2(
        [cp.array(X) for X in data_list],
        rank=rank,
        n_iter_max=200,
        init="svd",
        svd="randomized_svd",
        normalize_factors=True,
        verbose=False,
        return_errors=True,
        n_iter_parafac=20,
        linesearch=True,
    )

    tl.set_backend("numpy")
    rec_errors = cp.asnumpy(cp.array(rec_errors))

    return (
        cp.asnumpy(weights),
        [cp.asnumpy(f) for f in factors],
        [cp.asnumpy(p) for p in projections],
    ), rec_errors
