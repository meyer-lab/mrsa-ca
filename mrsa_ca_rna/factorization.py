"""
This file will include parafac2 tensor factoization methods.
We will use the tensorly library to perform the factorization.

We will also introduce the xarray packages and the dataset class to
hold our data in an all-in-one format to ease with manipulation.
"""

# Main module imports
import anndata as ad
import cupy as cp
import matcouply as mcp
import numpy as np
import tensorly as tl
import xarray as xr
from pacmap import PaCMAP


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


def perform_parafac2(data: xr.Dataset, rank: int = 10, l1: float | None = None, mapping: bool = False):
    """
    Perform the parafac2 tensor factorization on passed xarray dataset data,
    with a specified rank. The data should be in the form of a dataset with
    DataArrays for each expansion label, chosen during data preparation
    in the prepare_data method.

    Parameters:
        data (xarray.Dataset): The xarray dataset of the rna data
        rank (int): The rank of the tensor factorization | default=10
        l1 (float): The L1 regularization penalty | default=None
        mapping (bool): Whether to perform mapping of the factorization | default=False

    Returns:
        tuple of:
        weights (np.ndarray): The weights of the factorization
        factors (list): The list of factor matrices, ordered by slices, rows,
                        and columns w.r.t. rank (R). The unaligned dimension is
                        replaced with eigenvalues (lambda) of the factorization
                        ex. rows unaligned: (slices*rows*columns) =>
                            (slices*R), (lambda*R), (columns*R)
        projections (list): The list of projection matrices

        if mapping, tuple of:
        mapped_p (np.ndarray): The mapped projection matrices
        mapped_wp (np.ndarray): The mapped weighted projection matrices

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
    if l1 is not None:
        out = mcp.decomposition.parafac2_aoadmm(
            matrices=data_list,
            rank=rank,
            l1_penalty=l1,
            n_iter_max=1000,
            init="svd",
            svd="truncated_svd",
            inner_n_iter_max=5,
            return_errors=True,
        )
        (weights, factors, projections), diag = out 
        rec_errors = diag.rec_errors[-1]

    else:
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

        # convert the factors and projections to numpy arrays ahead of mapping
        weights = cp.asnumpy(weights)
        factor_list = [cp.asnumpy(f) for f in factors]
        projection_list = [cp.asnumpy(p) for p in projections]

    if mapping:

        patient_projections = [x @ factor_list[1] for x in projection_list]
        patient_projections = np.concatenate(patient_projections, axis=0)

        weighted_projections = [x * weights for x in projection_list]
        patient_weighted_projections = [x @ factor_list[1] for x in weighted_projections]
        patient_weighted_projections = np.concatenate(patient_weighted_projections, axis=0)

        # define a pacmap object for us to use, then fit_transform the data
        pacmap = PaCMAP(n_components=2, n_neighbors=10)
        # fit_transform needs explicit np.arrays
        mapped_p = pacmap.fit_transform(patient_projections)
        mapped_wp = pacmap.fit_transform(patient_weighted_projections)

        # explicitly cast the mapped matrices as np.array
        mapped_p = np.array(mapped_p)
        mapped_wp = np.array(mapped_wp)

        # include code to make dataframes for the mapped matrices
        # for ease of plotting later and to add disease labels?

        return (
        weights,
        factor_list,
        projection_list,
        ), (
        mapped_p,
        mapped_wp,
        ), rec_errors

    else:
        return (
            weights,
            factor_list,
            projection_list,
        ), rec_errors


def calculate_factor_correlation(factors: list):
    """
    Calculate the correlation between the factor matrices
    to determine the similarity between the factors

    Parameters:
        factors (list): The list of factor matrices

    Returns:
        corr (np.ndarray): The correlation matrix of the factors
    """

    corr = []
    # compare every factor matrix to every other factor matrix
    for i in range(len(factors) - 1):
        corr.append(
            tl.metrics.correlation_index(factors[i], factors[i + 1], method="max_score")
        )

    return min(corr)
