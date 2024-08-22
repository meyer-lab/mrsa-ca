"""
This file will include parafac2 tensor factoization methods.
We will use the tensorly library to perform the factorization.

We will also introduce the xarray packages and the dataset class to
hold our data in an all-in-one format to ease with manipulation.
"""

import numpy as np
import pandas as pd
import tensorly as tl
import xarray as xr
import anndata as ad

from mrsa_ca_rna.import_data import concat_datasets, extract_time_data


# prepare the data to form a numpy list using xarray to pass to tensorly's parafac2
def prepare_data(data_ad: ad.AnnData = None, expansion_dim: str = None):
    """
    Prepare data for parafac2 tensor factorization by pushing the anndata object
    into an xarray dataset. Takes an expansion dimension to split the data into
    DataArrays for each expansion label.

    Parameters:
        data_ad (anndata.AnnData): The anndata object to convert to an xarray dataset | default=None
        expansion_dim (str): The dimension to split the data into DataArrays | default=None

    Returns:
        data_xr (xarray.Dataset): The xarray dataset of the rna data
    """

    if data_ad is None:
        data_ad = concat_datasets(scaled=True, tpm=True)
        expansion_dim = "disease"
    assert (
        expansion_dim is not None
    ), "Please provide the expansion dimension for the data"

    """Something is wrong with the .from_dataframe method in xarray. It takes too long to convert"""
    # make a multiindex dataframe of the rna data with the disease as level 0 and the sample as level 1
    # mrsa_rna = data_ad[data_ad.obs["disease"]=="MRSA"].to_df()
    # ca_rna = data_ad[data_ad.obs["disease"]=="Candidemia"].to_df()

    # rna_df = pd.concat([mrsa_rna, ca_rna], keys=["MRSA", "Candidemia"], axis=0)
    # rna_df.index.names = ["disease", "sample"]

    # rna_xr = xr.Dataset.from_dataframe(rna_df)

    # manually form DataArrays for each expansion label and then combine them into a dataset, aligned by genes
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
        {label: data_ar for label, data_ar in zip(expansion_labels, data_arrays)}
    )

    # # individually make DataArrays for each disease and then combine them into a dataset, aligned by genes
    # mrsa_data = data_ad[data_ad.obs["disease"]=="MRSA"]
    # samples = mrsa_data.obs.index
    # genes = mrsa_data.var.index
    # mrsa_ar = xr.DataArray(mrsa_data.X, coords=[("sample_mrsa", samples), ("gene", genes)])

    # ca_data = data_ad[data_ad.obs["disease"]=="Candidemia"]
    # samples = ca_data.obs.index
    # # genes = ca_data.var.index
    # ca_ar = xr.DataArray(ca_data.X, coords=[("sample_ca", samples), ("gene", genes)])

    # rna_xr = xr.Dataset({"MRSA": mrsa_ar, "Candidemia": ca_ar})

    return data_xr


def perform_parafac2(data: xr.Dataset, rank: int = 10):
    """
    Perform the parafac2 tensor factorization on passed xarray dataset data, with a specified rank.
    The data should be in the form of a dataset with DataArrays for each expansion label, chosen
    during data preparation in the prepare_data method.

    Parameters:
        data (xarray.Dataset): The xarray dataset of the rna data
        rank (int): The rank of the tensor factorization | default=10

    Returns:
        weights (np.ndarray): The weights of the factorization
        factors (list): The list of factor matrices, ordered by slices, rows, and columns w.r.t. rank (R)
                        The unaligned dimension is replaced with eigenvalues (lambda) of the factorization
                        ex. rows unaligned: (slices*rows*columns) => (slices*R), (lambda*R), (columns*R)
        projection_matrices (list): The list of projection matrices
    """

    # convert the xarray dataset to a numpy list
    data_list = []
    for data_var in data.data_vars:
        data_list.append(data[data_var].values)

    # data_np = [data["MRSA"].values, data["Candidemia"].values]

    # perform the factorization
    (weights, factors, projection_matrices), rec_errors = tl.decomposition.parafac2(
        data_list, rank=rank, n_iter_max=100, verbose=True, return_errors=True
    )

    return (weights, factors, projection_matrices), rec_errors
