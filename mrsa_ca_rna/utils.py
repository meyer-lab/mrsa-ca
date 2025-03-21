"""This file will contain utility functions for the project.
These functions will be used throughout the project to perform various common tasks."""

from copy import deepcopy
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.import_data import (
    concat_ca,
    import_breast_cancer,
    import_covid,
    import_healthy,
    import_human_annot,
    import_mrsa_rna,
)


def gene_converter(
    data, old_id: str, new_id: str, method: str = "values"
) -> pd.DataFrame | ad.AnnData:
    """Converts gene ids from one type to another in a dataframe

    Parameters:
        dataframe (pd.DataFrame): dataframe containing gene ids to convert
        old_id (str): column name of the current gene id
            Options: "Symbol", "EnsemblGeneID"
        new_id (str): column name of the desired gene id
            Options: "Symbol", "EnsemblGeneID"
        method (str): method to use for conversion
            Options: "values", "index", "columns" | Default = "values"

    Returns:
        dataframe (pd.DataFrame) or adata (ad.AnnData): data with gene ids converted"""

    human_annot = import_human_annot()
    gene_conversion = dict(zip(human_annot[old_id], human_annot[new_id], strict=False))

    # first check if the data is a pd.DataFrame or an ad.AnnData,
    # then convert the gene ids based on the method
    if isinstance(data, pd.DataFrame):
        dataframe: pd.DataFrame = data.copy()
        if method == "values":
            dataframe = dataframe.replace(gene_conversion)
        elif method == "index":
            dataframe.index = dataframe.index.map(gene_conversion)
        elif method == "columns":
            dataframe.columns = dataframe.columns.map(gene_conversion)
        return dataframe
    # if the data is an AnnData object, convert the gene ids based on the method
    elif isinstance(data, ad.AnnData):
        adata: ad.AnnData = data.copy()
        assert method != "values", "Cannot convert values in AnnData object"
        if method == "index":
            adata.obs.index = adata.obs.index.map(gene_conversion)
        elif method == "columns":
            adata.var.index = adata.var.index.map(gene_conversion)
        return adata
    else:
        raise ValueError("Data must be a pandas DataFrame or an AnnData object")


# gene activity function to pair down gene matrices to important genes
def gene_filter(
    data: ad.AnnData | pd.DataFrame,
    threshold: float,
    method: str,
    top_n: int = 0,
) -> ad.AnnData | pd.DataFrame:
    """
    Pairs down gene matrices to the most active genes,
    either by threshold or top_n genes. Using threshold,
    the method can be "mean", "any", or "total".
    Mean: mean expression across all samples exceeds threshold
    Any: expression in any sample exceeds threshold
    Total: sum of expression across all samples exceeds threshold

    Parameters:
        data (ad.AnnData | pd.DataFrame): RNA data to filter.
            Assumes (samples x genes) format.
        top_n (int): number of top genes to keep
        threshold (float): minimum expression threshold to keep a gene
        method (str): method to use for filtering genes


    Returns:
        data (ad.AnnData | pd.DataFrame): filtered RNA data"""

    # check that either top_n or threshold is provided
    assert threshold or top_n, "Must provide either a threshold or top_n value"

    # convert to dataframe if AnnData because numpy cannot column-wise filter
    if isinstance(data, ad.AnnData):
        data_to_filter = data.to_df().copy()
    else:
        data_to_filter = data.copy()

    if threshold:
        # if method is any:
        # we keep genes that exceed the threshold in any sample
        if method == "any":
            # filter out genes with low expression across all samples
            data_filtered = data_to_filter.loc[
                :, (data_to_filter.abs() > threshold).any()
            ]
        # if method is mean:
        # we keep genes that exceed the threshold in the mean of all samples
        elif method == "mean":
            # filter out genes with low expression across all samples
            data_filtered = data_to_filter.loc[
                :, data_to_filter.abs().mean() > threshold
            ]
        # if method is total:
        # we keep genes that exceed the threshold as a sum across all samples
        elif method == "total":
            # filter out genes with low expression across all samples
            data_filtered = data_to_filter.loc[
                :, data_to_filter.abs().sum() > threshold
            ]
        else:
            raise ValueError(
                "Method must be 'mean', 'any', or 'total' when threshold is provided"
            )

        # print out how many genes were filtered out
        print(
            f"Filtered out {data_to_filter.shape[1] - data_filtered.shape[1]} genes "
            f"({data_filtered.shape[1] / data_to_filter.shape[1]:.2%} remaining)"
        )
    else:
        data_filtered = data_to_filter

    if top_n:
        # make sure data filtered is a dataframe so that we can nlargest
        assert isinstance(data_filtered, pd.DataFrame)

        # keep only the top_n genes
        meaned_expression = pd.Series(data_filtered.abs().mean())
        data_filtered: pd.DataFrame = data_filtered.loc[
            :, meaned_expression.nlargest(top_n).index
        ]

    if isinstance(data, ad.AnnData):
        return data[:, data.var.index.isin(data_filtered.columns)].copy()
    else:
        return data_filtered


def concat_datasets(
    ad_list=None,
    filter_threshold: float = 0,
    filter_method: str = "mean",
    shrink: bool = True,
    scale: bool = True,
    tpm: bool = True,
) -> ad.AnnData:
    """
    Concatenate any group of AnnData objects together along the genes axis.
    Truncates to shared genes and optionally expands obs to include all observations,
    fillig in missing values with NaN.

    Parameters:
        ad_list (list of strings): datasets to concatenate | Default = ["mrsa", "ca"].
            Options: "mrsa", "ca", "bc", "covid", "healthy"
        trim (tuple): threshold and method for gene filtering.
            Options: (threshold, method) | Default = (0, "mean")
        shrink (bool): whether to shrink the resulting obs to only the shared obs
        scale (bool): whether to scale the data
        tpm (bool): whether to normalize the data to TPM

    Returns:
        ad (AnnData): concatenated AnnData object
    """

    # if no list is provided, default to MRSA and CA
    if ad_list is None:
        ad_list = ["mrsa", "ca"]

    # create a dictionary of the possible data import functions
    data_dict = {
        "mrsa": import_mrsa_rna,
        "ca": concat_ca,
        "bc": import_breast_cancer,
        "covid": import_covid,
        "healthy": import_healthy,
    }

    # call the data import functions and store the resulting AnnData objects
    ad_list = [data_dict[ad]() for ad in ad_list]

    # collect the obs data from each AnnData object
    obs_list = [ad.obs for ad in ad_list]

    # concat all anndata objects together keeping only the vars and obs in common
    whole_ad = ad.concat(ad_list, join="inner")

    # if shrink is False,
    # replace the resulting obs with a pd.concat of all obs data in obs_list
    if not shrink:
        whole_ad.obs = pd.concat(obs_list, axis=0, join="outer")

    if tpm:
        desired_value = 1000000
        # I know whole_ad.X is an ndarray, but pyright doesn't
        # replace this with proper type gating to avoid the cast
        X = cast(np.ndarray, whole_ad.X)
        row_sums = X.sum(axis=1)

        scaling_factors = desired_value / row_sums

        X_normalized = X * scaling_factors[:, np.newaxis]

        whole_ad.X = X_normalized

    # if trim is True, filter out genes with low expression
    if filter_threshold:
        whole_ad = gene_filter(
            whole_ad, threshold=filter_threshold, method=filter_method
        )
        assert isinstance(whole_ad, ad.AnnData), "whole_ad must be an AnnData object"

    if scale:
        whole_ad.X = StandardScaler().fit_transform(whole_ad.X)

    return whole_ad


def check_sparsity(array: np.ndarray, threshold: float = 1e-4) -> float:
    """Check the sparsity of a numpy array

    Parameters:
        array (np.ndarray): the array to check
        threshold (float): the threshold for sparsity | default=1e-4

    Returns:
        sparsity (float): the sparsity of the array"""

    A = deepcopy(array)
    A[np.abs(A) < threshold] = 0
    sparsity = 1.0 - (np.count_nonzero(A) / A.size)
    return sparsity


def resample_adata(X_in: ad.AnnData) -> ad.AnnData:
    """Resamples AnnData with unique observation indices, with replacement.

    Parameters
    ----------
    X_in : ad.AnnData
        AnnData object to be resampled

    Returns
    -------
    ad.AnnData
        Resampled AnnData object with unique observation indices
    """

    # make a random index with replacement for resampling
    random_index = np.random.randint(0, X_in.shape[0], size=(X_in.shape[0],))

    # independently subset the data and obs with the random indices
    assert isinstance(X_in.X, np.ndarray)
    X_resampled = X_in.X[random_index]
    obs_resampled = X_in.obs.iloc[random_index].copy()

    # Create unique indices for the resampled observations
    obs_resampled.index = [f"bootstrap_{i}" for i in range(len(obs_resampled))]

    # Create a new AnnData object with the resampled data
    uns_dict = dict(X_in.uns)
    X_in_resampled = ad.AnnData(
        X=X_resampled, obs=obs_resampled, var=X_in.var.copy(), uns=uns_dict
    )

    return X_in_resampled
