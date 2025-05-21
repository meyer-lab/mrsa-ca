"""This file will contain utility functions for the project.
These functions will be used throughout the project to perform various common tasks."""

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.import_data import (
    import_bc,
    import_bc_tcr,
    import_ca,
    import_covid,
    import_covid_marine,
    import_em,
    import_hbv,
    import_heme,
    import_hiv,
    import_lupus,
    import_mrsa,
    import_ra,
    import_t1dm,
    import_tb,
    import_uc,
    import_zika,
    import_sepsis,
)


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

        top_genes = meaned_expression.nlargest(top_n).index
        data_filtered = data_filtered.loc[:, top_genes]

    if isinstance(data, ad.AnnData):
        return data[:, data.var.index.isin(data_filtered.columns)].copy()
    else:
        return data_filtered


def concat_datasets(
    ad_list=None,
    diseases=None,
    filter_threshold: float = 0,
    filter_method: str = "mean",
    shrink: bool = True,
    scale: bool = True,
) -> ad.AnnData:
    """
    Concatenate any group of AnnData objects together along the genes axis.
    Truncates to shared genes and optionally filters to specific diseases.

    Parameters:
        ad_list (list of strings or "all"): datasets to concatenate | Default = "all".
            Options: "mrsa", "ca", "bc", "tb", "uc", "t1dm" or any new datasets added
        diseases (list of strings or None): specific diseases to include |
            Default = None (all diseases)
        filter_threshold (float): threshold for gene filtering
        filter_method (str): method for gene filtering. Options: "mean", "any", "total"
        shrink (bool): whether to shrink the resulting obs to only the shared obs
        scale (bool): whether to scale the data

    Returns:
        ad (AnnData): concatenated AnnData object
    """
    # Create a dictionary of all available import functions
    data_dict = {
        "mrsa": import_mrsa,
        "ca": import_ca,
        "bc": import_bc,
        "tb": import_tb,
        "uc": import_uc,
        "t1dm": import_t1dm,
        "covid": import_covid,
        "lupus": import_lupus,
        "hiv": import_hiv,
        "em": import_em,
        "zika": import_zika,
        "heme": import_heme,
        "ra": import_ra,
        "hbv": import_hbv,
        "covid_marine": import_covid_marine,
        "bc_tcr": import_bc_tcr,
        "sepsis": import_sepsis,
    }

    # If no list is provided or "all" is specified, use all available datasets
    if ad_list is None or ad_list == "all":
        ad_list = list(data_dict.keys())

    # Ensure ad_list is a list
    if isinstance(ad_list, str) and ad_list != "all":
        ad_list = [ad_list]

    # Call the data import functions and store the resulting AnnData objects
    adata_list = []

    for ad_key in ad_list:
        if ad_key not in data_dict:
            raise RuntimeError(f"Dataset '{ad_key}' not found in available datasets.")
        else:
            adata_list.append(data_dict[ad_key]())

    if not adata_list:
        raise ValueError("No valid datasets provided or found")

    # Collect the obs data from each AnnData object
    obs_list = [ad.obs for ad in adata_list]

    # Concat all anndata objects together keeping only the vars and obs in common
    whole_ad = ad.concat(adata_list, join="inner")

    # TMM normalize the data
    counts_tmm = normalize(counts=whole_ad.X, tmm_outlier=0.05)

    # Replace the X matrix with the normalized counts
    whole_ad.layers["raw"] = whole_ad.to_df().copy()
    whole_ad.X = counts_tmm

    # If shrink is False,
    # replace the resulting obs with a pd.concat of all obs data in obs_list
    if not shrink:
        whole_ad.obs = pd.concat(obs_list, axis=0, join="outer")

    # Filter by specified diseases if provided
    if diseases:
        if isinstance(diseases, str):
            diseases = [diseases]
        disease_mask = whole_ad.obs["disease"].isin(diseases)
        whole_ad = whole_ad[disease_mask.to_numpy()]

    # If filter_threshold is provided, filter out genes with low expression
    if filter_threshold:
        whole_ad = gene_filter(
            whole_ad, threshold=filter_threshold, method=filter_method
        )
        assert isinstance(whole_ad, ad.AnnData), "whole_ad must be an AnnData object"

    if scale:
        whole_ad.X = StandardScaler().fit_transform(whole_ad.X).copy()

    return whole_ad


def normalize(counts, tmm_outlier=0.05):
    """
    Normalize the count matrix using a specified method.
    "tmm": Perform trimmed mean normalization

    Args:
        counts (np.ndarray): A numpy array representing the count matrix.

    Returns:
        np.ndarray: A normalized count matrix.
    """
    if counts.shape[0] < counts.shape[1]:
        counts = counts.T
        rotated = True
    else:
        rotated = False

    # Convert to numpy array if not already
    counts_array = np.asarray(counts)

    # Perform TMM normalization
    norm_exp = tmm_norm(counts_array, tmm_outlier)

    if rotated:
        norm_exp = norm_exp.T

    return norm_exp.astype(np.float32)


def tmm_norm(exp, percentage=0.05):
    """
    Perform TMM (Trimmed Mean of M-values) normalization.

    Args:
        exp (np.ndarray): Expression matrix to normalize.
        percentage (float): Percentage of data to trim when calculating means.

    Returns:
        np.ndarray: Normalized expression matrix.
    """
    # Add 1 and log2 transform to handle zeros
    lexp = np.log2(1 + exp).astype(np.float32)

    # Calculate trimmed means for each column
    tmm = trimmed_mean(lexp, percentage)

    # Create normalization factors matrix (repeated for each sample)
    nf = np.tile(tmm, (exp.shape[0], 1))

    # Normalize by dividing log-expression by normalization factors
    temp = lexp / nf

    return temp


def trimmed_mean(matrix, percentage):
    matrix = np.array(matrix)
    trimmed_means = []
    for col in range(matrix.shape[1]):
        data = matrix[:, col].copy()
        data = data[data > 0]
        n_trim = int(len(data) * percentage)
        sorted_values = np.sort(data)
        trimmed_values = sorted_values[n_trim:-n_trim]
        trimmed_mean = np.mean(trimmed_values)
        trimmed_means.append(trimmed_mean)
    return trimmed_means


def check_sparsity(array: np.ndarray, threshold: float = 1e-4) -> float:
    """Check the sparsity of a numpy array

    Parameters:
        array (np.ndarray): the array to check
        threshold (float): the threshold for sparsity | default=1e-4

    Returns:
        sparsity (float): the sparsity of the array"""

    return float(np.mean(threshold > array))


def resample_adata(X_in: ad.AnnData, random_state=None) -> ad.AnnData:
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
    rng = np.random.default_rng(random_state)

    # make a random index with replacement for resampling
    random_index = rng.integers(0, X_in.shape[0], size=(X_in.shape[0],))

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
