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
)


def concat_datasets(
    ad_list: list[str] | None = None,
    filter_threshold: float = 0.1,
) -> ad.AnnData:
    """Concatenate multiple AnnData objects from different datasets into
    a single AnnData object. Perform filtering of genes based on a threshold,
    normalization of counts, log2 transformation, and z-score scaling.

    Parameters
    ----------
    ad_list : list[str], optional
        list of datasets to include, by default None = All datasets
    filter_threshold : float, optional
        mean gene expression cutoff, by default 0.1

    Returns
    -------
    ad.AnnData
        Concatenated and preprocessed AnnData object containing all datasets

    Raises
    ------
    RuntimeError
        If requested dataset is not found in the available datasets.
    ValueError
        If no valid datasets are provided or found.
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

    # Concat all anndata objects together keeping only the vars and obs in common
    adata = ad.concat(adata_list, join="inner")

    # Filter low expression genes
    filtered_genes = gene_filter(adata.to_df(), threshold=filter_threshold)
    adata_filtered = adata[:, filtered_genes.columns].copy()

    # RPM normalize and z-score the data
    norm_counts = normalize_counts(counts=np.asarray(adata_filtered.X))

    # Preserve the raw counts in a new layer and add the norm counts to the adata object
    adata_filtered.layers["raw"] = np.asarray(adata_filtered.X).copy()
    adata_filtered.X = norm_counts

    return adata_filtered


def gene_filter(genes: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Generate a filter for genes based on a threshold. Dataframe is expected
    to have genes as columns and samples as rows. The filter will keep only those
    genes that have a mean expression above the threshold across all samples.

    Parameters
    ----------
    genes : pd.DataFrame
        DataFrame containing gene expression data
    threshold : float
        Threshold for filtering genes | default=0.1

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with genes that have mean expression above the threshold
    """
    return genes.loc[:, genes.abs().mean(axis=0) > threshold]


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Read-depth normalization, log2 transformation, and z-score scaling of the data.

    Parameters
    ----------
    counts : np.ndarray
        gene count matrix to be normalized

    Returns
    -------
    np.ndarray
        normalized gene count matrix
    """

    # Convert to numpy array if not already
    counts_array = np.asarray(counts)

    # Perform RPM normalization
    norm_exp = rpm_norm(counts_array)

    # Log transform the data
    counts_array = np.log2(counts_array + 1).astype(np.float32)

    # z-score the data
    scaled_norm = StandardScaler().fit_transform(norm_exp)

    return scaled_norm.astype(np.float32)


def rpm_norm(exp):
    # Calculate the library size
    total_counts = np.sum(exp, axis=1, keepdims=True)

    # Avoid division by zero (should not happen due to previous filtering)
    total_counts = np.maximum(total_counts, 1)

    # RPM normalization
    rpm_normalized = exp / total_counts * 1e6

    return rpm_normalized.astype(np.float32)


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


if __name__ == "__main__":
    adata = concat_datasets()
