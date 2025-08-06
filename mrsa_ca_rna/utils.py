"""This file will contain utility functions for the project.
These functions will be used throughout the project to perform various common tasks."""

import gzip
import os
import re

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
    min_pct: float = 0.25,
) -> ad.AnnData:
    """Concatenate multiple AnnData objects from different datasets into
    a single AnnData object. Perform filtering of genes based on a threshold,
    normalization of counts, log2 transformation, and z-score scaling.

    Parameters
    ----------
    ad_list : list[str], optional
        list of datasets to include, by default None = All datasets
    filter_threshold : float, optional
        CPM threshold for filtering genes, -1 to disable, by default 0.1
    min_pct : float, optional
        Minimum fraction of samples required to express gene
        above threshold, by default 0.25

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
            print(f"Importing dataset: {ad_key}")
            adata_list.append(data_dict[ad_key]())

    if not adata_list:
        raise ValueError("No valid datasets provided or found")

    # Concat all anndata objects together keeping only the vars and obs in common
    adata = ad.concat(adata_list, join="inner")

    # Filtering now optional if filter_threshold is set to -1
    if filter_threshold >= 0:
        # Filter low expression genes
        filtered_genes = gene_filter(
            adata.to_df(), threshold=filter_threshold, min_pct=min_pct
        )
        var_mask = adata.var_names.isin(filtered_genes.columns)
        adata_filtered = adata[:, var_mask].copy()

        # Print out percentage of genes removed
        num_genes_before = adata.shape[1]
        num_genes_after = adata_filtered.shape[1]
        pct_removed = 100 * (num_genes_before - num_genes_after) / num_genes_before
        print(
            f"Filtered genes: {num_genes_before - num_genes_after} "
            f"removed ({pct_removed:.2f}%)"
        )
    else:
        adata_filtered = adata.copy()

    # Preserve the raw counts in a new layer
    adata_filtered.layers["raw"] = np.asarray(adata_filtered.X).copy()

    # Normalize the filtered data
    norm_counts = normalize_counts(np.asarray(adata_filtered.X))
    adata_filtered.X = norm_counts

    return adata_filtered


def calculate_cpm(counts: np.ndarray) -> np.ndarray:
    """Calculate counts per million (CPM) for gene expression data.

    Parameters
    ----------
    counts : np.ndarray
        Gene expression counts with samples as rows and genes as columns

    Returns
    -------
    np.ndarray
        CPM values as a numpy array
    """
    # Calculate library sizes
    total_counts = np.sum(counts, axis=1, keepdims=True)

    # Avoid division by zero
    total_counts = np.maximum(total_counts, 1)

    # CPM normalization
    return (counts / total_counts * 1e6).astype(np.float64)


def gene_filter(
    genes: pd.DataFrame, threshold: float = 1.0, min_pct: float = 0.25
) -> pd.DataFrame:
    """Filter genes based on CPM threshold in a minimum percentage of samples.
    Keeps genes with CPM > threshold in at least min_pct of samples.

    Parameters
    ----------
    genes : pd.DataFrame
        DataFrame with samples as rows and genes as columns (raw counts)
    threshold : float
        CPM threshold for filtering genes | default=1.0
    min_pct : float
        Minimum fraction of samples required to express gene
        above threshold | default=0.25

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with genes passing the CPM filter
    """
    # Convert to numpy array for CPM calculation
    genes_array = genes.values

    # Calculate CPM
    cpm_array = calculate_cpm(genes_array)

    # Calculate fraction of samples above threshold for each gene
    frac_above = np.mean(cpm_array > threshold, axis=0)

    # Create mask for genes to keep
    keep_genes = frac_above >= min_pct

    # Apply filter to original DataFrame
    return genes.iloc[:, keep_genes]


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
    # Perform CPM normalization
    norm_exp = calculate_cpm(counts)

    # Log transform the data
    trans_exp = np.log2(norm_exp + 1).astype(np.float64)

    # z-score the data
    scaled_exp = StandardScaler().fit_transform(trans_exp)

    return scaled_exp.astype(np.float64)


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


def find_top_features(
    features_df: pd.DataFrame,
    threshold_fraction: float = 0.5,
    feature_name: str = "feature",
) -> pd.DataFrame:
    """Find top features in each component that exceed a threshold of the max value,
    treating positive and negative loadings separately.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features as index and components as columns
    threshold_fraction : float, default=0.5
        Fraction of max/min value to use as threshold for positive/negative loadings
    feature_name : str, default="feature"
        Name to use for the feature column in output (e.g., "gene", "disease")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - {feature_name}: Feature name
        - component: Component identifier
        - value: Loading value of feature in component
        - abs_value: Absolute value of the loading
        - direction: "positive" or "negative" loading
        - rank: Rank of feature within its component and direction (by absolute value)
    """
    # DataFrame format with feature, component, value columns
    result_data = []
    for cmp in features_df.columns:
        # Handle positive loadings
        pos_vals = features_df[cmp][features_df[cmp] > 0]
        if not pos_vals.empty:
            max_pos = pos_vals.max()
            pos_mask = features_df[cmp] >= threshold_fraction * max_pos

            # Get the features and their values
            pos_features = features_df.index[pos_mask].tolist()
            pos_values = features_df.loc[pos_mask, cmp].tolist()

            # Add to result data
            for feature, value in zip(pos_features, pos_values, strict=False):
                result_data.append(
                    {
                        feature_name: feature,
                        "component": cmp,
                        "value": value,
                        "abs_value": abs(value),
                        "direction": "positive",
                    }
                )

        # Handle negative loadings
        neg_vals = features_df[cmp][features_df[cmp] < 0]
        if not neg_vals.empty:
            min_neg = neg_vals.min()
            neg_mask = features_df[cmp] <= threshold_fraction * min_neg

            # Get the features and their values
            neg_features = features_df.index[neg_mask].tolist()
            neg_values = features_df.loc[neg_mask, cmp].tolist()

            # Add to result data
            for feature, value in zip(neg_features, neg_values, strict=False):
                result_data.append(
                    {
                        feature_name: feature,
                        "component": cmp,
                        "value": value,
                        "abs_value": abs(value),
                        "direction": "negative",
                    }
                )

    # Convert to DataFrame
    result_df = pd.DataFrame(result_data)

    # Add rank within each component and direction (by absolute value)
    if not result_df.empty:
        result_df["rank"] = result_df.groupby(["component", "direction"])[
            "abs_value"
        ].rank(ascending=False)
        result_df = result_df.sort_values(["component", "direction", "rank"])

        return result_df
    else:
        return pd.DataFrame(
            columns=[
                feature_name,
                "component",
                "value",
                "abs_value",
                "direction",
                "rank",
            ]
        )


def get_gene_mapping(gtf_path: str | None = None) -> pd.DataFrame:
    """Parse a GTF file and extract gene ID to symbol mappings.

    Parameters
    ----------
    gtf_path : str, optional
        Path to the GTF file. If None, looks for common GTF file locations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'ensembl_id' and 'gene_name'
    """

    # Check if file exists
    if not os.path.exists(gtf_path):
        print(f"GTF file not found at: {gtf_path}")
        return pd.DataFrame(columns=["ensembl_id", "gene_name"])

    gene_mappings = {}

    # Determine if file is gzipped
    open_func = gzip.open if gtf_path.endswith(".gz") else open
    mode = "rt" if gtf_path.endswith(".gz") else "r"

    try:
        with open_func(gtf_path, mode) as f:
            for line in f:
                # Skip comment lines
                if line.startswith("#"):
                    continue

                # Split the line by tabs
                fields = line.strip().split("\t")

                # We need at least 9 fields for a valid GTF line
                if len(fields) < 9:
                    continue

                # Only process gene entries
                feature_type = fields[2]
                if feature_type != "gene":
                    continue

                # Parse the attributes field (9th column)
                attributes = fields[8]

                # Extract gene_id and gene_name using regex
                gene_id_match = re.search(r'gene_id\s+"([^"]+)"', attributes)
                gene_name_match = re.search(r'gene_name\s+"([^"]+)"', attributes)

                if gene_id_match:
                    gene_id = gene_id_match.group(1)
                    # Remove version number from Ensembl ID
                    # (e.g., ENSG00000139618.2 -> ENSG00000139618)
                    gene_id_base = gene_id.split(".")[0]

                    # Use gene_name if available, otherwise use gene_id
                    gene_name = (
                        gene_name_match.group(1) if gene_name_match else gene_id_base
                    )
                    gene_mappings[gene_id_base] = gene_name

    except Exception as e:
        print(f"Error reading GTF file: {e}")
        return pd.DataFrame(columns=["ensembl_id", "gene_name"])

    # Convert to DataFrame
    if gene_mappings:
        df = pd.DataFrame(
            [
                {"ensembl_id": ensembl_id, "gene_name": gene_name}
                for ensembl_id, gene_name in gene_mappings.items()
            ]
        )
        print(f"Loaded {len(df)} gene mappings from GTF file")
        return df
    else:
        print("No gene mappings found in GTF file")
        return pd.DataFrame(columns=["ensembl_id", "gene_name"])


def map_genes(
    gene_list: list[str],
    gtf_path: str | None = None,
    from_type: str = "ensembl",
    to_type: str = "symbol",
) -> dict[str, str]:
    """Map between Ensembl IDs and gene symbols using GTF file.

    Parameters
    ----------
    gene_list : list[str]
        List of gene identifiers to map
    gtf_path : str, optional
        Path to the GTF file
    from_type : str, default="ensembl"
        Type of input identifiers ("ensembl" or "symbol")
    to_type : str, default="symbol"
        Type of output identifiers ("ensembl" or "symbol")

    Returns
    -------
    dict[str, str]
        Dictionary mapping from input to output gene identifiers
    """
    # Get the gene mapping
    gene_df = get_gene_mapping(gtf_path)

    if gene_df.empty:
        return {}

    # Create appropriate mapping dictionary
    if from_type == "ensembl" and to_type == "symbol":
        mapping_dict = dict(
            zip(gene_df["ensembl_id"], gene_df["gene_name"], strict=False)
        )
    elif from_type == "symbol" and to_type == "ensembl":
        mapping_dict = dict(
            zip(gene_df["gene_name"], gene_df["ensembl_id"], strict=False)
        )
    elif from_type == to_type:
        # Return identity mapping
        mapping_dict = {gene: gene for gene in gene_list}
    else:
        raise ValueError("Invalid from_type or to_type. Use 'ensembl' or 'symbol'")

    # Map the genes
    result = {}
    for gene in gene_list:
        # If mapping from Ensembl, strip version number for lookup
        if from_type == "ensembl":
            gene_base = gene.split(".")[0]  # Remove version number
            if gene_base in mapping_dict:
                result[gene] = mapping_dict[gene_base]  # Use original gene as key
            else:
                # Keep original if no mapping found
                result[gene] = gene
        else:
            # For symbol to ensembl mapping, use gene directly
            if gene in mapping_dict:
                result[gene] = mapping_dict[gene]
            else:
                # Keep original if no mapping found
                result[gene] = gene

    return result
