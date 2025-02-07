"""This file will contain utility functions for the project.
These functions will be used throughout the project to perform various common tasks."""

import anndata as ad
import numpy as np
import pandas as pd


# gene activity function to pair down gene matrices to important genes
def gene_activity(
    data: ad.AnnData | pd.DataFrame, top_n: int, threshold: float
) -> ad.AnnData | pd.DataFrame:
    """
    Pairs down gene matrices to the most active genes,
    either by threshold or top_n genes. Using threshold, any gene with
    expression exceeding the threshold will be kept, across samples.
    Using top_n, only the top n genes will be kept, based on average
    expression across samples. To preserve directionality, the absolute
    value of the expression is used.

    Parameters:
        data (ad.AnnData | pd.DataFrame): RNA data to filter.
            Assumes (samples x genes) format.
        threshold (float | default 0.1): minimum expression threshold to keep a gene
        top_n (int): number of top genes to keep

    Returns:
        data (ad.AnnData | pd.DataFrame): filtered RNA data"""

    assert threshold or top_n, "Must provide either a threshold or top_n value"

    # check if the data is a pd.DataFrame or an ad.AnnData
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        # filter out genes with low expression
        if threshold:
            data = data.loc[:, data.abs() > threshold]
        # keep the top n genes
        if top_n:
            data = data.loc[:, data.abs().mean().nlargest(top_n).index]
        return data

    # if the data is an AnnData object, filter the genes based on the threshold
    elif isinstance(data, ad.AnnData):
        data = data.copy()
        # filter out genes with low expression
        if threshold:
            data = data[:, np.abs(data.X) > threshold]
        # keep the top n genes
        if top_n:
            data = data[:, np.abs(data.X).mean(axis=0).argsort()[-top_n:][::-1]]
        return data
    else:
        raise ValueError("Data must be a pandas DataFrame or an AnnData object")
