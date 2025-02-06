"""This file will contain utility functions for the project.
These functions will be used throughout the project to perform various common tasks."""

import anndata as ad
import numpy as np
import pandas as pd

from mrsa_ca_rna.import_data import import_human_annot


def gene_converter(
    data, old_id: str, new_id: str, method: str = "values"
) -> pd.DataFrame | ad.AnnData:
    """Converts gene ids from one type to another in a dataframe

    Parameters:
        dataframe (pd.DataFrame): dataframe containing gene ids to convert
        old_id (str): column name of the current gene id
        new_id (str): column name of the desired gene id

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


# gene filtering function
def gene_filter(
    data: ad.AnnData, threshold: float = 0.1, rbc: bool = False
) -> ad.AnnData:
    """Filters genes based on specific criteria.
    By default, filters out genes with low expression.

    Parameters:
        data (ad.AnnData): RNA data to filter. Assumes (samples x genes) format.
        threshold (float | default 0.1): minimum expression threshold to keep a gene
        rbc (bool | default False): whether to filter out RBC related genes

    Returns:
        data (ad.AnnData): filtered RNA data
    """
    # list of RBC related genes
    rbc_genes = [
        "RN7SL1",
        "RN7SL2",
        "HBA1",
        "HBA2",
        "HBB",
        "HBQ1",
        "HBZ",
        "HBD",
        "HBG2",
        "HBE1",
        "HBG1",
        "HBM",
        "MIR3648-1",
        "MIR3648-2",
        "AC104389.6",
        "AC010507.1",
        "SLC25A37",
        "SLC4A1, NRGN",
        "SNCA",
        "BNIP3L",
        "EPB42",
        "ALAS2",
        "BPGM",
        "OSBP2",
    ]

    # check if the genes are in Ensembl format, if so, convert them to gene symbols
    if data.var.index.str.contains("ENSG").all():
        data_converted = gene_converter(
            data, "EnsemblGeneID", "Symbol", method="columns"
        )
        assert isinstance(data_converted, ad.AnnData), "Gene conversion did not result \
            in an AnnData object"
        revert = True
    else:
        data_converted = data.copy()
        revert = False

    # drop the RBC genes from the data
    if rbc:
        data_trimmed = data_converted[:, ~data_converted.var.index.isin(rbc_genes)]

    # drop genes with low expression
    else:
        data_trimmed = data_converted[
            :, np.abs(data_converted.X.mean(axis=0)) > threshold
        ]

    # if the data was converted to gene symbols, convert it back to EnsemblGeneID
    if revert:
        data_trimmed = gene_converter(
            data_trimmed, "Symbol", "EnsemblGeneID", method="columns"
        )

    return data_trimmed


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
