"""Plots gsea analysis for multiple components in a grid"""

import anndata as ad

from .base import setupBase
from ..factorization import perform_parafac2
from ..gene_analysis import gsea_analysis_per_cmp
from ..utils import concat_datasets, gene_converter


def setup_figure():
    # import data
    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    X: ad.AnnData = concat_datasets(disease_list, filter_threshold=4, scale=True)

    # convert to gene symbols
    X = gene_converter(X, old_id="EnsemblGeneID", new_id="Symbol", method="columns")

    # perform pf2
    _, factors, _, _ = perform_parafac2(X, condition_name="disease", rank=20, l1=1e-4)
    
    # add the c factor matrix as a varm attribute
    X.varm["Pf2_C"] = factors[2]

    return X

def genFig():
    layout = {"ncols": 1, "nrows": 2}
    fig_size = (10, 12)
    ax, f, _ = setupBase(fig_size, layout)

    X = setup_figure()
    a = gsea_analysis_per_cmp(X, 1, ax=ax)

    return f