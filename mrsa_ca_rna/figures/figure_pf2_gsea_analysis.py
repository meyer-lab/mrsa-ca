"""Plots gsea analysis for multiple components in a grid"""

import anndata as ad
import pandas as pd

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.gene_analysis import gsea_analysis_per_cmp
from mrsa_ca_rna.utils import concat_datasets, gene_converter


def setup_figure():
    # import data
    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    X: ad.AnnData = concat_datasets(disease_list, filter_threshold=4, scale=True)

    # convert to gene symbols
    X = gene_converter(X, old_id="EnsemblGeneID", new_id="Symbol", method="columns")

    # perform pf2
    # _, factors, _, _ = perform_parafac2(X, condition_name="disease", rank=20, l1=1e-4)
    # or import from previous analysis
    pf2_genes_4 = pd.read_csv("output/pf2_genes_4.csv", index_col=0)
    X.varm["Pf2_C"] = pf2_genes_4.to_numpy()
    
    # # add the c factor matrix as a varm attribute
    # X.varm["Pf2_C"] = factors[2]

    return X

def genFig():
    X = setup_figure()
    
    cmps = [x for x in range(1, 11)]
    for cmp in cmps:
        f = gsea_analysis_per_cmp(X, cmp, figsize=(4, 4), ofname=f"output_gsea/gsea_cmp_{cmp}.svg")
        f.savefig(f"output_gsea/gsea_cmp_{cmp}.svg", bbox_inches="tight", pad_inches=0.1)
    
    return f