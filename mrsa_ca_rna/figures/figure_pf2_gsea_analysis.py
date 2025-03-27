"""Plots gsea analysis for multiple components in a grid"""

import anndata as ad
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.gene_analysis import gsea_analysis_per_cmp
from mrsa_ca_rna.utils import (
    check_sparsity,
    concat_datasets,
    gene_converter,
    gene_filter,
)


def export_projections(X: ad.AnnData, projections):
    rank_labels = [x for x in range(1, projections[0].shape[1] + 1)]

    # extract the MRSA and CA projections (104 and 88 samples, respectively)
    for projection in projections:
        if projection.shape[0] == 104:
            ca_index = X.obs[X.obs["disease"] == "Candidemia"].index
            Pf2_CA = pd.DataFrame(
                projection, index=ca_index, columns=pd.Index(rank_labels)
            )

        if projection.shape[0] == 88:
            mrsa_index = X.obs[X.obs["disease"] == "MRSA"].index
            Pf2_MRSA = pd.DataFrame(
                projection, index=mrsa_index, columns=pd.Index(rank_labels)
            )

    Pf2_CA.to_csv("output_gsea/Pf2_CA.csv")
    Pf2_MRSA.to_csv("output_gsea/Pf2_MRSA.csv")

    return 0


def setup_figure():
    # set high fms parameters to get a good decomposition
    threshold = 4
    rank = 15
    l1 = 1e-4

    # import data
    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    X: ad.AnnData = concat_datasets(
        disease_list, filter_threshold=threshold, scale=True
    )

    # convert to gene symbols
    X = gene_converter(X, old_id="EnsemblGeneID", new_id="Symbol", method="columns")

    # define a callback to probe pf2 fitting
    sparsities = []

    def callback(_, __, factors, ___):
        sparsity = check_sparsity(factors[2])
        sparsities.append(sparsity)
        return 0

    _, factors, projections, R2X = perform_parafac2(
        X, condition_name="disease", rank=rank, l1=l1, rnd_seed=42, callback=callback
    )

    # Attach the C factor matrix to the AnnData object for gene expression analysis
    X.varm["Pf2_C"] = factors[2]

    # dress up the factors for heatmap plotting by making pd.DataFrames with labels
    rank_labels = [x for x in range(1, factors[0].shape[1] + 1)]
    disease_labels = X.obs["disease"].unique()

    Pf2_A = pd.DataFrame(
        factors[0], index=pd.Index(disease_labels), columns=pd.Index(rank_labels)
    )
    Pf2_B = pd.DataFrame(
        factors[1], index=pd.Index(rank_labels), columns=pd.Index(rank_labels)
    )
    Pf2_C = pd.DataFrame(factors[2], index=X.var.index, columns=pd.Index(rank_labels))

    # export_projections(X, projections)

    # trim the gene factor matrix to the top 100 genes
    Pf2_C = gene_filter(Pf2_C.T, threshold=0, method="mean", top_n=100)
    Pf2_C = Pf2_C.T

    # collect metrics for the heatmaps
    metrics = {
        "l1": l1,
        "rank": rank,
        "threshold": threshold,
        "R2X": R2X,
        "sparsity": sparsities[-1],
    }

    return X, Pf2_A, Pf2_B, Pf2_C, metrics


def genFig():
    figsize = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(figsize, layout)

    X, Pf2_A, Pf2_B, Pf2_C, metrics = setup_figure()

    # set up diverging colormaps for the heatmaps
    A_cmap = sns.color_palette("light:#df20df", as_cmap=True)
    BC_cmap = sns.diverging_palette(145, 300, as_cmap=True)

    # plot heatmaps of the factor matrices
    a = sns.heatmap(
        Pf2_A,
        ax=ax[0],
        cmap=A_cmap,
        vmin=0,
        xticklabels=Pf2_A.columns,
        yticklabels=Pf2_A.index,
    )
    a.set_title(f"Disease Factors\nR2X: {metrics["R2X"]:.2f}")
    a.set_xlabel("Rank")
    a.set_ylabel("Disease")

    b = sns.heatmap(
        Pf2_B,
        ax=ax[1],
        cmap=BC_cmap,
        center=0,
        xticklabels=Pf2_B.columns,
        yticklabels=Pf2_B.index,
    )
    b.set_title("Eigenstate Factors")
    b.set_xlabel("Rank")
    b.set_ylabel("Eigenstates")

    c = sns.heatmap(
        Pf2_C,
        ax=ax[2],
        cmap=BC_cmap,
        center=0,
        xticklabels=Pf2_C.columns,
        yticklabels=False,
    )
    c.set_title(
        f"Gene Factors\nTop 100 genes\nTPM > {metrics["threshold"]}\n"
        f"Sparsity: {metrics["sparsity"]:.2f} @ L1: {metrics["l1"]:.2e}"
    )
    c.set_xlabel("Rank")
    c.set_ylabel("Genes")

    # perfrom gsea analysis on each component of the gene factor matrix
    cmps = [x for x in range(1, Pf2_C.shape[1] + 1)]
    for cmp in tqdm(cmps, desc="Performing GSEA Analysis", leave=True):
        gsea_analysis_per_cmp(X, cmp, figsize=(4, 4), out_dir="output_gsea/")

    return f
