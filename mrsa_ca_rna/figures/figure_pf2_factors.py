"""This file plots the pf2 factor matrices for the disease datasets"""

import os

import anndata as ad
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.figures.helpers import plot_gene_matrix
from mrsa_ca_rna.gsea import gsea_analysis_per_cmp
from mrsa_ca_rna.utils import (
    check_sparsity,
    concat_datasets,
)


def figure_setup():
    """Set up the data for the tensor factorization and return the results"""

    rank = 10

    X = concat_datasets()

    X, r2x = perform_parafac2(
        X,
        slice_col="disease",
        rank=rank,
    )

    return X, r2x


def plot_gsea(X: ad.AnnData, gene_set: str = "KEGG_2021_Human"):
    out_dir = os.path.join("output", "gsea", gene_set)
    os.makedirs(out_dir, exist_ok=True)

    # Plot GSEA results for each component
    for i in range(X.varm["Pf2_C"].shape[1]):
        gsea_analysis_per_cmp(
            X,
            cmp=i + 1,
            term_ranks=5,
            gene_set=gene_set,
            figsize=(6, 4),
            out_dir=out_dir,
        )


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    X, r2x = figure_setup()

    ranks_labels = [str(x) for x in range(1, X.uns["Pf2_A"].shape[1] + 1)]

    # Export gene factor matrix for external analysis
    genes_df = pd.DataFrame(
        X.varm["Pf2_C"],
        index=X.var.index,
        columns=pd.Index(ranks_labels),
    )
    genes_df.to_csv("output/pf2_genes.csv")

    # Check sparsity of the gene factor matrix
    sparsity = check_sparsity(genes_df.to_numpy())

    # Use diverging coolwarm palette for B and C matrices, match A with one-way color
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    warm_half = cmap(np.linspace(0.5, 1, 256))
    A_cmap = colors.ListedColormap(warm_half)

    # plot the disease factor matrix using non-negative cmap
    a = sns.heatmap(
        X.uns["Pf2_A"],
        ax=ax[0],
        cmap=A_cmap,
        vmin=0,
        xticklabels=ranks_labels,
        yticklabels=list(X.obs["disease"].unique().astype(str)),
    )
    a.set_title(f"Disease Factor Matrix\nR2X: {r2x:.2f}")
    a.set_xlabel("Rank")
    a.set_ylabel("Disease")

    # plot the eigenstate factor matrix using diverging cmap
    b = sns.heatmap(
        X.uns["Pf2_B"],
        ax=ax[1],
        cmap="coolwarm",
        center=0,
        xticklabels=ranks_labels,
        yticklabels=ranks_labels,
    )
    b.set_title("Eigenstate Factor Matrix")
    b.set_xlabel("Rank")
    b.set_ylabel("Eigenstate")

    # plot the gene factor matrix using diverging cmap
    plot_gene_matrix(X, ax=ax[2])

    # Plot the GSEA results for KEGG 2021 Human
    plot_gsea(X, gene_set="KEGG_2021_Human")

    return f
