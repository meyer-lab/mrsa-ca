"""This file plots the pf2 factor matrices for the disease datasets"""

import os

import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.figures.helpers import plot_gene_matrix
from mrsa_ca_rna.gsea import gsea_analysis_per_cmp
from mrsa_ca_rna.utils import (
    check_sparsity,
    find_top_features,
    prepare_data,
)


def figure_setup():
    """Set up the data for the tensor factorization and return the results"""

    X = prepare_data()

    X, r2x = perform_parafac2(X)

    return X, r2x


# GSEA analysis will make separate plots for each component
def plot_gsea(X: ad.AnnData, gene_set: str = "KEGG_2021_Human"):
    out_dir = os.path.join("output", "gsea", gene_set)
    os.makedirs(out_dir, exist_ok=True)

    # Plot GSEA results for each component
    for i in range(X.varm["Pf2_C"].shape[1]):
        gsea_analysis_per_cmp(
            X,
            cmp=i + 1,
            term_ranks=10,
            gene_set=gene_set,
            figsize=(6, 8),
            out_dir=out_dir,
        )


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (14, 4)
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
    top_genes = find_top_features(genes_df, threshold_fraction=0.75, feature_name="gene")
    top_genes.to_csv(f"output/pf2_genes_{len(ranks_labels)}.csv")

    # Check sparsity of the gene factor matrix
    sparsity = check_sparsity(genes_df.to_numpy())

    # plot the disease factor matrix using coolwarm cmap
    A_df = pd.DataFrame(
        X.uns["Pf2_A"],
        index=X.obs["disease"].unique(),
        columns=ranks_labels,
    )
    a = sns.heatmap(
        A_df,
        ax=ax[0],
        cmap="coolwarm",
        center=0,
        xticklabels=2,
    )
    a.set_title(f"Disease Factor Matrix\nR2X: {r2x:.2f}")
    a.set_xlabel("Rank")
    a.set_ylabel("Disease")

    # Add vertical lines every 5 components
    n_components = X.uns["Pf2_A"].shape[1]
    for i in range(5, n_components, 5):
        a.axvline(i, color="black", linestyle="--", linewidth=0.8)

    # plot the eigenstate factor matrix using diverging cmap
    B_df = pd.DataFrame(
        X.uns["Pf2_B"],
        index=[f"Eigenstate {i + 1}" for i in range(X.uns["Pf2_B"].shape[0])],
        columns=ranks_labels,
    )
    b = sns.heatmap(
        B_df,
        ax=ax[1],
        cmap="coolwarm",
        center=0,
        xticklabels=2,
        yticklabels=2,
    )
    b.set_title("Eigenstate Factor Matrix")
    b.set_xlabel("Rank")
    b.set_ylabel("Eigenstate")

    # plot the gene factor matrix using diverging cmap
    plot_gene_matrix(X, ax=ax[2], title=f"Gene Factor Matrix\nSparsity: {sparsity:.2f}")

    # Plot the GSEA results for desired gene set
    # plot_gsea(X, gene_set="KEGG_2021_Human")

    return f
