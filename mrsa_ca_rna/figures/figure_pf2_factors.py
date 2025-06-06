"""This file plots the pf2 factor matrices for the disease datasets"""

import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import (
    check_sparsity,
    concat_datasets,
)


def figure_setup():
    """Set up the data for the tensor factorization and return the results"""

    rank = 5

    X = concat_datasets()

    X, r2x = perform_parafac2(
        X,
        slice_col="disease",
        rank=rank,
    )

    return X, r2x


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 8)
    layout = {"ncols": 3, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    X, r2x = figure_setup()

    disease_ranks = range(1, X.uns["Pf2_A"].shape[1] + 1)
    disease_ranks_labels = [str(x) for x in disease_ranks]
    # x axis label: rank
    x_ax_label = "Rank"
    # y axis labels: disease, eigen, genes
    d_ax_labels = ["Disease", "Eigen-states", "Genes"]

    genes_df = pd.DataFrame(
        X.varm["Pf2_C"],
        index=X.var.index,
        columns=pd.Index(disease_ranks_labels),
    )

    genes_df.to_csv("output/pf2_genes.csv")

    # Check sparsity of the gene factor matrix
    sparsity = check_sparsity(genes_df.to_numpy())

    # tick labels: disease, rank, genes
    disease_labels = [
        X.obs["disease"].unique(),
        disease_ranks_labels,
        False,
    ]

    # Set the A matrix colors
    A_cmap = sns.color_palette("light:#df20df", as_cmap=True)

    # set the B and C matrix colors
    BC_cmap = sns.diverging_palette(145, 300, as_cmap=True)

    # plot the disease factor matrix using non-negative cmap
    a = sns.heatmap(
        X.uns["Pf2_A"],
        ax=ax[0],
        cmap=A_cmap,
        vmin=0,
        xticklabels=disease_ranks_labels,
        yticklabels=disease_labels[0],
    )
    a.set_title(f"Disease Factor Matrix\nR2X: {r2x:.2f}")
    a.set_xlabel(x_ax_label)
    a.set_ylabel(d_ax_labels[0])

    # plot the eigenstate factor matrix using diverging cmap
    b = sns.heatmap(
        X.uns["Pf2_B"],
        ax=ax[1],
        cmap=BC_cmap,
        xticklabels=disease_ranks_labels,
        yticklabels=disease_labels[1],
    )
    b.set_title("Eigenstate Factor Matrix")
    b.set_xlabel(x_ax_label)
    b.set_ylabel(d_ax_labels[1])

    # plot the gene factor matrix using diverging cmap
    c = sns.heatmap(
        X.varm["Pf2_C"],
        ax=ax[2],
        cmap=BC_cmap,
        center=0,
        xticklabels=disease_ranks_labels,
        yticklabels=disease_labels[2],
    )
    c.set_title(f"Gene Factor Matrix\nR2X: {r2x:.2f}\nSparsity: {sparsity:.2f}")
    c.set_xlabel(x_ax_label)
    c.set_ylabel(d_ax_labels[2])

    d = sns.scatterplot(
        x=X.obsm["Pf2_PaCMAP"][:, 0],
        y=X.obsm["Pf2_PaCMAP"][:, 1],
        hue=X.obs["disease"],
        ax=ax[3],
    )
    d.set_title("PaCMAP Projection of Weighted Projections")
    d.set_xlabel("PaCMAP 1")
    d.set_ylabel("PaCMAP 2")

    return f
