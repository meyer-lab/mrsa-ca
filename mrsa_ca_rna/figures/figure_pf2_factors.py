"""This file plots the pf2 factor matrices for the disease datasets"""

import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import (
    check_sparsity,
    concat_datasets,
    gene_filter,
)


def figure_setup():
    """Set up the data for the tensor factorization and return the results"""

    rank = 5

    datasets = "all"

    disease_data = concat_datasets(
        datasets,
        scale=True,
    )

    _, factors, _, r2x = perform_parafac2(
        disease_data,
        condition_name="disease",
        rank=rank,
    )

    return factors, r2x, disease_data


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    disease_factors, r2x, disease_data = figure_setup()

    disease_ranks = range(1, disease_factors[0].shape[1] + 1)
    disease_ranks_labels = [str(x) for x in disease_ranks]
    # x axis label: rank
    x_ax_label = "Rank"
    # y axis labels: disease, eigen, genes
    d_ax_labels = ["Disease", "Eigen-states", "Genes"]

    genes_df = pd.DataFrame(
        disease_factors[2],
        index=disease_data.var.index,
        columns=pd.Index(disease_ranks_labels),
    )

    genes_df.to_csv("output/pf2_genes.csv")

    # Check sparsity of the gene factor matrix
    sparsity = check_sparsity(genes_df.to_numpy())

    # grab the top 300 genes
    top_n = 300
    genes_df: pd.DataFrame = gene_filter(
        genes_df.T, threshold=0, method="mean", top_n=top_n
    ).T

    # put the new genes_df back into the disease_factors[2]
    disease_factors[2] = genes_df.values

    # tick labels: disease, rank, genes
    disease_labels = [
        disease_data.obs["disease"].unique(),
        disease_ranks_labels,
        False,
    ]

    # plot heatmap of disease factors with independent cmaps

    # Set the A matrix colors
    A_cmap = sns.color_palette("light:#df20df", as_cmap=True)

    # set the B and C matrix colors
    BC_cmap = sns.diverging_palette(145, 300, as_cmap=True)

    # plot the disease factor matrix using non-negative cmap
    a = sns.heatmap(
        disease_factors[0],
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
        disease_factors[1],
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
        disease_factors[2],
        ax=ax[2],
        cmap=BC_cmap,
        center=0,
        xticklabels=disease_ranks_labels,
        yticklabels=disease_labels[2],
    )
    c.set_title(
        f"Gene Factor Matrix\n"
        f"R2X: {r2x:.2f} | Top {top_n} genes\n"
        f"Sparsity: {sparsity:.2f}"
    )
    c.set_xlabel(x_ax_label)
    c.set_ylabel(d_ax_labels[2])

    return f
