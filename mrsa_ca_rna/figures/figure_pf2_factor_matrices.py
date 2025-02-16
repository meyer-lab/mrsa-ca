"""This file plots the pf2 factor matrices for the disease datasets"""

import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import (
    concat_datasets,
    gene_filter,
    normalize_factors,
    sparsity_check,
)


def figure_setup(l1_strength: float = 0.5):
    """Set up the data for the tensor factorization and return the results"""

    # data import, concatenation, scaling, and preparation
    # same as figure11_setup
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, _, diag = perform_parafac2(
        disease_xr, rank=10, l1=l1_strength, normalize=False
    )
    disease_factors = tensor_decomp[1]
    # disease_projections = tensor_decomp[2]
    r2x = 1 - diag.rec_errors[-1]

    return disease_factors, r2x, disease_data


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    strenghts = [2000, 5000, 10000]
    nrows = len(strenghts)

    fig_size = (12, nrows * 4)
    layout = {"ncols": 3, "nrows": nrows}
    ax, f, _ = setupBase(fig_size, layout)

    for i, l1_strength in enumerate(strenghts):
        disease_factors, r2x, disease_data = figure_setup(l1_strength)

        disease_ranks = range(1, disease_factors[0].shape[1] + 1)
        disease_ranks_labels = [str(x) for x in disease_ranks]
        # x axis label: rank
        x_ax_label = "Rank"
        # y axis labels: disease, eigen, genes
        d_ax_labels = ["Disease", "Eigen-states", "Genes"]

        # check sparsity of the genes_df prior to normalization
        sparsity = sparsity_check(disease_factors[2], threshold=1e-4)

        # normalize the disease and gene factors
        normalized_factors, _ = normalize_factors([disease_factors[0], disease_factors[2]])
        disease_factors[0] = normalized_factors[0]
        disease_factors[2] = normalized_factors[1]

        # get the top genes and sparsity for the gene factor matrix
        genes_df = pd.DataFrame(disease_factors[2], index=disease_data.var.index)
        top_genes = gene_filter(genes_df.T, threshold=0, method="any", top_n=300)
        top_genes = top_genes.T

        # put the new genes_df back into the disease_factors[2]
        disease_factors[2] = top_genes.values

        # tick labels: disease, rank, genes
        disease_labels = [
            disease_data.obs["disease"].unique(),
            disease_ranks_labels,
            False,
        ]

        # plot heatmap of disease factors

        # plot the disease factors
        a = sns.heatmap(
            disease_factors[0],
            ax=ax[0 + 3 * i],
            cmap="viridis",
            xticklabels=disease_ranks_labels,
            yticklabels=disease_labels[0],
        )
        a.set_title(f"Normalized Disease Factor Matrix\n" f"R2X: {r2x:.2f}")
        a.set_xlabel(x_ax_label)
        a.set_ylabel(d_ax_labels[0])

        # plot the eigenstates
        a = sns.heatmap(
            disease_factors[1],
            ax=ax[1 + 3 * i],
            cmap="viridis",
            xticklabels=disease_ranks_labels,
            yticklabels=disease_labels[1],
        )
        a.set_title(f"Eigen-state Factor Matrix\n" f"R2X: {r2x:.2f}")
        a.set_xlabel(x_ax_label)
        a.set_ylabel(d_ax_labels[1])

        # plot the gene factors
        a = sns.heatmap(
            disease_factors[2],
            ax=ax[2 + 3 * i],
            cmap="viridis",
            xticklabels=disease_ranks_labels,
            yticklabels=disease_labels[2],
        )
        a.set_title(
            f"Normalized Gene Factor Matrix\n"
            f"Showing top {top_genes.shape[0]} genes out of {genes_df.shape[0]}\n"
            f"Sparsity (tol: 1e-4) of gene matrix: {sparsity:.2f}\n"
            f"l1 Strength: {l1_strength}. R2X: {r2x:.2f}"
        )
        a.set_xlabel(x_ax_label)
        a.set_ylabel(d_ax_labels[2])

    return f
