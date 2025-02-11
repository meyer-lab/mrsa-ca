"""This file plots the pf2 factor matrices for the disease datasets"""

import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets, gene_filter


def figure_setup(l1_strength: float = 0.5):
    """Set up the data for the tensor factorization and return the results"""

    # data import, concatenation, scaling, and preparation
    # same as figure11_setup
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, _, recon_err = perform_parafac2(disease_xr, rank=5, l1=l1_strength)
    disease_factors = tensor_decomp[1]
    # disease_projections = tensor_decomp[2]
    r2x = 1 - recon_err

    return disease_factors, r2x, disease_data


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (8, 8)
    layout = {"ncols": 2, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    strenghts = [500, 750]

    for i, l1_strength in enumerate(strenghts):
        disease_factors, r2x, disease_data = figure_setup(l1_strength)

        disease_ranks = range(1, disease_factors[0].shape[1] + 1)
        disease_ranks_labels = [str(x) for x in disease_ranks]
        # x axis label: rank
        x_ax_label = "Rank"
        # y axis labels: disease, eigen, genes
        d_ax_labels = ["Disease", "Eigen-states", "Genes"]

        # get the top genes and sparsity for the gene factor matrix
        threshold = 0.3
        genes_df = pd.DataFrame(disease_factors[2], index=disease_data.var.index)
        top_genes = gene_filter(genes_df.T, threshold = 0.3, method = "any")
        top_genes = top_genes.T

        # check sparsity
        A = genes_df.to_numpy()
        A[np.abs(A) < 0.01] = 0
        sparsity = 1.0 - (np.count_nonzero(A) / A.size)

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
        a.set_title(
            f"Disease Factor Matrix\n"
            f"R2X: {r2x:.2f}"
        )
        a.set_xlabel(x_ax_label)
        a.set_ylabel(d_ax_labels[0])

        # # plot the eigenstates
        # a = sns.heatmap(
        #     disease_factors[1],
        #     ax=ax[1 + 3 * i],
        #     cmap="viridis",
        #     xticklabels=disease_ranks_labels,
        #     yticklabels=disease_labels[1],
        # )
        # a.set_title(
        #     f"Eigen-state Factor Matrix\n"
        #     f"R2X: {r2x:.2f}"
        # )
        # a.set_xlabel(x_ax_label)
        # a.set_ylabel(d_ax_labels[1])

        # plot the gene factors
        a = sns.heatmap(
            disease_factors[2],
            ax=ax[1 + 3 * i],
            cmap="viridis",
            xticklabels=disease_ranks_labels,
            yticklabels=disease_labels[2],
        )
        a.set_title(
            f"Gene Factor Matrix, genes > {threshold}\n"
            f"Sparsity of entire Gene Matrix: {sparsity:.2f}\n"
            f"l1 Strength: {l1_strength}. R2X: {r2x:.2f}"
        )
        a.set_xlabel(x_ax_label)
        a.set_ylabel(d_ax_labels[2])

    return f
