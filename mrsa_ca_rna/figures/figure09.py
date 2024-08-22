"""This file will plot the results of the tensor factorization of
MRSA and CA rna data, either together or just CA in time."""

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.import_data import concat_datasets, extract_time_data
from mrsa_ca_rna.figures.base import setupBase

import numpy as np
import pandas as pd
import seaborn as sns


def figure09_setup():
    """Set up the data for the tensor factorization and return the results"""

    disease_data = concat_datasets(scaled=True, tpm=True)

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, _ = perform_parafac2(disease_xr, rank=20)
    disease_factors = tensor_decomp[1]

    return disease_factors


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (16, 8)
    layout = {"ncols": 4, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    disease_data = concat_datasets(scaled=True, tpm=True)

    disease_factors = figure09_setup()

    disease_ranks = range(1, 21)
    # x axis label: rank
    x_ax_label = "Rank"
    # y axis labels: disease, eigen, genes
    d_ax_labels = ["Disease", "Rank", "Genes"]

    # tick labels: disease, rank, genes
    disease_labels = [disease_data.obs["disease"].unique(), disease_ranks, 500]

    # plot heatmap of disease factors
    for i, factor in enumerate(disease_factors):
        a = sns.heatmap(
            factor,
            ax=ax[i],
            cmap="viridis",
            xticklabels=disease_ranks,
            yticklabels=disease_labels[i],
        )
        a.set_title(f"Disease Factor Matrix {i+1}")
        a.set_xlabel(x_ax_label)
        a.set_ylabel(d_ax_labels[i])

    return f
