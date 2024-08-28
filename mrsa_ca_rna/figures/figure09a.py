"""This file will explore the effect of scale on the diseease factor matrix by scaling
the data in various ways and comparing the results of parafac2"""

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.import_data import concat_datasets, extract_time_data
from mrsa_ca_rna.figures.base import setupBase

import numpy as np
import pandas as pd
import seaborn as sns


def figure09a_setup(disease_data=None):
    """Set up the data for the tensor factorization and return the results"""

    if disease_data is None:
        disease_data = concat_datasets(scale=True, tpm=True)

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, _ = perform_parafac2(disease_xr, rank=20)
    disease_factors = tensor_decomp[1]

    return disease_factors


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 16)
    layout = {"ncols": 3, "nrows": 4}
    ax, f, _ = setupBase(fig_size, layout)

    cases = [0, 1, "f", "s"]

    for j, case in enumerate(cases):
        disease_data = concat_datasets(scaled=(True, case), tpm=True)

        disease_factors = figure09a_setup(disease_data)

        disease_ranks = range(1, 21)
        # x axis label: rank
        x_ax_label = "Rank"
        # y axis labels: disease, eigen, genes
        d_ax_labels = ["Disease", "Eigen-states", "Genes"]

        # push disease_factors[2] to a pandas and pick out the top 20 most correlated/anti-correlated, then trim the data
        genes_df = pd.DataFrame(disease_factors[2], index=disease_data.var.index)
        top_genes = genes_df.abs().mean(axis=1).nlargest(20).index
        # bottom_genes = genes_df.abs().mean(axis=1).nsmallest(10).index
        genes_df = genes_df.loc[top_genes]

        # put the new genes_df back into the disease_factors[2]
        disease_factors[2] = genes_df.values

        # tick labels: disease, rank, genes
        disease_labels = [
            disease_data.obs["disease"].unique(),
            disease_ranks,
            genes_df.index,
        ]

        # plot heatmap of disease factors
        for i, factor in enumerate(disease_factors):
            a = sns.heatmap(
                factor,
                ax=ax[i + j * 3],
                cmap="viridis",
                xticklabels=disease_ranks,
                yticklabels=disease_labels[i],
            )
            a.set_title(f"Disease Factor Matrix {i+1}, case: {case}")
            a.set_xlabel(x_ax_label)
            a.set_ylabel(d_ax_labels[i])

    return f
