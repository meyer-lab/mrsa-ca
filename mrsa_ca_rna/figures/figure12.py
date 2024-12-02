"""This file plots the pf2 factor matrices for the disease datasets"""

import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets


def figure12_setup(l1_strength: float = 0.5):
    """Set up the data for the tensor factorization and return the results"""

    # data import, concatenation, scaling, and preparation
    # same as figure11_setup
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, recon_err = perform_parafac2(disease_xr, rank=50, l1=l1_strength)
    disease_factors = tensor_decomp[1]
    # disease_projections = tensor_decomp[2]
    r2x = 1 - min(recon_err)

    return disease_factors, r2x, disease_data


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 20)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    strenghts = [0.5]

    for i, l1_strength in enumerate(strenghts):
        disease_factors, r2x, disease_data = figure12_setup(l1_strength)

        disease_ranks = range(1, disease_factors[0].shape[1] + 1)
        disease_ranks_labels = [str(x) for x in disease_ranks]
        # x axis label: rank
        x_ax_label = "Rank"
        # y axis labels: disease, eigen, genes
        d_ax_labels = ["Disease", "Eigen-states", "Genes"]

        # push disease_factors[2] to a pandas and pick out the top 200 most
        # correlated/anti-correlated, then trim the data
        genes_df = pd.DataFrame(disease_factors[2], index=disease_data.var.index)
        mean_genes = pd.Series(genes_df.abs().mean(axis=1))
        top_genes = mean_genes.nlargest(200).index
        genes_df = genes_df.loc[top_genes]

        # put the new genes_df back into the disease_factors[2]
        disease_factors[2] = genes_df.values

        # tick labels: disease, rank, genes
        disease_labels = [
            disease_data.obs["disease"].unique(),
            disease_ranks_labels,
            False,
        ]

        # plot heatmap of disease factors
        for j, factor in enumerate(disease_factors):
            a = sns.heatmap(
                factor,
                ax=ax[j + (i * 3)],
                cmap="viridis",
                xticklabels=disease_ranks_labels,
                yticklabels=disease_labels[j],
            )
            a.set_title(
                f"Disease Factor Matrix {j+1}\n"
                f"l1 Strength: {l1_strength}. R2X: {r2x:.2f}"
            )
            a.set_xlabel(x_ax_label)
            a.set_ylabel(d_ax_labels[j])

    return f
