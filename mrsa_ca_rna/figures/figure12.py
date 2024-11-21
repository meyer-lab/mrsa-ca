"""This file plots the pf2 factor matrices for the disease datasets"""

import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets


def figure12_setup():
    """Set up the data for the tensor factorization and return the results"""

    # data import, concatenation, scaling, and preparation
    # same as figure11_setup
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, recon_err = perform_parafac2(disease_xr, rank=50, l1=0.5)
    disease_factors = tensor_decomp[1]
    disease_projections = tensor_decomp[2]
    r2x = 1 - min(recon_err)

    return disease_factors, disease_projections, r2x, disease_data


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 8)
    layout = {"ncols": 3, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    disease_factors, disease_projections, r2x, disease_data = figure12_setup()

    disease_ranks = range(1, disease_factors[0].shape[1] + 1)
    disease_ranks_labels = [str(x) for x in disease_ranks]
    # x axis label: rank
    x_ax_label = "Rank"
    # y axis labels: disease, eigen, genes
    d_ax_labels = ["Disease", "Eigen-states", "Genes"]

    # push disease_factors[2] to a pandas and pick out the top 20 most
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
    for i, (factor, projection) in enumerate(
        zip(disease_factors, disease_projections, strict=False)
    ):
        a = sns.heatmap(
            factor,
            ax=ax[i],
            cmap="viridis",
            xticklabels=disease_ranks_labels,
            yticklabels=disease_labels[i],
        )
        a.set_title(f"Disease Factor Matrix {i+1}\nR2X: {r2x:.2f}")
        a.set_xlabel(x_ax_label)
        a.set_ylabel(d_ax_labels[i])

        weighted_projection = projection @ disease_factors[1]
        wp_y_labels = [
            str(d + " Patients") for d in disease_data.obs["disease"].unique()
        ]
        a = sns.heatmap(
            weighted_projection,
            ax=ax[i + 3],
            cmap="viridis",
            xticklabels=5,
            yticklabels=5,
        )
        a.set_title(f"{disease_data.obs['disease'].unique()[i]} Weighted Projection")
        a.set_ylabel(wp_y_labels[i])
        a.set_xlabel("Rank")
    return f
