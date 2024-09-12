"""This file will plot the factor matrices of the CA time data"""

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.import_data import ca_data_split
from mrsa_ca_rna.figures.base import setupBase

import pandas as pd
import seaborn as sns


def figure10_setup():
    """Set up the data for the tensor factorization and return the results"""

    time_data, _, _ = ca_data_split(scale=True, tpm=True)

    time_xr = prepare_data(time_data, expansion_dim="subject_id")

    tensor_decomp, _ = perform_parafac2(time_xr, rank=2)
    time_factors = tensor_decomp[1]

    return time_factors


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    time_data, _, _ = ca_data_split(scale=True, tpm=True)

    time_factors = figure10_setup()

    time_ranks = range(1, 3)
    # x axis label: rank
    x_ax_label = "Rank"

    # y axis labels: subject_id, eigen, genes
    t_ax_labels = ["Subject ID", "Eigen-states", "Genes"]

    # push time_factors[2] to a pandas and pick out the top 10 and bottom 10 genes, then trim the data
    genes_df = pd.DataFrame(time_factors[2], index=time_data.var.index)
    top_genes = genes_df.abs().mean(axis=1).nlargest(20).index
    # bottom_genes = genes_df.abs().mean(axis=1).nsmallest(10).index
    genes_df = genes_df.loc[top_genes]

    # put the new genes_df back into the time_factors[2]
    time_factors[2] = genes_df.values

    # tick labels: subject_id, rank, genes
    time_labels = [time_data.obs["subject_id"].unique(), time_ranks, genes_df.index]

    # plot heatmap of disease factors
    for i, factor in enumerate(time_factors):
        a = sns.heatmap(
            factor,
            ax=ax[i],
            cmap="viridis",
            xticklabels=time_ranks,
            yticklabels=time_labels[i],
        )
        a.set_title(f"Time Factor Matrix {i+1}")
        a.set_xlabel(x_ax_label)
        a.set_ylabel(t_ax_labels[i])
        # a.set_xticklabels(disease_ranks)
        # a.set_yticklabels(disease_labels[i])

    return f
