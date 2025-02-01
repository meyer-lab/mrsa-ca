"""This file plots the pf2 factor matrices for the disease datasets"""

import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets, gene_converter


def figure12_setup(l1_strength: float = 0.5):
    """Set up the data for the tensor factorization and return the results"""

    # data import, concatenation, scaling, and preparation
    # same as figure11_setup
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    tensor_decomp, _, recon_err = perform_parafac2(disease_xr, rank=20, l1=l1_strength)
    disease_factors = tensor_decomp[1]
    # disease_projections = tensor_decomp[2]
    r2x = 1 - recon_err

    return disease_factors, r2x, disease_data

def get_top_genes(genes: pd.DataFrame, n_genes: int = 200, n_comp: int = 0, print_csv: bool = False):
    """Reports the top n_genes by mean absolute value across all components
    organized by component. If n_comp is specified, only the top n_comp components"""

    mean_genes = pd.Series(genes.abs().mean(axis=1))
    top_genes = mean_genes.nlargest(n_genes).index
    top_df = genes.loc[top_genes]

    # convert EnsemblGeneID to Symbol, then print to csv
    top_df = gene_converter(
        top_df, "EnsemblGeneID", "Symbol", "index"
    )

    # if n_comp is specified return only the top weighted components and order
    # the genes by absolute value per component
    if n_comp > 0:
        top_comps = top_df.abs().sum(axis=0).nlargest(n_comp).index
        top_df = top_df[top_comps]


    # print to csv
    if print_csv:
        popped_df = top_df.reset_index(names=["Gene"])

        popped_df["Gene"].to_csv(
            "mrsa_ca_rna/output/figure12_top_genes.csv",
            index=False,
            header=False
        )

    return top_df


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 20)
    layout = {"ncols": 3, "nrows": 5}
    ax, f, _ = setupBase(fig_size, layout)

    strenghts = [100]

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
        top_genes = get_top_genes(genes_df, n_genes=200, print_csv=True)

        # put the new genes_df back into the disease_factors[2]
        disease_factors[2] = top_genes.values

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