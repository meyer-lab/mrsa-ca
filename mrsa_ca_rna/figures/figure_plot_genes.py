"""This file plots the gene factor matrix from the PF2 model."""

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.figures.helpers import plot_gene_matrix
from mrsa_ca_rna.utils import concat_datasets


def get_data():
    """Get the data for plotting the gene factor matrix."""
    X = concat_datasets()

    rank = 5

    # Perform PARAFAC2 factorization
    X, _ = perform_parafac2(X, slice_col="disease", rank=rank)

    return X


def genFig():
    """Generate the figure for the gene factor matrix from the PF2 model."""

    # Get the data
    X = get_data()

    # Set up the figure
    layout = {"ncols": 1, "nrows": 1}
    fig_size = (8, 8)
    ax, f, _ = setupBase(fig_size, layout)

    plot_gene_matrix(X, ax[0])

    return f
