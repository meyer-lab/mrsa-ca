"""This file plots the pf2 factor matrices for the disease datasets"""

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import (
    check_sparsity,
    concat_datasets,
)


def figure_setup():
    """Set up the data for the tensor factorization and return the results"""

    rank = 5

    disease_data = concat_datasets()

    _, factors, _, r2x = perform_parafac2(
        disease_data,
        condition_name="disease",
        rank=rank,
    )

    return factors, r2x, disease_data


def plot_gene_matrix_with_datashader(data_df, ax, title=None, cmap="coolwarm"):
    """
    Plot a gene matrix using datashader with proper data display.
    """

    # Create DataArray with proper coordinates
    data_array = xr.DataArray(
        data_df.values,
        coords=[
            ("gene", np.arange(len(data_df))),
            ("rank", np.arange(len(data_df.columns))),
        ],
    )

    # Create canvas with dimensions matching the data
    cvs = ds.Canvas(
        plot_width=len(data_df.columns) * 100,  # Each rank gets 100 pixels width
        plot_height=1000,  # Fixed height
        x_range=(-0.5, len(data_df.columns) - 0.5),
        y_range=(-0.5, len(data_df) - 0.5),
    )

    # Render directly as raster with no grid lines
    agg = cvs.raster(data_array)

    # Get value range for colormap
    vmin, vmax = np.nanpercentile(data_df.values, [5, 95])
    max_abs = max(abs(vmin), abs(vmax))

    # Create image with proper colormap and NO grid lines
    img = tf.shade(agg, cmap=cmap, how="linear", span=[-max_abs, max_abs])

    # Plot the image with proper extent
    artist = ax.imshow(
        np.array(img),
        aspect="auto",
        extent=(-0.5, len(data_df.columns) - 0.5, len(data_df) - 0.5, -0.5),
        interpolation="nearest",  # Prevent interpolation between ranks
    )

    # Add colorbar
    norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.01)

    # Add labels and title
    ax.set_xlabel("Rank")
    ax.set_ylabel("Genes")
    if title:
        ax.set_title(title)

    # Set tick positions centered on ranks
    ax.set_xticks(range(len(data_df.columns)))
    ax.set_xticklabels(data_df.columns)
    ax.set_yticks([])

    return artist


# Backup function to plot gene matrix using matplotlib's rasterization
def plot_gene_matrix_with_rasterize(
    data_df: pd.DataFrame, ax, title=None, cmap="coolwarm"
):
    """
    Plot a gene matrix using matplotlib's rasterization for efficiency.

    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame with genes as index and ranks as columns
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot on
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name

    Returns:
    --------
    artist : matplotlib.artist.Artist
        The created plot artist
    """
    # Find min/max values for colormap
    vmin, vmax = data_df.values.min(), data_df.values.max()
    max_abs = max(abs(vmin), abs(vmax))

    # Plot directly with imshow (rasterized)
    artist = ax.imshow(
        data_df.values,
        aspect="auto",
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = ax.figure.colorbar(
        artist, ax=ax, orientation="vertical", shrink=0.8, pad=0.01
    )
    cbar.ax.tick_params(labelsize=8)

    # Add zero line to colorbar for emphasis
    cbar.ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # Set axis labels and title
    ax.set_xlabel("Rank")
    ax.set_ylabel("Genes")
    if title:
        ax.set_title(title)

    # Set x-ticks to rank names
    ax.set_xticks(range(len(data_df.columns)))
    ax.set_xticklabels(data_df.columns)

    # Hide y-tick labels (too many genes)
    ax.set_yticks([])

    return artist


def genFig():
    """Generate heatmaps of the factor matrices"""
    layout, fig_size = calculate_layout(num_plots=4, scale_factor=4)
    ax, f, _ = setupBase(fig_size, layout)

    disease_factors, r2x, disease_data = figure_setup()

    disease_ranks = range(1, disease_factors[0].shape[1] + 1)
    disease_ranks_labels = [str(x) for x in disease_ranks]

    # x axis label: rank
    x_ax_label = "Rank"
    # y axis labels: disease, eigen, genes
    d_ax_labels = ["Disease", "Eigen-states", "Genes"]

    # Create a DataFrame for the gene factor matrix
    genes_df = pd.DataFrame(
        disease_factors[2],
        index=disease_data.var.index,
        columns=pd.Index(disease_ranks_labels),
    )

    genes_df.to_csv("output/pf2_genes.csv")

    # Check sparsity of the gene factor matrix
    sparsity = check_sparsity(genes_df.to_numpy())
    gene_count = len(genes_df.index)

    # tick labels: disease, rank
    disease_labels = [
        disease_data.obs["disease"].unique(),
        disease_ranks_labels,
    ]

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

    # plot the gene factor matrix using datashader
    plot_gene_matrix_with_rasterize(
        genes_df,
        ax=ax[2],
        title=f"Gene Factor Matrix\n{gene_count} genes, {sparsity:.2%} sparsity",
        cmap=BC_cmap,
    )

    plot_gene_matrix_with_datashader(
        genes_df,
        ax=ax[3],
        title=f"Gene Factor Matrix (Datashader)\n{gene_count} genes, {sparsity:.2%} sparsity",
        cmap=BC_cmap,
    )

    return f
