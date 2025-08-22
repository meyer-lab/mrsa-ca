"""
This file analyzes associations between diseases and genes for each component
in the PARAFAC2 decomposition by directly comparing the A (disease) and C (gene) modes.
"""

from math import ceil

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.figures.helpers import plot_component_features
from mrsa_ca_rna.utils import (
    calculate_cpm,
    find_top_features,
    prepare_data,
)


def get_data():
    """Set up the data for analysis"""
    X = prepare_data()

    X, r2x = perform_parafac2(X)

    # Organize genes into a DataFrame then analyze
    gene_factors_df = pd.DataFrame(
        data=X.varm["Pf2_C"],
        index=X.var_names,
        columns=pd.Index(
            range(1, X.varm["Pf2_C"].shape[1] + 1),
        ),
    )
    top_genes_df = find_top_features(
        gene_factors_df, threshold_fraction=0.75, feature_name="gene"
    )

    return X, r2x, top_genes_df


def highlight_heatmap_columns(
    ax_heatmap, column_indices, data_shape=None, **rect_kwargs
):
    """
    Add rectangular highlights over specified columns in a heatmap.

    Parameters
    ----------
    ax_heatmap : matplotlib.axes.Axes
        The heatmap axes to add rectangles to
    column_indices : list or int
        Column index or list of column indices to highlight (0-indexed)
    **rect_kwargs : dict
        Additional keyword arguments to pass to plt.Rectangle
        (e.g., fill, edgecolor, linewidth, etc.)

    Returns
    -------
    list
        List of Rectangle patches that were added
    """
    # Ensure column_indices is a list
    if isinstance(column_indices, int):
        column_indices = [column_indices]

    # Default rectangle properties
    default_props = {
        "fill": False,
        "edgecolor": "black",
        "linewidth": 2,
        "transform": ax_heatmap.transData,
    }

    # Update with user-provided properties
    rect_props = {**default_props, **rect_kwargs}

    rectangles = []

    # Get the number of rows in the heatmap
    if data_shape is not None:
        num_rows = data_shape[0]
    else:
        # Infer from axes properties
        yticks = ax_heatmap.get_yticks()
        if len(yticks) > 0:
            num_rows = len(yticks)
        else:
            # Try to extract from image data
            images = ax_heatmap.get_images()
            if images:
                num_rows = images[0].get_array().shape[0]
            else:
                # Try to use the current axes size
                ylim = ax_heatmap.get_ylim()
                num_rows = ceil(ylim[1] - ylim[0])

    for col_idx in column_indices:
        rect = Rectangle(
            (col_idx, 0),  # x, y position
            1,  # width
            num_rows,  # height
            **rect_props,
        )
        ax_heatmap.add_patch(rect)
        rectangles.append(rect)

    return rectangles


def genFig():
    """Generate figures showing the associations between diseases and genes"""
    X, r2x, top_genes_df = get_data()

    # Create a multi-panel figure with A and C matrices + component details
    components_to_show = [2, 3, 4, 5]
    n_genes_to_plot = 25

    # Two column layout - first row for matrices, subsequent rows for components
    nrows = ceil(len(components_to_show) / 2) + 1
    layout = {"ncols": 2, "nrows": nrows}
    fig_size = (10, nrows * 6)
    ax, f, gs = setupBase(fig_size, layout)

    # Remove first row to make an A matrix spanning all columns
    for i in range(layout["ncols"]):
        f.delaxes(ax[i])
    ax_a_matrix = f.add_subplot(gs[0, :])

    # Get components and disease labels
    ranks_labels = [str(x) for x in range(1, X.uns["Pf2_A"].shape[1] + 1)]
    disease_labels = list(X.obs["disease"].unique().astype(str))

    # Plot A matrix (diseases)
    A_df = pd.DataFrame(
        X.uns["Pf2_A"],
        index=pd.Index(disease_labels),
        columns=pd.Index(ranks_labels),
    )
    sns.heatmap(
        A_df,
        ax=ax_a_matrix,
        cmap="coolwarm",
        center=0,
        xticklabels=2,
        yticklabels=disease_labels,
    )
    ax_a_matrix.set_title(f"Disease Factor Matrix (A) - R2X: {r2x:.2f}")
    ax_a_matrix.set_xlabel("Component")
    ax_a_matrix.set_ylabel("Disease")

    # Plot each selected component's details
    for i, comp_num in enumerate(components_to_show):

        # Skip the first row since it contains the A matrix
        row_idx = i + layout["ncols"]

        # Use the helper function to plot component features
        plot_component_features(
            ax[row_idx],
            top_genes_df,
            comp_num,
            feature_name="gene",
            n_features=n_genes_to_plot,
        )

        # Highlight this component in the A matrix
        highlight_heatmap_columns(ax_a_matrix, comp_num - 1)

    f.suptitle("PARAFAC2 Component Analysis: Disease-Gene Associations", fontsize=16)

    # Create histogram figure using the overall top genes from all components
    top_genes_df = top_genes_df.loc[
        top_genes_df["component"].isin(components_to_show), :
    ]
    pos_genes_all = top_genes_df.loc[top_genes_df["direction"] == "positive", :]
    neg_genes_all = top_genes_df.loc[top_genes_df["direction"] == "negative", :]
    g = plot_gene_expression_histograms(
        X, pos_genes_all, neg_genes_all, n_genes=n_genes_to_plot
    )

    return f, g


def plot_gene_expression_histograms(
    X: ad.AnnData, pos_genes: pd.DataFrame, neg_genes: pd.DataFrame, n_genes: int = 10
):
    """Plot histograms for raw expression data
    of top positive and negative eigen genes."""

    # Get top genes from each direction
    top_pos_genes = pos_genes.head(n_genes)["gene"].tolist()
    top_neg_genes = neg_genes.head(n_genes)["gene"].tolist()
    all_genes = top_pos_genes + top_neg_genes

    if not all_genes:
        return None

    # Calculate layout and setup figure
    fig_size, layout = calculate_layout(len(all_genes), scale_factor=3)
    ax, f, _ = setupBase(fig_size, layout)

    # Get CPM expression data
    exp = calculate_cpm(np.asarray(X.layers["raw"]))

    for i, gene in enumerate(all_genes):
        ax_i = ax[i]

        # Get expression data for this gene
        gene_idx = np.where(X.var.index == gene)[0][0]
        gene_data = exp[:, gene_idx]

        # Determine color and direction
        if gene in top_pos_genes:
            color = "red"
            direction = "Positive"
        else:  # Must be negative
            color = "blue"
            direction = "Negative"

        # Plot histogram
        sns.histplot(
            gene_data,
            ax=ax_i,
            color=color,
            alpha=0.7,
            kde=False,
            bins=30,
            element="bars",
        )

        ax_i.set_title(f"{direction}: {gene}")
        ax_i.set_xlabel("Expression (Log CPM)")
        ax_i.set_ylabel("Count")
        ax_i.set_yscale("log")

    f.suptitle("Expression Histograms for Top Genes", fontsize=16)
    return f
