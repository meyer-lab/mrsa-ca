"""
This file analyzes associations between diseases and genes for each component
in the PARAFAC2 decomposition by directly comparing the A (disease) and C (gene) modes.
"""

import os
from math import ceil

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import (
    concat_datasets,
    find_top_features,
    calculate_cpm,
)


def get_data():
    """Set up the data for analysis"""
    rank = 1
    X = concat_datasets(filter_threshold=5, min_pct=0.9)

    X, r2x = perform_parafac2(X, slice_col="disease", rank=rank)

    # Organize genes into a DataFrame then analyze
    gene_factors_df = pd.DataFrame(
        X.varm["Pf2_C"],
        index=X.var_names,
        columns=[f"Component_{i + 1}" for i in range(X.uns["Pf2_A"].shape[1])],
    )
    top_genes_df = find_top_features(
        gene_factors_df, threshold_fraction=0.5, feature_name="gene"
    )

    # Organize diseases into a DataFrame then analyze
    disease_factors_df = pd.DataFrame(
        X.uns["Pf2_A"],
        index=X.obs["disease"].unique(),
        columns=[f"Component_{i + 1}" for i in range(X.uns["Pf2_A"].shape[1])],
    )
    top_diseases_df = find_top_features(
        disease_factors_df, threshold_fraction=0.1, feature_name="disease"
    )

    return X, r2x, top_genes_df, top_diseases_df


def highlight_heatmap_columns(ax_heatmap, column_indices, **rect_kwargs):
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
        'fill': False,
        'edgecolor': 'black',
        'linewidth': 2,
        'transform': ax_heatmap.transData
    }
    
    # Update with user-provided properties
    rect_props = {**default_props, **rect_kwargs}
    
    rectangles = []
    
    # Get the number of rows in the heatmap
    # This assumes the heatmap data shape can be inferred from axis limits
    ylim = ax_heatmap.get_ylim()
    num_rows = ylim[1] - ylim[0]
    
    for col_idx in column_indices:
        rect = plt.Rectangle(
            (col_idx, 0),  # x, y position
            1,             # width
            num_rows,      # height
            **rect_props
        )
        ax_heatmap.add_patch(rect)
        rectangles.append(rect)
    
    return rectangles


def plot_component_details(
    ax_disease, ax_pos_gene, ax_neg_gene, 
    comp_num: int, 
    top_diseases_df: pd.DataFrame, 
    top_genes_df: pd.DataFrame, 
    n_genes: int = 10,
    n_diseases: int = 5
):
    """
    Plot disease and gene associations for a specific component.
    
    Parameters
    ----------
    ax_disease, ax_pos_gene, ax_neg_gene : matplotlib.axes.Axes
        Axes for plotting diseases, positive genes, and negative genes
    comp_num : int
        Component number to plot
    top_diseases_df : pd.DataFrame
        DataFrame from find_top_features with disease data
    top_genes_df : pd.DataFrame
        DataFrame from find_top_features with gene data
    n_genes : int, default=10
        Number of genes to display in each direction
    n_diseases : int, default=5
        Number of diseases to display
    """
    comp_key = f"Component_{comp_num}"
    
    # Filter data for the current component
    comp_diseases = top_diseases_df[top_diseases_df["component"] == comp_key]
    comp_genes = top_genes_df[top_genes_df["component"] == comp_key]
    
    # Get top diseases (sorted by absolute value, take top n_diseases)
    top_diseases = comp_diseases.nlargest(n_diseases, 'abs_value')
    
    # Separate positive and negative genes for this component
    pos_genes = comp_genes[comp_genes["direction"] == "positive"].head(n_genes)
    neg_genes = comp_genes[comp_genes["direction"] == "negative"].head(n_genes)
    
    # Get total counts for titles
    total_pos_genes = len(comp_genes[comp_genes["direction"] == "positive"])
    total_neg_genes = len(comp_genes[comp_genes["direction"] == "negative"])
    
    # Plot diseases
    if not top_diseases.empty:
        sns.barplot(x="value", y="disease", data=top_diseases, ax=ax_disease)
        ax_disease.set_title(f"Component {comp_num}: Top Diseases")
        ax_disease.axvline(x=0, color="gray", linestyle="--")
    else:
        ax_disease.text(
            0.5, 0.5, "No significant diseases",
            ha="center", va="center", transform=ax_disease.transAxes
        )
        ax_disease.set_title(f"Component {comp_num}: Top Diseases")
    
    # Plot positive genes
    if not pos_genes.empty:
        sns.barplot(
            x="value", y="gene", data=pos_genes, ax=ax_pos_gene, color="red"
        )
        ax_pos_gene.set_title(f"Positive Genes (n={total_pos_genes})")
        ax_pos_gene.axvline(x=0, color="gray", linestyle="--")
    else:
        ax_pos_gene.text(
            0.5, 0.5, "No positive genes found",
            ha="center", va="center", transform=ax_pos_gene.transAxes
        )
        ax_pos_gene.set_title("Positive Genes")
    
    # Plot negative genes
    if not neg_genes.empty:
        sns.barplot(
            x="value", y="gene", data=neg_genes, ax=ax_neg_gene, color="blue"
        )
        ax_neg_gene.set_title(f"Negative Genes (n={total_neg_genes})")
        ax_neg_gene.axvline(x=0, color="gray", linestyle="--")
    else:
        ax_neg_gene.text(
            0.5, 0.5, "No negative genes found",
            ha="center", va="center", transform=ax_neg_gene.transAxes
        )
        ax_neg_gene.set_title("Negative Genes")
    
    # Ensure the axis limits are well-balanced for gene plots
    for axis in [ax_pos_gene, ax_neg_gene]:
        if axis.patches:  # Check if there are any bars plotted
            xlim = max(abs(min(axis.get_xlim()[0], 0)), abs(max(axis.get_xlim()[1], 0)))
            axis.set_xlim(-xlim, xlim)


def genFig():
    """Generate figures showing the associations between diseases and genes"""
    X, r2x, top_genes_df, top_diseases_df = get_data()

    # Create a multi-panel figure with A matrix + component details
    components_to_show = [1, 2, 3, 4, 5]
    n_genes_to_plot = 10
    n_diseases_to_plot = 5

    # Single column layout - one row for A matrix, one row per component
    nrows = len(components_to_show) + 1
    layout = {"ncols": 1, "nrows": nrows}
    fig_size = (18, nrows * 5)
    ax, f, gs = setupBase(fig_size, layout)

    # A matrix at the top spanning the full width
    ax_heatmap = ax[0]

    # Get components and disease labels
    ranks_labels = [str(x) for x in range(1, X.uns["Pf2_A"].shape[1] + 1)]
    disease_labels = list(X.obs["disease"].unique().astype(str))

    sns.heatmap(
        X.uns["Pf2_A"],
        ax=ax_heatmap,
        cmap="coolwarm",
        center=0,
        xticklabels=ranks_labels,
        yticklabels=disease_labels,
    )
    ax_heatmap.set_title(f"Disease Factor Matrix (A) - R2X: {r2x:.2f}")
    ax_heatmap.set_xlabel("Component")
    ax_heatmap.set_ylabel("Disease")

    # Plot each selected component's diseases and genes
    for i, comp_num in enumerate(components_to_show):
        # Each component gets its own row
        ax_i = ax[i + 1]

        # Split the subplot area for diseases, positive genes, and negative genes
        divider = make_axes_locatable(ax_i)
        ax_disease = ax_i
        ax_pos_gene = divider.append_axes("right", size="100%", pad=1.5)
        ax_neg_gene = divider.append_axes("right", size="100%", pad=1.5)

        # Use the new plotting function
        plot_component_details(
            ax_disease, ax_pos_gene, ax_neg_gene,
            comp_num, top_diseases_df, top_genes_df,
            n_genes=n_genes_to_plot, n_diseases=n_diseases_to_plot
        )

        # Highlight this component in the A matrix
        highlight_heatmap_columns(ax_heatmap, comp_num - 1)

    f.suptitle("PARAFAC2 Component Analysis", fontsize=16)

    # Create histogram figure using the overall top genes from all components
    pos_genes_all = top_genes_df[top_genes_df["direction"] == "positive"]
    neg_genes_all = top_genes_df[top_genes_df["direction"] == "negative"]
    g = plot_gene_expression_histograms(X, pos_genes_all, neg_genes_all, n_genes=n_genes_to_plot)

    return f, g


def plot_gene_expression_histograms(X: ad.AnnData, pos_genes: pd.DataFrame, neg_genes: pd.DataFrame, n_genes: int = 10):
    """Plot histograms for raw expression data of top positive and negative eigen genes."""
    
    # Get top genes from each direction
    top_pos_genes = pos_genes.head(n_genes)["gene"].tolist()
    top_neg_genes = neg_genes.head(n_genes)["gene"].tolist()
    all_genes = top_pos_genes + top_neg_genes
    
    if not all_genes:
        return None
    
    # Calculate layout and setup figure
    layout, fig_size = calculate_layout(len(all_genes), scale_factor=3)
    ax, f, _ = setupBase(fig_size, layout)
    
    # Get CPM expression data
    exp = calculate_cpm(X.layers["raw"])
    
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
            element="bars"
        )
        
        ax_i.set_title(f"{direction}: {gene}")
        ax_i.set_xlabel("Expression (CPM)")
        ax_i.set_ylabel("Count")
    
    f.suptitle("Expression Histograms for Top Genes", fontsize=16)
    return f
