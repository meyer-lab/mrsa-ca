"""This file contains helper functions for plotting pf2 factor matrices"""

import time

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from fastcluster import linkage
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import leaves_list


def plot_table_rasterized(data_df: pd.DataFrame, ax: Axes, title=None, cmap="coolwarm"):
    # Find min/max values for colormap
    vmin, vmax = data_df.values.min(), data_df.values.max()
    max_abs = max(abs(vmin), abs(vmax))

    # Plot directly with imshow (rasterized) but align with seaborn coordinate system
    artist = ax.imshow(
        data_df.values,
        aspect="auto",
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
        extent=(0, len(data_df.columns), 0, len(data_df.index)),  # Align with seaborn
    )

    # Add colorbar
    cbar = ax.figure.colorbar(
        artist, ax=ax, orientation="vertical", shrink=0.8, pad=0.01
    )
    cbar.ax.tick_params(labelsize=8)

    # Add zero line to colorbar for emphasis
    cbar.ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # Add vertical lines every 5th column - now aligned with seaborn coordinates
    n_columns = len(data_df.columns)
    for i in range(5, n_columns + 1, 5):  # Every 5th column boundary
        ax.axvline(i, color="black", linestyle="-", linewidth=0.8)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Genes")
    if title:
        ax.set_title(title)

    # Set ticks to match seaborn heatmap style
    ax.set_xticks(np.arange(len(data_df.columns)) + 0.5)  # Center ticks on columns
    ax.set_xticklabels(data_df.columns)
    ax.set_yticks([])

    return artist


def plot_gene_matrix(data: ad.AnnData, ax: Axes, title=None):
    """Plots Pf2 gene factors"""
    rank = data.varm["Pf2_C"].shape[1]
    X = np.array(data.varm["Pf2_C"])
    yt = data.var.index.values

    ind = reorder_table(X)
    X = X[ind]
    X = X / np.max(np.abs(X))
    yt = [yt[ii] for ii in ind]
    xticks = np.arange(1, rank + 1)

    artist = plot_table_rasterized(
        pd.DataFrame(X, index=pd.Index(yt), columns=pd.Index(xticks)), ax, title=title
    )

    return artist


def reorder_table(X):
    """Reorders a table's rows using hierarchical clustering."""
    start_time = time.time()
    try:
        # Perform hierarchical clustering
        Z = linkage(X, method="ward")
        ind = leaves_list(Z)

        print(f"Clustering completed in {time.time() - start_time:.2f} seconds.")
        return ind
    except Exception as e:
        print(f"Clustering failed after {time.time() - start_time:.2f} seconds: {e}")
        return np.arange(X.shape[0])

def plot_component_features(
    ax: Axes,
    features_df: pd.DataFrame,
    component: int | str,
    feature_name: str = "feature",
    n_features: int = 5,
    pos_color: str = "red",
    neg_color: str = "blue",
    pad: float = 0.6,
) -> tuple[Axes, Axes]:
    """
    Plot positive and negative features in a split barplot layout.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to split for plotting
    features_df : pd.DataFrame
        DataFrame from find_top_features with feature data
    component : str
        Component identifier to filter for (e.g., "Component_1")
    feature_name : str, default="feature"
        Name of the feature column in the DataFrame
    n_features : int, default=10
        Number of features to display in each direction
    pos_color : str, default="red"
        Color for positive features
    neg_color : str, default="blue"
        Color for negative features
    title : str, optional
        Title for the plot. If None, uses component name
    pad : float, default=0.6
        Padding between positive and negative plots
        
    Returns
    -------
    tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        Axes for positive and negative feature plots
    """
    
    # Filter data for the specified component
    comp_features = features_df[features_df["component"] == component].copy()
    
    # Separate positive and negative features
    pos_features = comp_features[comp_features["direction"] == "positive"].head(n_features)
    neg_features = comp_features[comp_features["direction"] == "negative"].head(n_features)
    
    # Get total counts for titles
    total_pos_features = len(comp_features[comp_features["direction"] == "positive"])
    total_neg_features = len(comp_features[comp_features["direction"] == "negative"])
    
    # Split the axes
    divider = make_axes_locatable(ax)
    ax_pos = ax  # Top plot uses the original axis
    ax_neg = divider.append_axes("bottom", size="100%", pad=pad)
    
    # Plot positive features
    if not pos_features.empty:
        sns.barplot(
            data=pos_features, 
            x="value", 
            y=feature_name, 
            ax=ax_pos, 
            color=pos_color
        )
        ax_pos.set_title(f"{component}: Positive Features (n={total_pos_features})")
        ax_pos.axvline(x=0, color="gray", linestyle="--")
    else:
        ax_pos.text(
            0.5, 0.5, "No positive features found",
            ha="center", va="center", transform=ax_pos.transAxes
        )
        ax_pos.set_title(f"{component}: Positive Features")
    
    # Plot negative features
    if not neg_features.empty:
        sns.barplot(
            data=neg_features, 
            x="value", 
            y=feature_name, 
            ax=ax_neg, 
            color=neg_color
        )
        ax_neg.set_title(f"{component}: Negative Features (n={total_neg_features})")
        ax_neg.axvline(x=0, color="gray", linestyle="--")
    else:
        ax_neg.text(
            0.5, 0.5, "No negative features found",
            ha="center", va="center", transform=ax_neg.transAxes
        )
        ax_neg.set_title(f"{component}: Negative Features")
    
    # Balance axis limits
    for axis in [ax_pos, ax_neg]:
        if axis.patches:  # Check if there are any bars plotted
            xlim = max(abs(min(axis.get_xlim()[0], 0)), abs(max(axis.get_xlim()[1], 0)))
            axis.set_xlim(-xlim, xlim)
    
    
    return ax_pos, ax_neg