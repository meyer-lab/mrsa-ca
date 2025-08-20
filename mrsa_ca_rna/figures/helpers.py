"""This file contains helper functions for plotting pf2 factor matrices"""

import time

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from fastcluster import linkage
from matplotlib.axes import Axes
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
) -> Axes:
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
    n_features : int, default=5
        Number of features to display in each direction
    pos_color : str, default="red"
        Color for positive features
    neg_color : str, default="blue"
        Color for negative features
    pad : float, default=0.6
        Padding between positive and negative plots

    Returns
    -------
    tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        Axes for positive and negative feature plots
    """

    # Filter data for the specified component
    comp_features = features_df[features_df["component"] == component].copy()

    # We only want the top n features in each direction
    pos_features = comp_features.loc[comp_features["direction"] == "positive"].head(
        n_features
    )
    neg_features = comp_features.loc[comp_features["direction"] == "negative"].head(
        n_features
    )
    combined_features = pd.concat([pos_features, neg_features])

    if not combined_features.empty:
        sns.barplot(
            data=combined_features,
            x="value",
            y=feature_name,
            hue="direction",
            palette={"positive": pos_color, "negative": neg_color},
            ax=ax,
            orient="h",
            legend=False,
        )
        ax.set_title(
            f"Top {n_features} Positive and Negative "
            f"{feature_name.capitalize()}s\nComponent: {component}"
        )
        ax.set_xlabel("")
        ax.set_ylabel(feature_name.capitalize())
    else:
        ax.text(
            0.5,
            0.5,
            f"No {feature_name}s found for {component}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"No features found for {component}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        return ax

    if ax.patches:  # Check if there are any bars plotted
        xlim = max(abs(min(ax.get_xlim()[0], 0)), abs(max(ax.get_xlim()[1], 0)))
        ax.set_xlim(-xlim, xlim)

    return ax
