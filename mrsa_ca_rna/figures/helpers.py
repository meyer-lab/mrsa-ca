"""This file contains helper functions for plotting pf2 factor matrices"""

import time

import anndata as ad
import numpy as np
import pandas as pd
from fastcluster import linkage
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import leaves_list


def plot_table_rasterized(data_df: pd.DataFrame, ax: Axes, title=None, cmap="coolwarm"):
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
    
    # Add vertical lines every 5th column
    n_columns = len(data_df.columns)
    for i in range(4, n_columns, 5):  # Starting at 4 (5th column, 0-indexed)
        ax.axvline(i + 0.5, color="black", linestyle="-", linewidth=0.8)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Genes")
    if title:
        ax.set_title(title)
    ax.set_xticks(range(len(data_df.columns)))
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
        pd.DataFrame(X, index=yt, columns=xticks), ax, title=title
    )

    return artist


def reorder_table(X):
    """Reorders a table's rows using heirarchical clustering."""
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
