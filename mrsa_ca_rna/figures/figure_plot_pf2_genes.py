import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import xarray as xr

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import (
    concat_datasets,
)


def get_data():
    """import the data and perform the factorization"""

    X = concat_datasets()

    rank = 5

    _, factors, _, r2x = perform_parafac2(
        X,
        condition_name="disease",
        rank=rank,
    )

    X.varm["Pf2_C"] = factors[2]

    return X, r2x


def plot_datashader_rasterized(data_df: pd.DataFrame, ax, title=None, cmap="viridis"):
    """
    Plot a gene matrix using datashader with proper data display.
    """
    # Print debug info
    print(f"Data shape: {data_df.shape}")
    vmin, vmax = data_df.values.min(), data_df.values.max()
    print(f"Data range: [{vmin:.6f}, {vmax:.6f}]")
    
    # Use the exact same data range approach as the matplotlib version
    max_abs = max(abs(vmin), abs(vmax))
    
    # Create DataArray with numerical coordinates for datashader
    data_array = xr.DataArray(
        data_df.values,
        coords=[
            ("gene", np.arange(len(data_df))),
            ("rank", np.arange(len(data_df.columns))),
        ],
    )
    
    # Create a separate canvas for each column to prevent interpolation between them
    n_columns = len(data_df.columns)
    height = min(1000, max(500, len(data_df) // 50))
    # Final composite image will be assembled from individual column images
    images = []
    
    for col_idx in range(n_columns):
        # Create a canvas for just this column
        width = 100  # Fixed width for each column
        
        cvs = ds.Canvas(
            plot_width=width,
            plot_height=height,
            x_range=(col_idx-0.5, col_idx+0.5),  # Just this column
            y_range=(-0.5, len(data_df)-0.5),
        )
        
        # Render just this column
        agg = cvs.raster(data_array)
        
        # Apply colormap to this column
        img = tf.shade(agg, how='linear', span=[-max_abs, max_abs])
        
        # Store this column's image
        images.append(img.data)
    
    # Combine all column images horizontally
    combined_img = np.hstack(images)
    
    # Plot with correct extent
    artist = ax.imshow(
        combined_img,
        cmap=cmap,
        aspect="auto",
        extent=(-0.5, len(data_df.columns) - 0.5, len(data_df) - 0.5, -0.5),
        interpolation="nearest",
    )
    
    # Add colorbar with the same scaling as the matplotlib version
    norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    
    # Add zero line to colorbar
    cbar.ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # Add grid lines between columns for visual separation
    for col_idx in range(n_columns-1):
        ax.axvline(x=col_idx+0.5, color='black', linewidth=0.5, alpha=0.2)
    
    # Add labels and title
    ax.set_xlabel("Rank")
    ax.set_ylabel("Genes")
    if title:
        ax.set_title(title)
    
    # Set tick positions
    ax.set_xticks(range(len(data_df.columns)))
    ax.set_xticklabels(data_df.columns)
    ax.set_yticks([])
    
    return artist


def plot_matplot_rasterized(data_df: pd.DataFrame, ax, title=None, cmap="coolwarm"):
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

    ax.set_xlabel("Rank")
    ax.set_ylabel("Genes")
    if title:
        ax.set_title(title)
    ax.set_xticks(range(len(data_df.columns)))
    ax.set_xticklabels(data_df.columns)
    ax.set_yticks([])

    return artist


def genFig():
    fig_size = (8, 4)
    layout = {"ncols": 2, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    X, r2x = get_data()

    genes_df = pd.DataFrame(
        X.varm["Pf2_C"],
        index=X.var.index,
        columns=pd.Index([str(x) for x in range(1, X.varm["Pf2_C"].shape[1] + 1)]),
    )

    plot_matplot_rasterized(
        genes_df,
        ax=ax[0],
        title=f"Pf2 Gene Factors\nR2X: {r2x:.2f}",
        cmap="coolwarm",
    )
    plot_datashader_rasterized(
        genes_df,
        ax=ax[1],
        title=f"Pf2 Gene Factors\nR2X: {r2x:.2f}",
        cmap="coolwarm",
    )
    f.suptitle("Pf2 Gene Factors Comparison", fontsize=16)

    return f
