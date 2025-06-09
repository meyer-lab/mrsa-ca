import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm as cm
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


def plot_datashader_rasterized(data_df: pd.DataFrame, ax, title=None, cmap="coolwarm"):
    """
    Plot a gene matrix using datashader with proper data display.
    """

    # Create DataArray with numerical coordinates for datashader
    data_array = xr.DataArray(
        data_df.values,
        coords=[
            ("gene", np.arange(len(data_df))),
            ("rank", np.arange(len(data_df.columns))),
        ],
    )

    # Create canvas with dimensions matching the data
    cvs = ds.Canvas(
        plot_width=len(data_df.columns) * 100,
        plot_height=1000,
        x_range=(-0.5, len(data_df.columns) - 0.5),
        y_range=(-0.5, len(data_df) - 0.5),
    )

    # Render directly as raster with no grid lines
    agg = cvs.raster(data_array)

    # Get value range for colormap
    vmin, vmax = np.nanpercentile(data_df.values, [5, 95])
    max_abs = max(abs(vmin), abs(vmax))

    # Convert matplotlib colormap to datashader format, didn't like 'coolwarm'
    if isinstance(cmap, str):
        mpl_cmap = cm.get_cmap(cmap)
        colors_list = [mpl_cmap(i)[:3] for i in np.linspace(0, 1, 256)]
        ds_cmap = colors_list
    else:
        ds_cmap = cmap

    # Create image with proper colormap and no grid lines
    img = tf.shade(agg, cmap=ds_cmap, how="linear", span=[-max_abs, max_abs])

    # Plot the image with proper extent
    artist = ax.imshow(
        np.array(img),
        aspect="auto",
        extent=(-0.5, len(data_df.columns) - 0.5, len(data_df) - 0.5, -0.5),
        interpolation="nearest",
    )

    # Add colorbar
    norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.01)

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
