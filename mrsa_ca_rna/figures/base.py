"""
Sets up the base figure upon which subplots will be placed in a manuscript style.

To-do:
    Graph PC2 vs. PC3, 4, 5, 6, etc. to remove PC1 (batch) from analysis.
    Graph more PCs.
    Make heatmap of ~40 components annotated by cohort to try to see patterns.
        Jackson has heatmap code I can use to learn from.
    Make a figure detailing results of a logistical regression model for MRSA outcome.
        Rgression file and functions must be written.
"""

import importlib
import logging
import math
import sys
import time
from os.path import join

import matplotlib
import matplotlib.figure
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt

# C++ anti-grain geometry backend
matplotlib.use("AGG")

# plot area styles
matplotlib.rcParams["grid.linestyle"] = "dotted"

# legend styles
matplotlib.rcParams["legend.borderpad"] = 0.25
matplotlib.rcParams["legend.framealpha"] = 0.75
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.handletextpad"] = 0.25
matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.markerscale"] = 0.7

# svg handling
matplotlib.rcParams["svg.fonttype"] = "none"

# axes labels
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["ytick.minor.pad"] = 0.9


def calculate_layout(num_plots, scale_factor=4):
    """
    Calculate appropriate layout and figure size based on number of plots.
    Parameters:
    -----------
    num_plots : int
        Number of plots to be displayed
    scale_factor : float, optional
        Scale factor for figure size (default: 4)
    Returns:
    --------
    layout : dict
        Dictionary with 'nrows' and 'ncols'
    fig_size : tuple
        Figure size (width, height)
    """

    if num_plots <= 0:
        raise ValueError("Number of plots must be a positive integer.")
    if scale_factor <= 0:
        raise ValueError("Scale factor must be a positive number.")

    # Calculate number of rows and columns (aim for square-ish layout)
    # Prefer more columns than rows if not perfectly square
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    # Calculate figure size based on scale factor
    width = cols * scale_factor
    height = rows * scale_factor

    layout = {"ncols": cols, "nrows": rows, "num_plots": num_plots}
    fig_size = (width, height)

    return fig_size, layout


def setupBase(figsize, gridd):
    """
    Sets up base figure for plotting subplots

    Accepts:
        figsize (tuple): size of figure in inches
        gridd (dict): subplot dimensions and number of plots
        style (str): graph style (default: 'whitegrid')

    Returns:
        ax (list of matplotlib.figure): list subplots contained in overall figure
        f (matplotlib.figure): figure object with subplots
        gs (gridspec.GridSpec): gridded subplots present on figure
    """
    sns.set_theme(style="whitegrid", font_scale=1.2, color_codes=True, palette="tab20")

    # Seaborn set_theme overrides some plt.rcParams, so we need to reset them
    matplotlib.rcParams["font.family"] = ["sans-serif"]
    matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["axes.labelsize"] = 11
    matplotlib.rcParams["axes.linewidth"] = 0.7
    matplotlib.rcParams["axes.titlesize"] = 11
    matplotlib.rcParams["legend.fontsize"] = 9
    matplotlib.rcParams["xtick.labelsize"] = 9
    matplotlib.rcParams["ytick.labelsize"] = 9

    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(nrows=gridd["nrows"], ncols=gridd["ncols"], figure=f)

    # Get the number of actual plots needed
    num_plots = gridd.get("num_plots", gridd["nrows"] * gridd["ncols"])

    ax = list()
    for x in range(num_plots):
        ax.append(f.add_subplot(gs[x]))

    # Remove any unused subplot spaces to prevent layout issues
    total_subplots = gridd["nrows"] * gridd["ncols"]
    if num_plots < total_subplots:
        for x in range(num_plots, total_subplots):
            # Create the subplot and immediately remove it
            unused_ax = f.add_subplot(gs[x])
            unused_ax.remove()

    return ax, f, gs


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    # Dynamically import the figure module based on the name provided
    module_name = f"mrsa_ca_rna.figures.{nameOut}"
    figure_module = importlib.import_module(module_name)

    # Run the figure generation function from the imported module
    figures = figure_module.genFig()

    # Check if multiple figures were returned
    if isinstance(figures, list | tuple):
        for i, fig in enumerate(figures):
            fig_name = f"{nameOut}_{i + 1}"
            fig.savefig(
                join(fdir, f"{fig_name}.svg"), bbox_inches="tight", pad_inches=0.1
            )
            logging.info(f"Saved figure {i + 1} as {fig_name}.svg")
    else:
        # Single figure returned
        figures.savefig(
            join(fdir, f"{nameOut}.svg"), bbox_inches="tight", pad_inches=0.1
        )
        logging.info(f"Saved figure as {nameOut}.svg")

    logging.info("%s is done after %s seconds.", nameOut, time.time() - start)
