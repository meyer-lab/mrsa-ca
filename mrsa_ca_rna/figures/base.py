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

import logging
import sys
import time

import matplotlib
import matplotlib.figure
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt

# C++ anti-grain geometry backend
matplotlib.use("AGG")

# axes styles
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.linewidth"] = 0.7
matplotlib.rcParams["axes.titlesize"] = 11

# font styles
matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
matplotlib.rcParams["font.size"] = 10

# plot area styles
matplotlib.rcParams["grid.linestyle"] = "dotted"

# legend styles
matplotlib.rcParams["legend.borderpad"] = 0.25
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["legend.framealpha"] = 0.75
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.handletextpad"] = 0.25
matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.markerscale"] = 0.7

# svg handling
matplotlib.rcParams["svg.fonttype"] = "none"

# axes labels
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.labelsize"] = 8
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
    import math

    # Calculate number of rows and columns (aim for square-ish layout)
    # Prefer more columns than rows if not perfectly square
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    # Calculate figure size based on scale factor
    width = cols * scale_factor
    height = rows * scale_factor

    layout = {"ncols": cols, "nrows": rows}
    fig_size = (width, height)

    return layout, fig_size


def setupBase(figsize, gridd):
    """
    Sets up base figure for plotting subplots

    Accepts:
        figsize (tuple): size of figure in inches
        gridd (ndarray): subplot dimensions in rows and columns
        style (str): graph style (default: 'whitegrid')

    Returns:
        ax (list of matplotlib.figure): list subplots contained in overall figure
        f (matplotlib.figure): figure object with subplots
        gs (gridspec.GridSpec): gridded subplots present on figure
    """
    sns.set_theme(style="whitegrid", font_scale=0.7, color_codes=True, palette="turbo")

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
    gs = gridspec.GridSpec(**gridd, figure=f)

    x = 0
    ax = list()
    while x < gridd["nrows"] * gridd["ncols"]:
        ax.append(
            f.add_subplot(
                gs[x],
            )
        )
        x += 1

    return ax, f, gs


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from mrsa_ca_rna.figures import " + nameOut)
    ff = eval(nameOut + ".genFig()")
    ff.savefig(fdir + nameOut + ".svg", bbox_inches="tight", pad_inches=0.1)

    logging.info("%s is done after %s seconds.", nameOut, time.time() - start)
