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

import matplotlib.figure
import seaborn as sns
import matplotlib
from matplotlib import gridspec, pyplot as plt
import svgutils.transform as st

# C++ anti-grain geometry backend 
matplotlib.use("AGG")

# axes styles
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.linewidth"] = 0.7
matplotlib.rcParams["axes.titlesize"] = 14

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

def setupBase(figsize, gridd, style="whitegrid"):
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
    sns.set_theme(
        style=style,
        font_scale=0.7,
        color_codes=True,
        palette="viridis",
        rc=plt.rcParams
    )

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
