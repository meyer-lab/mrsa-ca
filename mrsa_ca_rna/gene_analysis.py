import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gseapy import dotplot
from gseapy.plot import gseaplot2

from mrsa_ca_rna.figures.base import setupBase


# Edit this function to match figure_pf2_*.py style
def gsea_analysis_per_cmp(
    X: ad.AnnData,
    cmp: int,
    term_ranks=slice(0, 5),
    gene_set="GO_Biological_Process_2023",
    ax=None,
    figsize=(10, 12),
):
    """Perform GSEA analysis and plot the results in a vertical layout.

    Parameters
    ----------
    X : ad.AnnData
        AnnData object with gene data and Pf2_C in varm
    cmp : int
        Component number (1-based indexing)
    term_ranks : slice or int, default slice(0, 5)
        Terms to include in the GSEA plot
    gene_set : str, default "GO_Biological_Process_2023"
        Gene set to use for enrichment analysis
    ax : matplotlib axes or None, default None
        If provided, should be a 2-element list/array of axes objects
        If None, new figure with axes will be created
    figsize : tuple, default (10, 12)
        Figure size if creating a new figure

    Returns
    -------
    tuple
        (figure, axes) containing the plots
    """
    # Create figure and axes if not provided
    if ax is None:
        layout = {"ncols": 1, "nrows": 2}
        ax, fig, _ = setupBase(figsize, layout)
    else:
        # If axes are provided, try to get the figure
        if isinstance(ax, (list, np.ndarray)):
            fig = ax[0].figure
        else:
            fig = ax.figure
            # Convert single axis to list for consistency
            ax = [ax]

    # make a two column dataframe for prerank (gene, rank)
    df = pd.DataFrame([])
    df["Gene"] = X.var.index
    df["Rank"] = X.varm["Pf2_C"][:, cmp - 1]

    # sort the dataframe by rank (most expressed genes first)
    df = df.sort_values("Rank", ascending=False).reset_index(drop=True)

    # run the analysis and extract the results
    pre_res = gp.prerank(rnk=df, gene_sets=gene_set, seed=0)
    terms = pre_res.res2d.Term[term_ranks]
    hits = [pre_res.results[t]["hits"] for t in terms]
    runes = [pre_res.results[t]["RES"] for t in terms]

    # Generate titles with component info
    dot_title = f"Component {cmp} Gene Enrichment Analysis\n{gene_set}"
    gsea_title = f"Component {cmp} GSEA Plot"

    # plot the dotplot at the top position
    dot_plot = dotplot(
        pre_res.res2d,
        column="FDR q-val",
        title=dot_title,
        cmap=plt.cm.viridis,
        top_term=10,
        show_ring=True,
        ax=ax[0],
    )

    # plot the GSEA plot at the bottom position
    gsea_plot = gseaplot2(
        terms=terms,
        RESs=runes,
        hits=hits,
        rank_metric=pre_res.ranking,
        ax=ax[1],
        title=gsea_title,
    )

    return fig, ax


# delete this and replace with figure function within TBD figure_pf2_*.py file
def generate_gsea_figure(X, components=[1, 2], gene_set="GO_Biological_Process_2023"):
    """Create a GSEA figure with multiple components in a grid.

    Parameters
    ----------
    X : ad.AnnData
        AnnData object with gene data
    components : list, default [1, 2]
        List of components to analyze (1-based indexing)
    gene_set : str, default "GO_Biological_Process_2023"
        Gene set to use for enrichment analysis

    Returns
    -------
    matplotlib.figure.Figure
        The complete figure
    """
    n_components = len(components)
    layout = {"ncols": 1 * n_components, "nrows": 2}
    figsize = (10, 10 * n_components)

    ax, fig, _ = setupBase(figsize, layout)

    for i, cmp in enumerate(components):
        # Get the two axes for this component
        component_axes = [ax[2 * i], ax[2 * i + 1]]

        # Create GSEA plots for this component
        gsea_analysis_per_cmp(X, cmp=cmp, gene_set=gene_set, ax=component_axes)

    fig.tight_layout()
    return fig
