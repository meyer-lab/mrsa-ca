import logging
from os.path import join

import anndata as ad
import gseapy as gp
import pandas as pd
from gseapy.plot import dotplot, gseaplot2


def gsea_analysis_per_cmp(
    X: ad.AnnData,
    cmp: int,
    term_ranks=5,
    gene_set="GO_Biological_Process_2023",
    figsize=(6, 4),
    out_dir=None,
):
    """Performs a preranked GSEA analysis on a component of a tensor factorization.

    Parameters
    ----------
    X : ad.AnnData
        Pre-factored AnnData object with the component ranks in varm["Pf2_C"]
    cmp : int
        Component to analyze
    term_ranks : int, optional
        Number of GO terms to include, by default 5
    gene_set : str, optional
        Gene set to use as background, by default "GO_Biological_Process_2023"
    figsize : tuple[int, int], optional
        Size of figure in inches_tall x inches_wide, by default (6, 4)
    out_dir : str, optional
        Output directory for figures, by default None

    Returns
    -------
    tuple[plt.Figure, plt.Figure]
        gsea plot and dotplot figures (do not play well with DIY plotting)
    """
    term_ranks = slice(0, term_ranks)

    # make a two column dataframe for prerank
    df = pd.DataFrame([])
    df["Gene"] = X.var.index
    df["Rank"] = X.varm["Pf2_C"][:, cmp - 1]  # type: ignore
    df = df.sort_values("Rank", ascending=False).reset_index(drop=True)

    # run the analysis and extract the results
    pre_res = gp.prerank(rnk=df, gene_sets=gene_set, seed=0)

    # check if the analysis ran successfully
    assert isinstance(pre_res.res2d, pd.DataFrame), "GSEA analysis failed to run"
    assert isinstance(pre_res.ranking, pd.Series), "GSEA analysis failed to run"

    # collect the results
    terms = pre_res.res2d.Term[term_ranks]
    hits = [pre_res.results[t]["hits"] for t in terms]
    runes = [pre_res.results[t]["RES"] for t in terms]

    # Generate titles with component info
    dotplot_title = f"Component {cmp} Dotplot"

    gsea_ofname = join(out_dir, f"gsea_{cmp}.svg") if out_dir is not None else None
    dotplot_ofname = (
        join(out_dir, f"dotplot_{cmp}.svg") if out_dir is not None else None
    )

    fig_g = gseaplot2(
        terms=terms,
        RESs=runes,
        hits=hits,
        rank_metric=pre_res.ranking.to_list(),
        figsize=figsize,
        ofname=gsea_ofname,
    )

    # dotplot will fail if there are no genes within the cutoff but we want it anyway
    try:
        fig_d = dotplot(
            pre_res.res2d,
            column="FDR q-val",
            title=dotplot_title,
            cmap="viridis",
            ofname=dotplot_ofname,
        )
    except ValueError:
        # make a dotplot with no cutoff with title indicating it failed
        logging.warning("Dotplot failed to generate. No genes within cutoff.")
        fig_d = dotplot(
            pre_res.res2d,
            column="FDR q-val",
            title="None w/in cutoff, showing all",
            cutoff=1,
            cmap="viridis",
            ofname=dotplot_ofname,
        )

    return fig_g, fig_d
