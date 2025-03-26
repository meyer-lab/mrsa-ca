
import anndata as ad
import gseapy as gp
import pandas as pd
from gseapy.plot import gseaplot2


def gsea_analysis_per_cmp(
    X: ad.AnnData,
    cmp: int,
    term_ranks=5,
    gene_set="GO_Biological_Process_2023",
    figsize=(6, 4),
    ofname=None,
):
    """Perform GSEA analysis and plot the results in a vertical layout."""

    term_ranks = slice(0, 5)

    # make a two column dataframe for prerank
    df = pd.DataFrame([])
    df["Gene"] = X.var.index
    df["Rank"] = X.varm["Pf2_C"][:, cmp - 1]
    df = df.sort_values("Rank", ascending=False).reset_index(drop=True)

    # run the analysis and extract the results
    pre_res = gp.prerank(rnk=df, gene_sets=gene_set, seed=0)
    terms = pre_res.res2d.Term[term_ranks]
    hits = [pre_res.results[t]["hits"] for t in terms]
    runes = [pre_res.results[t]["RES"] for t in terms]

    # Generate titles with component info
    gsea_title = f"Component {cmp} GSEA Plot"

    fig = gseaplot2(
        terms=terms,
        RESs=runes,
        hits=hits,
        rank_metric=pre_res.ranking,
        figsize=figsize,
        title=gsea_title,
        ofname=ofname,
    )
    return fig
