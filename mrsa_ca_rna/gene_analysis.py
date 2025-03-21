import anndata
import gseapy as gp
import matplotlib.pyplot as plt
import pandas as pd
from gseapy import dotplot
from gseapy.plot import gseaplot


def gsea_overrep_per_cmp(
    X: anndata,
    cmp: int,
    pos: bool = True,
    enrichr=True,
    gene_set="GO_Biological_Process_2023",
    output_file="output/figureS13b.svg",
):
    """Perform GSEA overrepresentation analysis and plot the results."""
    df = pd.DataFrame([])
    df["Gene"] = X.var.index
    df["Rank"] = X.varm["Pf2_C"][:, cmp - 1]
    df = df[df["Rank"] > 0] if pos else df[df["Rank"] < 0]

    df = df.sort_values("Rank").reset_index(drop=True)

    if enrichr is True:
        enr_up = gp.enrichr(df["Gene"].values.tolist(), gene_sets=gene_set)
        enr_up.res2d.Term = enr_up.res2d.Term.str.split(" \(GO").str[0]
        dotplot(enr_up.res2d, title=gene_set, cmap=plt.cm.viridis, ofname=output_file)
    else:
        pre_res = gp.prerank(rnk=df, gene_sets=gene_set, seed=0)
        dotplot(
            pre_res.res2d,
            column="FDR q-val",
            title=gene_set,
            cutoff=0.25,
            cmap=plt.cm.viridis,
            ofname=output_file,
        )


def gsea_analysis_per_cmp(
    X: anndata,
    cmp: int,
    term_rank=0,
    gene_set="GO_Biological_Process_2023",
    output_file="output/figureS13a.svg",
):
    """Perform GSEA analysis and plot the results."""
    df = pd.DataFrame([])
    df["Gene"] = X.var.index
    df["Rank"] = X.varm["Pf2_C"][:, cmp - 1]
    df = df.sort_values("Rank").reset_index(drop=True)
    pre_res = gp.prerank(rnk=df, gene_sets=gene_set, seed=0)

    out = []

    for term in list(pre_res.results):
        out.append(
            [
                term,
                pre_res.results[term]["fdr"],
                pre_res.results[term]["es"],
                pre_res.results[term]["nes"],
                pre_res.results[term]["pval"],
            ]
        )

    out_df = (
        pd.DataFrame(out, columns=["Term", "fdr", "es", "nes", "pval"])
        .sort_values(by=["nes", "es"], ascending=False)
        .reset_index(drop=True)
    )
    term_to_plot = out_df["Term"][term_rank]

    gseaplot(
        term=term_to_plot,
        hits=pre_res.results[term_to_plot]["hits"],
        nes=pre_res.results[term_to_plot]["nes"],
        pval=pre_res.results[term_to_plot]["pval"],
        fdr=pre_res.results[term_to_plot]["fdr"],
        RES=pre_res.results[term_to_plot]["RES"],
        rank_metric=pre_res.ranking,
        ofname=output_file,
    )

# DELETE:
# debug section
from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.utils import concat_datasets, gene_converter

# Load the data
disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
X = concat_datasets(disease_list, filter_threshold=4)
_, factors, _, _ = perform_parafac2(X, condition_name="disease", rank=20, l1=1e-4)
X = gene_converter(X, old_id="EnsemblGeneID", new_id="Symbol", method="columns")
X.varm["Pf2_C"] = factors[2]
gsea_overrep_per_cmp(X, 2, pos=True, enrichr=True)
gsea_analysis_per_cmp(X, 2, term_rank=0)