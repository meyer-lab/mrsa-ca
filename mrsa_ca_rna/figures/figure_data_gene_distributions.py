"""
This file plots the distribution of gene expression across datasets to observe
any possible irregularities leading to biased factorization results.
"""

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import calculate_cpm, prepare_data


def get_data(data_type: str = "raw") -> ad.AnnData:
    X = prepare_data(filter_threshold=-1)

    if data_type == "raw":
        # Use raw expression data
        exp = np.asarray(X.layers["raw"])
    elif data_type == "cpm":
        # Use normalized expression data
        exp = np.asarray(X.X)
    else:
        # Default to raw data
        exp = np.asarray(X.layers["raw"])

    # Add read depth for each sample
    X.obsm["read_depth"] = np.asarray(exp.sum(axis=1)).flatten()

    # Add expression information for each gene
    X.varm["total_expression"] = np.asarray(exp.sum(axis=0)).flatten()
    X.varm["mean_expression"] = np.asarray(exp.mean(axis=0)).flatten()

    return X


def prepare_read_depth(X: ad.AnnData) -> pd.DataFrame:
    """Make a DataFrame with read depth and disease information for seaborn plotting."""
    df = pd.DataFrame(
        {
            "read_depth": X.obsm["read_depth"],
            "disease": X.obs["disease"],
        }
    )
    return df


def prepare_expression_freq(X: ad.AnnData, threshold=5) -> pd.DataFrame:
    """Make a DataFrame with expression frequency and mean expression
    for seaborn plotting."""

    cpm_data = calculate_cpm(np.asarray(X.layers["raw"]))

    expr_freq = np.mean(cpm_data > threshold, axis=0) * 100

    df = pd.DataFrame(
        {
            "mean_expression": np.asarray(X.varm["mean_expression"]).flatten(),
            "expression_frequency": expr_freq,
            "gene": X.var.index,
        }
    )

    return df


def prepare_top_genes_data(X: ad.AnnData, n_genes: int = 5) -> pd.DataFrame:
    """Extract expression data for the top N genes with highest mean expression."""
    # Get the top genes by mean expression
    mean_expr_array = np.asarray(X.varm["mean_expression"]).flatten()
    top_gene_indices = np.argsort(mean_expr_array)[-n_genes:][::-1]
    top_gene_names = X.var_names[top_gene_indices]

    # Get raw expression data
    raw_data = np.asarray(X.layers["raw"])

    # Create DataFrame
    df_list = []
    for i, gene_idx in enumerate(top_gene_indices):
        gene = top_gene_names[i]
        # Extract gene expression
        gene_expr = np.asarray(raw_data[:, gene_idx]).flatten()

        # Create a dataframe for this gene
        df_gene = pd.DataFrame(
            {"gene": gene, "expression": gene_expr, "disease": X.obs["disease"].values}
        )
        df_list.append(df_gene)

    # Concatenate all gene dataframes
    return pd.concat(df_list, ignore_index=True)


def genFig():
    """Plot read depth and expression frequency."""

    data_type = "raw"
    X = get_data(data_type=data_type)

    read_depth = prepare_read_depth(X)

    threshold = 5
    expression_freq = prepare_expression_freq(X, threshold=threshold)

    # Prepare top genes data for additional analysis
    top_genes_data = prepare_top_genes_data(X, n_genes=5)

    # Add a column to expression_freq to mark top genes
    expression_freq["is_top_gene"] = expression_freq["gene"].isin(
        top_genes_data["gene"]
    )

    # Set up plots
    layout = {"ncols": 2, "nrows": 2}
    fig_size = (8, 8)
    ax, f, gs = setupBase(fig_size, layout)

    # Create boxplot of read depth by disease
    a = sns.boxplot(
        data=read_depth,
        x="disease",
        y="read_depth",
        ax=ax[0],
    )
    a.set_title("Sample Quality Among Disease Studies")
    a.set_xlabel("Disease")
    a.set_ylabel("Read Depth (Total Counts)")
    a.set_yscale("log")
    plt.setp(a.get_xticklabels(), rotation=45, ha="right")

    # Create scatterplot of expression frequency vs mean expression
    b = sns.scatterplot(
        data=expression_freq,
        x="mean_expression",
        y="expression_frequency",
        color="lightgray",
        alpha=0.5,
        ax=ax[1],
    )

    # Filter for top genes
    top_genes_expr_freq = expression_freq.loc[expression_freq["is_top_gene"], :]
    if not top_genes_expr_freq.empty:
        b = sns.scatterplot(
            data=top_genes_expr_freq,
            x="mean_expression",
            y="expression_frequency",
            hue="gene",
            ax=ax[1],
        )
    b.set_title(f"Gene Expression Profile with CPM>{threshold}")
    b.set_xlabel(f"Mean Expression ({data_type.upper()} Counts)")
    b.set_ylabel("Expression Frequency (%)")
    b.set_xscale("log")

    # Span the last two axes for top genes expression
    f.delaxes(ax[2])
    f.delaxes(ax[3])
    ax[2] = f.add_subplot(gs[1, :])

    # Create a boxplot for top genes expression with smaller outlier circles
    c = sns.boxplot(
        data=top_genes_data,
        x="disease",
        y="expression",
        hue="gene",
        ax=ax[2],
        flierprops=dict(marker="o", markersize=2),
    )
    c.set_title("Top Genes Expression Across Diseases")
    c.set_xlabel("Disease")
    c.set_ylabel(f"Expression ({data_type.upper()} Counts)")
    c.set_yscale("log")
    plt.setp(c.get_xticklabels(), rotation=45, ha="right")
    return f
