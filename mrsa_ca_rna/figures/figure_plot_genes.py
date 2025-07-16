"""This file plots the gene factor matrix from the PF2 model."""

import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import concat_datasets, find_top_genes_by_threshold


def calculate_gene_coverage(top_genes_df: pd.DataFrame, total_genes):
    """Calculate coverage statistics for the gene space.

    Parameters
    ----------
    top_genes_df : pd.DataFrame
        DataFrame with columns 'gene' and 'component'
    total_genes : int
        Total number of genes in the dataset

    Returns
    -------
    dict
        Dictionary with coverage statistics
    str
        Formatted coverage report
    """
    # Convert DataFrame to dict format for calculation
    top_genes = {}
    for comp, group in top_genes_df.groupby("component"):
        top_genes[comp] = group["gene"].tolist()

    # Calculate gene space coverage
    all_component_genes = set()
    unique_genes_by_component = {}

    # First, find genes that appear in each component
    for _, genes in top_genes.items():
        all_component_genes.update(genes)

    # Then, find genes that uniquely appear in only one component
    for comp, genes in top_genes.items():
        other_components_genes = set()
        for other_comp, other_genes in top_genes.items():
            if other_comp != comp:
                other_components_genes.update(other_genes)

        unique_genes = set(genes) - other_components_genes
        unique_genes_by_component[comp] = unique_genes

    # Calculate coverage statistics
    total_covered = len(all_component_genes)
    coverage_percent = (total_covered / total_genes) * 100

    # Create coverage report string
    coverage_report = (
        f"Gene Space Coverage: {total_covered}/{total_genes} "
        f"genes ({coverage_percent:.1f}%)\n"
    )

    for comp, unique_genes in unique_genes_by_component.items():
        coverage_report += (
            f"Component {comp}: {len(top_genes[comp])} genes total, "
            f"{len(unique_genes)} unique\n"
        )

    coverage_stats = {
        "total_covered": total_covered,
        "coverage_percent": coverage_percent,
        "unique_genes_by_component": unique_genes_by_component,
        "all_component_genes": all_component_genes,
    }

    return coverage_stats, coverage_report


def prepare_plot_dataframe(mean_genes, top_genes_df):
    """Prepare dataframe for plotting gene expression by component.

    Parameters
    ----------
    mean_genes : pd.DataFrame
        DataFrame with mean expression values
    top_genes_df : pd.DataFrame
        DataFrame with columns 'gene' and 'component'

    Returns
    -------
    pd.DataFrame
        DataFrame ready for plotting
    """
    # Create a plot dataframe with component information
    plot_df = mean_genes.copy()
    plot_df["component"] = "All genes"

    # Count genes per component for labels
    gene_counts = top_genes_df.groupby("component").size().to_dict()

    # Add component information for top genes
    for _, row in top_genes_df.iterrows():
        gene = row["gene"]
        comp = row["component"]
        if gene in plot_df.index:
            plot_df.loc[gene, "component"] = (
                f"Component {comp} ({gene_counts[comp]} genes)"
            )

    # Reset index to use gene names as a column
    plot_df = plot_df.reset_index()
    plot_df = plot_df.rename(columns={"index": "gene"})

    # Sort by mean expression for better visualization
    plot_df = plot_df.sort_values("mean_expression", ascending=False)
    plot_df["rank"] = range(len(plot_df))

    return plot_df


def get_data():
    """Get the data for plotting the gene factor matrix."""
    X = concat_datasets(filter_threshold=5, min_pct=0.5)

    # Perform PARAFAC2 factorization
    X, _ = perform_parafac2(X, slice_col="disease", rank=1)
    genes_df = pd.DataFrame(
        X.varm["Pf2_C"],
        index=X.var.index,
        columns=pd.Index([str(x) for x in range(1, X.uns["Pf2_A"].shape[1] + 1)]),
    )

    # # Grab latest gene matrix if not running the factorization
    # genes_df = pd.read_csv("output/pf2_genes_5.csv", index_col=0, header=0)

    # Calculate mean expression
    mean_exp = np.mean(X.layers["raw"], axis=0)
    mean_genes_df = pd.DataFrame(
        mean_exp,
        index=X.var.index,
        columns=["mean_expression"],
    )

    return mean_genes_df, genes_df


def genFig():
    """Generate the figure for the gene factor matrix from the PF2 model."""

    # Get the data
    mean_genes, genes_df = get_data()

    # Get top genes for each component
    top_genes_df = find_top_genes_by_threshold(genes_df, threshold_fraction=0.5)

    # Calculate gene space coverage
    total_genes = len(mean_genes)
    coverage_stats, coverage_report = calculate_gene_coverage(top_genes_df, total_genes)

    # Prepare data for plotting
    plot_df = prepare_plot_dataframe(mean_genes, top_genes_df)

    # Create figure and axes
    layout, _ = calculate_layout(num_plots=2, scale_factor=4)
    fig_size = (12, 10)
    ax, f, _ = setupBase(fig_size, layout)

    # Create a palette for components using seaborn's color_palette
    n_comps = len(top_genes_df["component"].unique())
    colors = sns.color_palette("tab20", n_comps)
    palette = {"All genes": "gray"}
    for i, comp in enumerate(set(c for c in plot_df["component"] if c != "All genes")):
        palette[comp] = colors[i % len(colors)]

    # Create the scatter plot
    a = sns.scatterplot(
        data=plot_df,
        x="rank",
        y="mean_expression",
        hue="component",
        size="component",
        sizes={"All genes": 5, **{c: 50 for c in palette if c != "All genes"}},
        alpha=0.7,
        palette=palette,
        ax=ax[0],
    )

    # Modify the plot styling
    a.set_xlabel("Genes (sorted by mean expression)")
    a.set_ylabel("Mean Expression (Raw Counts)")
    a.set_yscale("log")
    a.set_xticks([])

    # Adjust legend
    a.legend(title="Gene Groups", loc="upper right")

    # Add title with coverage information
    a.set_title(
        "Distribution of top genes by component across gene space\n"
        f"{coverage_report}\n"
        "Filtering: CPM>5 in at least 50% of samples\n"
        f"{n_comps} Component model"
    )

    # Combine mean expression and component weights
    df = genes_df.copy()
    df["mean_expression"] = mean_genes["mean_expression"]
    df = (
        df.reset_index()
        .melt(
            id_vars=["index", "mean_expression"],
            var_name="component",
            value_name="component_weight",
        )
        .rename(columns={"index": "gene"})
    )

    # Calculate correlation metrics for each component
    corr_text = []
    for comp in df["component"].unique():
        comp_data: pd.DataFrame = df[df["component"] == comp]

        # Calculate linear correlation (Pearson)
        pearson_r = comp_data["component_weight"].corr(
            comp_data["mean_expression"], method="pearson"
        )

        # Calculate non-linear correlations
        spearman_r = comp_data["component_weight"].corr(
            comp_data["mean_expression"], method="spearman"
        )
        kendall_tau = comp_data["component_weight"].corr(
            comp_data["mean_expression"], method="kendall"
        )

        corr_text.append(
            f"Component {comp}:\n"
            f"Pearson r: {pearson_r:.3f} (linear)\n"
            f"Spearman ρ: {spearman_r:.3f} (monotonic)\n"
            f"Kendall τ: {kendall_tau:.3f} (ordinal)"
        )

    # Plot a scatter of all the components
    b = sns.scatterplot(
        data=df,
        x="component_weight",
        y="mean_expression",
        hue="component",
        palette=colors,
        ax=ax[1],
    )

    # Add correlation metrics as text
    corr_info = "\n\n".join(corr_text)
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax[1].text(
        0.05,
        0.95,
        corr_info,
        transform=ax[1].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    b.set_yscale("log")
    b.set_xlabel("Component Weight")
    b.set_ylabel("Mean Expression (Raw Counts)")
    b.set_title("Correlation between mean expression and component weights")

    return f
