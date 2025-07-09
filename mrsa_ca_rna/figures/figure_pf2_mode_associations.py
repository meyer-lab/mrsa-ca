"""
This file analyzes associations between diseases and genes for each component
in the PARAFAC2 decomposition by directly comparing the A (disease) and C (gene) modes.
"""

import os
from math import ceil

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import concat_datasets


def analyze_mode_associations(X: ad.AnnData, threshold_pct=0.25):
    """
    Analyze associations between diseases and genes for each component.

    Parameters
    ----------
    X : ad.AnnData
        AnnData object with PARAFAC2 results
    top_n : int
        Number of top genes/diseases to report
    threshold_pct : float
        Threshold as percentage of max value to consider significant

    Returns
    -------
    dict
        Dictionary with analysis results for each component
    """
    # Get the disease and gene factors
    disease_factors = np.asarray(X.uns["Pf2_A"])
    gene_factors = np.asarray(X.varm["Pf2_C"])

    # Get disease and gene names
    disease_names = np.array(X.obs["disease"].unique())
    gene_names = np.array(X.var.index)

    # Number of components
    n_components = disease_factors.shape[1]

    results = {}

    for k in range(n_components):
        # Get factors for this component
        disease_factor = disease_factors[:, k]
        gene_factor = gene_factors[:, k]

        # Find maximum absolute values
        disease_max = np.max(np.abs(disease_factor))
        gene_max = np.max(np.abs(gene_factor))

        # Determine thresholds
        disease_threshold = disease_max * threshold_pct
        gene_threshold = gene_max * threshold_pct

        # Get indices of significant diseases and genes
        sig_disease_idx = np.where(np.abs(disease_factor) >= disease_threshold)[0]
        sig_gene_idx = np.where(np.abs(gene_factor) >= gene_threshold)[0]

        # Get names and values
        sig_diseases = [(disease_names[i], disease_factor[i]) for i in sig_disease_idx]
        sig_genes = [(gene_names[i], gene_factor[i]) for i in sig_gene_idx]

        # Sort by absolute value, then keep sign
        sig_diseases.sort(key=lambda x: abs(x[1]), reverse=True)
        sig_genes.sort(key=lambda x: abs(x[1]), reverse=True)

        # Store results
        results[k + 1] = {  # 1-indexed components
            "diseases": sig_diseases,
            "genes": sig_genes,
            "disease_threshold": disease_threshold,
            "gene_threshold": gene_threshold,
        }

    return results


def summarize_results(results: dict) -> pd.DataFrame:
    summary_rows = []
    for comp_num, comp_data in results.items():
        top_diseases = ", ".join(
            [f"{d[0]} ({d[1]:.2f})" for d in comp_data["diseases"][:5]]
        )
        top_genes = ", ".join([f"{g[0]}" for g in comp_data["genes"][:300]])

        summary_rows.append(
            {
                "Component": comp_num,
                "Top Diseases": top_diseases,
                "Top Genes": top_genes,
                "# Sig. Diseases": len(comp_data["diseases"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return summary_df


def create_component_heatmap(X: ad.AnnData, component, top_n=10) -> Figure:
    """
    Create a heatmap visualization showing how top genes relat
    to top diseases for a component.

    Parameters
    ----------
    X : ad.AnnData
        AnnData object with PARAFAC2 results
    component : int
        Component index (1-based)
    top_n : int
        Number of top genes/diseases to include

    Returns
    -------
    Figure
        Matplotlib figure with the heatmap
    """
    # Adjust to 0-based indexing
    k = component - 1

    # Get factors for this component
    disease_factor = X.uns["Pf2_A"][:, k]
    gene_factor = X.varm["Pf2_C"][:, k]

    # Get disease and gene names
    disease_names = np.array(X.obs["disease"].unique())
    gene_names = np.array(X.var.index)

    # Get indices of top diseases and genes by absolute value
    top_disease_idx = np.argsort(np.abs(disease_factor))[::-1][:top_n]
    top_gene_idx = np.argsort(np.abs(gene_factor))[::-1][:top_n]

    # Get names and values
    top_diseases = [(disease_names[i], disease_factor[i]) for i in top_disease_idx]
    top_genes = [(gene_names[i], gene_factor[i]) for i in top_gene_idx]

    # Create association matrix (outer product of factors)
    disease_values = np.array([d[1] for d in top_diseases])
    gene_values = np.array([g[1] for g in top_genes])
    association_matrix = np.outer(disease_values, gene_values)

    # Create DataFrame for heatmap
    disease_labels = [f"{d[0]} ({d[1]:.2f})" for d in top_diseases]
    gene_labels = [f"{g[0]} ({g[1]:.2f})" for g in top_genes]

    assoc_df = pd.DataFrame(
        association_matrix, index=disease_labels, columns=gene_labels
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(assoc_df, cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title(f"Component {component}: Disease-Gene Associations")
    plt.tight_layout()

    return fig


def figure_setup():
    """Set up the data for analysis"""
    rank = 1
    X = concat_datasets(filter_threshold=1)

    # outliers = [
    #     "SRR22854005", "SRR22854037", "SRR22854038", "SRR22854058",
    #     "GSM5361028", "GSM3534389", "GSM3926766", "GSM3926810",
    #     "GSM3926774", "GSM3926857", "GSM7677818"
    # ]

    # # Remove cancer datasets to avoid chemotherapy bias?
    # X = X[~X.obs.index.isin(outliers)].copy()

    # # Re-Z
    # from sklearn.preprocessing import StandardScaler
    # X.X = StandardScaler().fit_transform(X.X)

    X, r2x = perform_parafac2(X, slice_col="disease", rank=rank)

    # Analyze associations between diseases and genes
    results = analyze_mode_associations(X, threshold_pct=0.25)

    return X, results, r2x


def plot_raw_kde(X: ad.AnnData, cmp, genes):
    layout, fig_size = calculate_layout(len(genes), scale_factor=4)
    ax, f, gs = setupBase(fig_size, layout)

    X_raw = X.layers["raw"]

    for i, gene in enumerate(genes):
        ax_i = ax[i]
        gene_data = X_raw[:, X.var.index == gene].flatten()

        # Plot KDE for this gene
        sns.kdeplot(gene_data, ax=ax_i, fill=True, color="blue", alpha=0.5)
        ax_i.set_title(f"Raw Data KDE for {gene}")
        ax_i.set_xlabel("Expression Level")
        ax_i.set_ylabel("Density")

    f.suptitle(f"Raw Data KDEs for Component {cmp}", fontsize=16)

    return f


def genFig():
    """Generate figures showing the associations between diseases and genes"""
    X, results, r2x = figure_setup()

    # Create output directory
    output_dir = "output/pf2_associations"
    os.makedirs(output_dir, exist_ok=True)

    # Create summary DataFrame and save to CSV
    summary_df = summarize_results(results)
    summary_df.to_csv(os.path.join(output_dir, "component_summary.csv"), index=False)

    # Create a multi-panel figure with A matrix + component details
    components_to_show = [1]

    nrows = ceil(len(components_to_show) / 2) + 1  # +1 for the A matrix row
    layout = {"ncols": 2, "nrows": nrows}
    fig_size = (15, nrows * 4)
    ax, f, gs = setupBase(fig_size, layout)

    # Delete the first two axes to make space for the A matrix
    f.delaxes(ax[0])
    f.delaxes(ax[1])
    ax[0] = f.add_subplot(gs[0, :])

    # Get components and disease labels
    ranks_labels = [str(x) for x in range(1, X.uns["Pf2_A"].shape[1] + 1)]
    disease_labels = list(X.obs["disease"].unique().astype(str))

    sns.heatmap(
        X.uns["Pf2_A"],
        ax=ax[0],
        cmap="coolwarm",
        center=0,
        xticklabels=ranks_labels,
        yticklabels=disease_labels,
    )
    ax[0].set_title(f"Disease Factor Matrix (A) - R2X: {r2x:.2f}")
    ax[0].set_xlabel("Component")
    ax[0].set_ylabel("Disease")

    for i, comp_num in enumerate(components_to_show):
        # Create and save detailed heatmap for this component
        heatmap_fig = create_component_heatmap(X, comp_num, top_n=25)
        heatmap_fig.savefig(
            os.path.join(output_dir, f"component_{comp_num}_heatmap.png"), dpi=300
        )
        plt.close(heatmap_fig)

        # Make dataframes for diseases and genes of a component
        diseases_df = pd.DataFrame(
            results[comp_num]["diseases"], columns=["Disease", "Score"]
        )
        genes_df = pd.DataFrame(results[comp_num]["genes"], columns=["Gene", "Score"])

        # Determine axis index for this component (offset by 2 to account for first row)
        ax_i = ax[i + 2]

        # Create a horizontal bar plot for top 5 diseases and genes
        top_diseases = diseases_df.head(5)
        top_genes = genes_df.head(15)

        # Split the subplot area with diseases on the left and genes on the right
        divider = make_axes_locatable(ax_i)
        ax_disease = ax_i
        ax_gene = divider.append_axes("right", size="100%", pad=1.75)

        # Plot diseases
        sns.barplot(x="Score", y="Disease", data=top_diseases, ax=ax_disease)
        ax_disease.set_title(f"Component {comp_num}: Top Diseases")
        ax_disease.axvline(x=0, color="gray", linestyle="--")

        # Calculate the number of significant genes at 50% threshold
        gene_scores = np.array([g[1] for g in results[comp_num]["genes"]])
        max_abs_gene = np.max(np.abs(gene_scores))
        threshold_50pct = max_abs_gene * 0.5
        genes_above_threshold = np.sum(np.abs(gene_scores) >= threshold_50pct)

        # Plot gene barplot
        sns.barplot(x="Score", y="Gene", data=top_genes, ax=ax_gene)
        ax_gene.set_title(
            f"Component {comp_num}\nTop Genes (n={genes_above_threshold} at >50% max)"
        )
        ax_gene.axvline(x=0, color="gray", linestyle="--")

        # Consistently plot 0 centered gene scores
        x_min, x_max = ax_gene.get_xlim()
        max_abs = max(abs(x_min), abs(x_max))
        ax_gene.set_xlim(-max_abs, max_abs)

        ax_gene.tick_params(axis="y", which="major", pad=10)

        # Plot KDEs for raw data of top genes
        g = plot_raw_kde(X, comp_num, top_genes["Gene"])

        # Highlight this component in the A matrix
        rect = plt.Rectangle(
            (comp_num - 1, 0),
            1,
            X.uns["Pf2_A"].shape[0],
            fill=False,
            edgecolor="black",
            linewidth=2,
            transform=ax[0].transData,
        )
        ax[0].add_patch(rect)

    f.suptitle("PARAFAC2 Component Analysis", fontsize=16)

    return f, g
