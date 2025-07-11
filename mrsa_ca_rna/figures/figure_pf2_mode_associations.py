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
from mrsa_ca_rna.utils import concat_datasets, find_top_genes_by_threshold, calculate_cpm


def analyze_mode_associations(X: ad.AnnData, threshold_pct=0.25):
    """
    Analyze associations between diseases and genes for each component.

    Parameters
    ----------
    X : ad.AnnData
        AnnData object with PARAFAC2 results
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


def figure_setup():
    """Set up the data for analysis"""
    rank = 5
    X = concat_datasets(filter_threshold=5, min_pct=.9)

    X, r2x = perform_parafac2(X, slice_col="disease", rank=rank)

    # Use the new function to get top genes by threshold
    gene_factors_df = pd.DataFrame(
        X.varm["Pf2_C"], 
        index=X.var_names,
        columns=[f"Component_{i+1}" for i in range(X.uns["Pf2_A"].shape[1])]
    )
    
    # Get top genes by threshold, separated by positive and negative loadings
    top_genes_df = find_top_genes_by_threshold(gene_factors_df, threshold_fraction=0.5)
    
    # Traditional analysis for diseases
    results = analyze_mode_associations(X, threshold_pct=0.25)

    return X, results, r2x, top_genes_df


def genFig():
    """Generate figures showing the associations between diseases and genes"""
    X, results, r2x, top_genes_df = figure_setup()

    # Create output directory
    output_dir = "output/pf2_associations"
    os.makedirs(output_dir, exist_ok=True)

    # Create a multi-panel figure with A matrix + component details
    components_to_show = [1, 2, 3, 4, 5]

    # Single column layout - one row for A matrix, one row per component
    nrows = len(components_to_show) + 1
    layout = {"ncols": 1, "nrows": nrows}
    fig_size = (18, nrows * 5)  # Wider to accommodate the three panels side by side
    ax, f, gs = setupBase(fig_size, layout)

    # A matrix at the top spanning the full width
    ax_heatmap = ax[0]

    # Get components and disease labels
    ranks_labels = [str(x) for x in range(1, X.uns["Pf2_A"].shape[1] + 1)]
    disease_labels = list(X.obs["disease"].unique().astype(str))

    sns.heatmap(
        X.uns["Pf2_A"],
        ax=ax_heatmap,
        cmap="coolwarm",
        center=0,
        xticklabels=ranks_labels,
        yticklabels=disease_labels,
    )
    ax_heatmap.set_title(f"Disease Factor Matrix (A) - R2X: {r2x:.2f}")
    ax_heatmap.set_xlabel("Component")
    ax_heatmap.set_ylabel("Disease")

    for i, comp_num in enumerate(components_to_show):
        # Each component gets its own row
        ax_i = ax[i + 1]
        
        # Split the subplot area for diseases, positive genes, and negative genes
        divider = make_axes_locatable(ax_i)
        ax_disease = ax_i
        ax_pos_gene = divider.append_axes("right", size="100%", pad=1.5)
        ax_neg_gene = divider.append_axes("right", size="100%", pad=1.5)
        
        # Filter the top_genes_df for the current component
        comp_key = f"Component_{comp_num}"
        comp_genes = top_genes_df[top_genes_df["component"] == comp_key]
        
        # Get the total count of genes in each direction
        total_pos_genes = len(comp_genes[comp_genes["direction"] == "positive"])
        total_neg_genes = len(comp_genes[comp_genes["direction"] == "negative"])
        
        # Separate positive and negative genes for display
        pos_genes = comp_genes[comp_genes["direction"] == "positive"].head(10)
        neg_genes = comp_genes[comp_genes["direction"] == "negative"].head(10)
        
        # Make dataframe for diseases of this component
        diseases_df = pd.DataFrame(
            results[comp_num]["diseases"], columns=["Disease", "Score"]
        )
        top_diseases = diseases_df.head(5)
        
        # Plot diseases
        sns.barplot(x="Score", y="Disease", data=top_diseases, ax=ax_disease)
        ax_disease.set_title(f"Component {comp_num}: Top Diseases")
        ax_disease.axvline(x=0, color="gray", linestyle="--")
        
        # Plot positive genes
        if not pos_genes.empty:
            sns.barplot(x="value", y="gene", data=pos_genes, ax=ax_pos_gene, color="red")
            ax_pos_gene.set_title(f"Positive Genes (n={total_pos_genes})")
            ax_pos_gene.axvline(x=0, color="gray", linestyle="--")
        else:
            ax_pos_gene.text(0.5, 0.5, "No positive genes found", 
                             ha='center', va='center', transform=ax_pos_gene.transAxes)
            ax_pos_gene.set_title("Positive Genes")
        
        # Plot negative genes
        if not neg_genes.empty:
            sns.barplot(x="value", y="gene", data=neg_genes, ax=ax_neg_gene, color="blue")
            ax_neg_gene.set_title(f"Negative Genes (n={total_neg_genes})")
            ax_neg_gene.axvline(x=0, color="gray", linestyle="--")
        else:
            ax_neg_gene.text(0.5, 0.5, "No negative genes found", 
                             ha='center', va='center', transform=ax_neg_gene.transAxes)
            ax_neg_gene.set_title("Negative Genes")
        
        # Ensure the axis limits are well-balanced
        for axis in [ax_pos_gene, ax_neg_gene]:
            if not axis.lines:  # Skip if no data plotted
                continue
            xlim = max(abs(min(axis.get_xlim()[0], 0)), abs(max(axis.get_xlim()[1], 0)))
            axis.set_xlim(-xlim, xlim)
        
        # Highlight this component in the A matrix
        rect = plt.Rectangle(
            (comp_num - 1, 0),
            1,
            X.uns["Pf2_A"].shape[0],
            fill=False,
            edgecolor="black",
            linewidth=2,
            transform=ax_heatmap.transData,
        )
        ax_heatmap.add_patch(rect)

    f.suptitle("PARAFAC2 Component Analysis", fontsize=16)

    # Create a separate figure for the histograms, clearly separating positive and negative genes
    if not pos_genes.empty or not neg_genes.empty:
        # Get top 5 genes from each direction to plot
        pos_genes_to_plot = pos_genes.head(10) if not pos_genes.empty else None
        neg_genes_to_plot = neg_genes.head(10) if not neg_genes.empty else None
        
        # Combine gene names for plotting
        genes_to_plot = []
        if pos_genes_to_plot is not None:
            genes_to_plot.extend(pos_genes_to_plot["gene"].tolist())
        if neg_genes_to_plot is not None:
            genes_to_plot.extend(neg_genes_to_plot["gene"].tolist())
        
        if genes_to_plot:
            g = plot_raw_kde(
                X, 
                comp_num, 
                genes_to_plot, 
                pos_genes_df=pos_genes_to_plot, 
                neg_genes_df=neg_genes_to_plot
            )
        else:
            g = None
    else:
        g = None

    return f, g


def plot_raw_kde(X: ad.AnnData, cmp, genes, pos_genes_df=None, neg_genes_df=None):
    """Plot histograms for raw expression data of selected genes, 
    distinguishing between positive and negative loading genes.
    
    Parameters
    ----------
    X : ad.AnnData
        AnnData object with expression data
    cmp : int
        Component number
    genes : list
        List of gene names to plot
    pos_genes_df : pd.DataFrame, optional
        DataFrame with positive loading genes
    neg_genes_df : pd.DataFrame, optional
        DataFrame with negative loading genes
    """
    if not genes:
        return None
        
    layout, fig_size = calculate_layout(len(genes), scale_factor=4)
    ax, f, gs = setupBase(fig_size, layout)

    exp = X.layers["raw"]
    exp = calculate_cpm(exp)
    
    # Create sets of positive and negative genes for quick lookup
    pos_genes_set = set(pos_genes_df["gene"].values) if pos_genes_df is not None else set()
    neg_genes_set = set(neg_genes_df["gene"].values) if neg_genes_df is not None else set()

    for i, gene in enumerate(genes):
        if i >= len(ax):
            break
            
        ax_i = ax[i]
        gene_idx = np.where(X.var.index == gene)[0]
        
        if len(gene_idx) == 0:
            ax_i.text(0.5, 0.5, f"Gene {gene} not found", ha='center', va='center')
            continue
            
        gene_data = exp[:, gene_idx].flatten()
        
        # Determine if the gene is positive or negative loading
        if gene in pos_genes_set:
            color = "red"
            direction = "Positive"
        elif gene in neg_genes_set:
            color = "blue"
            direction = "Negative"
        else:
            color = "gray"
            direction = "Unknown"

        # Plot histogram instead of KDE
        sns.histplot(
            gene_data, 
            ax=ax_i, 
            color=color, 
            alpha=0.7, 
            kde=False,
            bins=50,
            element="bars"
        )
        ax_i.set_title(f"{direction} Gene: {gene}")
        ax_i.set_xlabel("Expression Level (CPM)")
        ax_i.set_ylabel("Count")

    f.suptitle(f"Expression Histograms for Component {cmp}", fontsize=16)

    return f
