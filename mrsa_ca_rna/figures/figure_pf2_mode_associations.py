"""
This file analyzes associations between diseases and genes for each component
in the PARAFAC2 decomposition by directly comparing the A (disease) and C (gene) modes.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets


def analyze_mode_associations(X, top_n=15, threshold_pct=0.5):
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
    disease_factors = X.uns["Pf2_A"]
    gene_factors = X.varm["Pf2_C"]
    
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
        
        # Take top N if needed
        if top_n > 0:
            sig_diseases = sig_diseases[:top_n]
            sig_genes = sig_genes[:top_n]
        
        # Store results
        results[k+1] = {  # 1-indexed components
            "diseases": sig_diseases,
            "genes": sig_genes,
            "disease_threshold": disease_threshold,
            "gene_threshold": gene_threshold
        }
    
    return results


def create_component_heatmap(X, component, top_n=10) -> Figure:
    """
    Create a heatmap visualization showing how top genes relate to top diseases for a component.
    
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
    
    assoc_df = pd.DataFrame(association_matrix, index=disease_labels, columns=gene_labels)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(assoc_df, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title(f"Component {component}: Disease-Gene Associations")
    plt.tight_layout()
    
    return fig


def figure_setup():
    """Set up the data for analysis"""
    rank = 10
    X = concat_datasets()
    X, r2x = perform_parafac2(X, slice_col="disease", rank=rank)
    
    # Analyze associations between diseases and genes
    results = analyze_mode_associations(X, top_n=15, threshold_pct=0.5)
    
    return X, results, r2x


def genFig():
    """Generate figures showing the associations between diseases and genes"""
    X, results, r2x = figure_setup()
    
    # Create output directory
    output_dir = "output/pf2_associations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary DataFrame
    summary_rows = []
    for comp_num, comp_data in results.items():
        top_diseases = ", ".join([f"{d[0]} ({d[1]:.2f})" for d in comp_data["diseases"][:3]])
        top_genes = ", ".join([f"{g[0]} ({g[1]:.2f})" for g in comp_data["genes"][:3]])
        
        summary_rows.append({
            "Component": comp_num,
            "Top Diseases": top_diseases,
            "Top Genes": top_genes,
            "# Sig. Diseases": len(comp_data["diseases"]),
            "# Sig. Genes": len(comp_data["genes"])
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "component_summary.csv"), index=False)
    
    # Create a multi-panel figure with A matrix + component details
    n_components_to_show = 4
    layout = {"ncols": 2, "nrows": 3}  # Updated to 3 rows: 1 for A matrix, 2 for components
    fig_size = (15, 18)  # Increased height to accommodate A matrix
    ax, f, _ = setupBase(fig_size, layout)
    
    # Plot the A matrix in the top row, spanning both columns
    f.delaxes(ax[1])  # Remove the second axis in the first row
    ax[0].set_position([0.1, 0.7, 0.8, 0.25])  # Adjust position to span both columns
    
    # Get components and disease labels
    ranks_labels = [str(x) for x in range(1, X.uns["Pf2_A"].shape[1] + 1)]
    disease_labels = list(X.obs["disease"].unique().astype(str))
    
    # Create heatmap for A matrix
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    warm_half = cmap(np.linspace(0.5, 1, 256))
    A_cmap = colors.ListedColormap(warm_half)
    
    sns.heatmap(
        X.uns["Pf2_A"],
        ax=ax[0],
        cmap=A_cmap,
        vmin=0,
        xticklabels=ranks_labels,
        yticklabels=disease_labels,
        annot=True,
        fmt=".2f"
    )
    ax[0].set_title(f"Disease Factor Matrix (A) - R2X: {r2x:.2f}")
    ax[0].set_xlabel("Component")
    ax[0].set_ylabel("Disease")
    
    # Plot component details in the remaining 4 axes
    for i in range(n_components_to_show):
        comp_num = i + 1
        # Create and save detailed heatmap for this component
        heatmap_fig = create_component_heatmap(X, comp_num, top_n=10)
        heatmap_fig.savefig(os.path.join(output_dir, f"component_{comp_num}_heatmap.png"), dpi=300)
        plt.close(heatmap_fig)
        
        # Save the top diseases and genes to CSV
        diseases_df = pd.DataFrame(results[comp_num]["diseases"], columns=["Disease", "Score"])
        genes_df = pd.DataFrame(results[comp_num]["genes"], columns=["Gene", "Score"])
        
        diseases_df.to_csv(os.path.join(output_dir, f"component_{comp_num}_diseases.csv"), index=False)
        genes_df.to_csv(os.path.join(output_dir, f"component_{comp_num}_genes.csv"), index=False)
        
        # Determine axis index for this component (offset by 2 to account for first row)
        ax_i = ax[i+2]
        
        # Create a horizontal bar plot for top 5 diseases and genes
        top_diseases = diseases_df.head(5)
        top_genes = genes_df.head(5)
        
        # Split the subplot area
        divider = make_axes_locatable(ax_i)
        ax_disease = ax_i
        ax_gene = divider.append_axes("right", size="100%", pad=0.1)
        
        # Plot diseases
        sns.barplot(x="Score", y="Disease", data=top_diseases, ax=ax_disease)
        ax_disease.set_title(f"Component {comp_num}: Top Diseases")
        ax_disease.axvline(x=0, color='gray', linestyle='--')
        
        # Plot genes
        sns.barplot(x="Score", y="Gene", data=top_genes, ax=ax_gene)
        ax_gene.set_title(f"Component {comp_num}: Top Genes")
        ax_gene.axvline(x=0, color='gray', linestyle='--')
        
        # Highlight this component in the A matrix
        rect = plt.Rectangle(
            (comp_num-1, -0.5), 1, X.uns["Pf2_A"].shape[0], 
            fill=False, edgecolor='black', linewidth=2, 
            transform=ax[0].transData
        )
        ax[0].add_patch(rect)
    
    f.suptitle(f"PARAFAC2 Component Analysis", fontsize=16)
    
    return f