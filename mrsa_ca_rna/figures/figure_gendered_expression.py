"""Gene expression analysis and visualization by gender with statistical testing."""

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from mrsa_ca_rna.figures.base import setupBase, calculate_layout
from mrsa_ca_rna.import_data import import_gene_tiers, import_mrsa
from mrsa_ca_rna.utils import rpm_norm


def plot_gene_expression_by_gender(ax, data: ad.AnnData, gene_list, max_genes_to_plot=None):
    """
    Plot boxplots comparing gene expression between males and females
    for a given gene list and perform statistical testing with BH correction.
    """
    # Get genes that exist in the dataset
    genes = data.var.index.intersection(gene_list)
    
    print(f"Found {len(genes)} out of {len(gene_list)} genes in the dataset")
    
    # Assume raw counts are in data.X, calculate rpm normalization
    data.X = rpm_norm(data.X)

    # Trim data to only the genes of interest
    data = data[:, genes].copy()

    # Perform log2(CPM + 1) transformation
    log_expr_data = np.log2(data.X + 1)
    
    gene_expr = pd.DataFrame(log_expr_data,
                            index=data.obs.index, 
                            columns=genes)
    
    # Add gender information
    gender = data.obs["gender"].values
    gender_labels = []
    
    for g in gender:
        if isinstance(g, np.integer | int):
            gender_labels.append("Male" if g == 0 else "Female")
        else:
            gender_labels.append(str(g).capitalize())
    
    # Create a new DataFrame with gene expression and gender
    gene_expr_with_gender = gene_expr.copy()
    gene_expr_with_gender["Gender"] = gender_labels
    
    # Perform Mann-Whitney U tests for each gene
    p_values = []
    u_stats = []
    genes_tested = []
    male_means = []
    female_means = []
    male_medians = []
    female_medians = []
    
    for gene in genes:
        male_expr = gene_expr_with_gender[
            gene_expr_with_gender["Gender"] == "Male"
        ][gene]
        female_expr = gene_expr_with_gender[
            gene_expr_with_gender["Gender"] == "Female"
        ][gene]
        
        # Perform Mann-Whitney U test (also known as Wilcoxon rank-sum test)
        u_stat, p_val = stats.mannwhitneyu(
            male_expr, female_expr, 
            alternative='two-sided'  # For two-sided test
        )
        
        p_values.append(p_val)
        u_stats.append(u_stat)
        genes_tested.append(gene)
        male_means.append(male_expr.mean())
        female_means.append(female_expr.mean())
        male_medians.append(male_expr.median())
        female_medians.append(female_expr.median())
    
    # Apply Benjamini-Hochberg correction
    reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Create results DataFrame
    stats_results = pd.DataFrame({
        'Gene': genes_tested,
        'U-statistic': u_stats,
        'P-value': p_values,
        'P-adjusted': p_corrected,
        'Significant': reject,
        'Male_mean': male_means,
        'Female_mean': female_means,
        'Male_median': male_medians,
        'Female_median': female_medians,
        'Mean_difference': np.array(female_means) - np.array(male_means),
        'Median_difference': np.array(female_medians) - np.array(male_medians)
    })
    
    # Sort by significance for plotting
    stats_results = stats_results.sort_values('P-adjusted')
    
    # Only plot if ax is provided and we have genes to plot
    if ax is not None:
        # Determine which genes to plot
        if max_genes_to_plot is not None and len(genes) > max_genes_to_plot:
            # Plot most significant genes
            genes_to_plot = stats_results.head(max_genes_to_plot)['Gene'].tolist()
            print(f"Plotting top {max_genes_to_plot} most significant genes out of {len(genes)} total")
        else:
            genes_to_plot = genes
        
        # Convert to long format for seaborn (only for genes we're plotting)
        gene_expr_plot = gene_expr_with_gender[list(genes_to_plot) + ["Gender"]]
        gene_expr_long = pd.melt(
            gene_expr_plot,
            id_vars=["Gender"],
            value_vars=genes_to_plot,
            var_name="Gene",
            value_name="Expression"
        )
        
        # Create the boxplot
        sns.boxplot(
            data=gene_expr_long,
            x="Gene",
            y="Expression",
            hue="Gender",
            ax=ax,
            palette=["skyblue", "lightpink"]
        )
        
        # Add individual data points for more detail
        sns.stripplot(
            data=gene_expr_long,
            x="Gene",
            y="Expression",
            hue="Gender",
            dodge=True,
            alpha=0.3,
            size=3,
            ax=ax,
            palette=["navy", "darkred"]
        )
        
        # Adjust the plot
        ax.set_xlabel("Genes")
        ax.set_ylabel("Log2(CPM + 1)")
        
        # Keep only one legend (remove stripplot legend)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Gender", loc="upper right")
        
        # Add significance indicators and mean expression annotations
        y_max = gene_expr_long["Expression"].max()
        y_min = gene_expr_long["Expression"].min()
        y_range = y_max - y_min
        
        for i, gene in enumerate(gene_expr_long["Gene"].unique()):
            gene_stats = stats_results[stats_results["Gene"] == gene]
            if not gene_stats.empty:
                p_adj = gene_stats["P-adjusted"].values[0]
                
                # Add significance stars
                if p_adj < 0.001:
                    text = "***"
                elif p_adj < 0.01:
                    text = "**"
                elif p_adj < 0.05:
                    text = "*"
                else:
                    text = "ns"
                    
                ax.text(i, y_max + y_range*0.05, text, ha='center', fontweight='bold')
            
            # Add median expression annotation (more appropriate for Mann-Whitney U)
            gene_data = gene_expr_long[gene_expr_long["Gene"] == gene]
            male_median = gene_data[gene_data["Gender"] == "Male"]["Expression"].median()
            female_median = gene_data[gene_data["Gender"] == "Female"]["Expression"].median()
            
            # Show gender-specific medians in a compact format
            ax.text(i, y_min - y_range*0.15, 
                    f"M:{male_median:.1f}\nF:{female_median:.1f}", 
                    ha='center', fontsize=7, color='gray', va='top')
        
        # Add significance legend
        ax.text(0.5, 1.08, 
                "* p<0.05, ** p<0.01, *** p<0.001, ns: not significant (BH-corrected)",
                transform=ax.transAxes, ha='center', fontsize=9)
        
        # Add median expression legend (updated for Mann-Whitney U)
        ax.text(0.5, -0.20, 
                "M/F = Male/Female median log2(CPM+1) expression",
                transform=ax.transAxes, ha='center', fontsize=8, color='gray')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), ha="center")
    
    return stats_results


def genFig():
    """
    Create figures with gene expression boxplots for Tier 1, statistical analysis for both tiers,
    and histograms for Tier 1 genes.
    
    Returns:
    --------
    tuple
        Figure objects, axes objects, and statistical results
    """
    # Load data and gene tiers
    data = import_mrsa()
    tiers = import_gene_tiers()
    
    # Create figure for gene expression boxplots (only Tier 1)
    expr_fig_size = (20, 8)  # Wider to accommodate more genes
    expr_layout = {"ncols": 1, "nrows": 1}
    expr_ax, expr_f, _ = setupBase(expr_fig_size, expr_layout)
    expr_f.suptitle(
        "Tier 1 Gene Expression Comparison by Gender", 
        fontsize=16
    )
    
    stats_results = {}
    
    # Tier 1 gene expression analysis with plotting
    print("Analyzing Tier 1 genes...")
    tier1_stats = plot_gene_expression_by_gender(
        expr_ax[0], data, tiers["Tier 1"], max_genes_to_plot=25  # Limit to top 25 for visualization
    )
    expr_ax[0].set_title(f"Tier 1 Gene Expression by Gender (n={len(tiers['Tier 1'])} genes)")
    stats_results["Tier 1"] = tier1_stats
    
    # Tier 2 gene expression analysis (statistics only, no plotting)
    print("Analyzing Tier 2 genes (statistics only)...")
    tier2_stats = plot_gene_expression_by_gender(
        None, data, tiers["Tier 2"]  # No axis = no plotting
    )
    stats_results["Tier 2"] = tier2_stats
    
    # Create histogram figure for Tier 1 genes
    print("Creating histograms for Tier 1 genes...")
    
    # Get genes that exist in the dataset for Tier 1
    tier1_genes = data.var.index.intersection(tiers["Tier 1"])
    print(f"Creating histograms for {len(tier1_genes)} Tier 1 genes")
    
    
    # Create histogram figure
    num_plots = len(tier1_genes)
    layout, fig_size = calculate_layout(num_plots, 4)
    hist_ax, hist_f, hist_gs = setupBase(fig_size, layout)
    hist_f.suptitle(
        f"Tier 1 Gene Expression Distributions (n={len(tier1_genes)} genes)", 
        fontsize=16
    )
    
    # Prepare data for histograms
    data_copy = data.copy()
    data_copy.X = rpm_norm(data_copy.X)
    data_copy = data_copy[:, tier1_genes].copy()
    log_expr_data = np.log2(data_copy.X + 1)
    
    gene_expr_df = pd.DataFrame(log_expr_data,
                               index=data_copy.obs.index, 
                               columns=tier1_genes)
    
    # Plot histogram for each gene
    for i, gene in enumerate(tier1_genes):
    
        # Plot histogram
        gene_data = gene_expr_df[gene]
        
        # Create histogram
        b = sns.histplot(gene_data, ax=hist_ax[i], bins=30, alpha=0.7, color='steelblue', edgecolor='black')

        # Add statistics to the plot
        mean_expr = gene_data.mean()
        median_expr = gene_data.median()
        std_expr = gene_data.std()
        
        # Set title with gene name and basic stats
        b.set_title(f"{gene}\nMean: {mean_expr:.2f}, Std: {std_expr:.2f}")
        b.set_xlabel("Log2(CPM + 1)")
        b.set_ylabel("Frequency")

        # Add vertical lines for mean and median
        b.axvline(mean_expr, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_expr:.2f}')
        b.axvline(median_expr, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_expr:.2f}')
        
        # Add legend if there's space
        if i < 5:  # Only add legend to first row to avoid clutter
            b.legend(fontsize=8, loc='upper right')

        # Add grid for better readability
        b.grid(True, alpha=0.3)

    # Hide empty subplots if we have fewer genes than grid spaces
    total_subplots = len(hist_ax)
    if num_plots < total_subplots:
        for i in range(num_plots, total_subplots):
            hist_ax[i].set_visible(False)
    
    # Save results to CSV files
    print("Saving statistical results to CSV files...")
    
    # Save Tier 1 results
    tier1_output_path = "tier1_gender_expression_stats.csv"
    tier1_stats.to_csv(tier1_output_path, index=False)
    print(f"Tier 1 results saved to: {tier1_output_path}")
    
    # Save Tier 2 results
    tier2_output_path = "tier2_gender_expression_stats.csv"
    tier2_stats.to_csv(tier2_output_path, index=False)
    print(f"Tier 2 results saved to: {tier2_output_path}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    for tier_name, results in stats_results.items():
        n_significant = results['Significant'].sum()
        n_total = len(results)
        print(f"{tier_name}: {n_significant}/{n_total} genes significantly different between genders (p<0.05, BH-corrected)")
        
        if n_significant > 0:
            top_significant = results[results['Significant']].head(5)
            print(f"Top 5 most significant genes in {tier_name}:")
            for _, row in top_significant.iterrows():
                direction = "higher in females" if row['Mean_difference'] > 0 else "higher in males"
                print(f"  {row['Gene']}: p_adj={row['P-adjusted']:.2e}, {direction}")
        print()
    
    return expr_f, hist_f