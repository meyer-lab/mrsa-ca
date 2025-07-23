"""This file will plot random genes from the entire dataset to judge 
distributions. We are looking for multimodal distributions, which would indicate
hetegeneous expression either across the dataset or within a single study.

To do this, we will randomly select genes from the dataset and plot their expression
frequencies as histograms. Then we will plot genes identified as having multimodal
distributions from this set against each other organized by study (hue). If the 
genes are multimodal across only studies, we will see a single cluster of points per
study. If the genes are multimodal within a study, or worse in the same way in a 
study and across studies, we will see multiple clusters of points per study, and 
at worst, the same clusters of points across studies."""

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.figures.base import setupBase, calculate_layout
from mrsa_ca_rna.utils import concat_datasets, calculate_cpm

def get_data():
    X: ad.AnnData = concat_datasets(filter_threshold=5, min_pct=.9)

    # Select 20 random genes
    rng = np.random.default_rng(42)
    random_genes = rng.choice(X.var.index, size=20, replace=False)

    # Subset the data to these genes
    X_subset = X[:, random_genes].copy()
    X_subset.layers["cpm"] = calculate_cpm(X_subset.layers["raw"])

    X_subset.varm["mean_cpm"] = X_subset.layers["cpm"].mean(axis=0)

    # Create wide-form dataframe with CPM values
    cpm_df = pd.DataFrame(
        X_subset.layers["cpm"], 
        index=X_subset.obs.index, 
        columns=X_subset.var.index
    )
    
    # Add disease information
    cpm_df['Disease'] = X_subset.obs['disease']
    
    # Reset index to make Sample a column before melting
    cpm_df = cpm_df.reset_index().rename(columns={'index': 'Sample'})
    
    # Melt to long-form dataframe
    expr_df = cpm_df.melt(
        id_vars=['Sample', 'Disease'], 
        var_name='Gene', 
        value_name='CPM'
    )

    return expr_df, X_subset

def genFig():

    expr_df, X_subset = get_data()

    # set up base figure for random gene plotting
    layout, fig_size = calculate_layout(20, 5)
    ax_hist, f, _ = setupBase(fig_size, layout)

    # Plot all genes individually
    for i, gene in enumerate(X_subset.var.index):
        gene_data = expr_df.loc[expr_df["Gene"] == gene, "CPM"]
        mean_cpm = gene_data.mean()
        
        a = sns.histplot(
            gene_data,
            bins=50,
            alpha=0.7,
            ax=ax_hist[i],
            kde=True
        )
        a.set_title(f"{gene} (mean CPM: {mean_cpm:.2f})")
        a.set_xlabel("Counts (CPM)")
        a.set_ylabel("Frequency")


    # Create wide-form dataframe for easier gene-vs-gene plotting
    gene_matrix = expr_df.pivot(
        index=['Sample', 'Disease'], 
        columns='Gene', 
        values='CPM'
    ).reset_index()
    
    # Select pairs of genes to compare (first 10 pairs to avoid too many plots)
    genes = list(X_subset.var.index)
    gene_pairs = [(genes[i], genes[i+1]) for i in range(0, min(20, len(genes)-1), 2)]
    
    layout, fig_size = calculate_layout(len(gene_pairs), 5)
    ax_compare, g, _ = setupBase(fig_size, layout)

    # Plot pairwise comparison of genes across samples
    for i, (gene1, gene2) in enumerate(gene_pairs):
        a = sns.scatterplot(
            data=gene_matrix,
            x=gene1,
            y=gene2,
            hue="Disease",
            ax=ax_compare[i],
            alpha=0.7
        )
        a.set_title(f"{gene1} vs {gene2}")
        a.set_xlabel(f"{gene1} (CPM)")
        a.set_ylabel(f"{gene2} (CPM)")

    return f, g

