"""This file will plot how genes are weighted by the strongest eigen state
in the latent space. Maybe looking at what genes are most strongly associated
with the strongest eigen state will help us understand the latent space better."""

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import calculate_cpm, find_top_features, prepare_data


def get_data() -> ad.AnnData:
    """Get the data for the figure."""
    X = prepare_data()

    X, _ = perform_parafac2(X)

    # Identify the strongest eigen state (row with largest sum across columns)
    strongest_eigenstate = np.sum(np.abs(np.asarray(X.uns["Pf2_B"])), axis=1).argmax()

    # Multiply the C matrix with the strongest eigen state
    weighted_genes = np.dot(X.varm["Pf2_C"], X.uns["Pf2_B"][strongest_eigenstate, :])

    X.varm["Pf2_weighted_genes"] = weighted_genes

    return X, strongest_eigenstate


def prepare_B_matrix(X: ad.AnnData) -> pd.DataFrame:
    """Prepare the B matrix for plotting."""
    B_matrix = X.uns["Pf2_B"]
    B_df = pd.DataFrame(
        B_matrix,
        index=[f"Eigenstate {i + 1}" for i in range(B_matrix.shape[0])],
        columns=[f"Component {i + 1}" for i in range(B_matrix.shape[1])],
    )
    return B_df


def prepare_weighted_genes(X: ad.AnnData) -> pd.DataFrame:
    """Prepare the weighted genes DataFrame for plotting."""
    weighted_genes = X.varm["Pf2_weighted_genes"]
    weighted_genes_df = pd.DataFrame(
        weighted_genes,
        index=X.var.index,
        columns=["Weighted Genes"],
    )

    # Find top genes by threshold
    top_genes = find_top_features(weighted_genes_df, threshold_fraction=0.75, feature_name="gene")
    top_genes.to_csv("output/eigen_genes.csv", index=True)

    return top_genes

def prepare_disease_expression(X: ad.AnnData, genes) -> pd.DataFrame:
    """
    Create a DataFrame of raw expression values for selected genes across samples,
    organized by associated diseases.
    """
    # Subset the expression matrix for the selected genes
    expr = X[:, genes].to_df()
    expr["disease"] = X.obs["disease"].values

    # Melt the DataFrame for plotting (long format: sample, gene, expression, disease)
    expr_long = expr.reset_index().melt(
        id_vars=["index", "disease"],
        value_vars=genes,
        var_name="gene",
        value_name="expression"
    )
    expr_long.rename(columns={"index": "sample"}, inplace=True)
    return expr_long


def plot_gene_expression_histograms(X: ad.AnnData, pos_genes: pd.DataFrame, neg_genes: pd.DataFrame, n_genes: int = 10):
    """Plot histograms for raw expression data of top positive and negative eigen genes."""
    
    # Get top genes from each direction
    top_pos_genes = pos_genes.head(n_genes)["gene"].tolist()
    top_neg_genes = neg_genes.head(n_genes)["gene"].tolist()
    all_genes = top_pos_genes + top_neg_genes
    
    if not all_genes:
        return None
    
    # Calculate layout and setup figure
    fig_size, layout = calculate_layout(len(all_genes), scale_factor=3)
    ax, f, _ = setupBase(fig_size, layout)
    
    # Get CPM expression data
    exp = calculate_cpm(X.layers["raw"])
    
    for i, gene in enumerate(all_genes):
        ax_i = ax[i]
        
        # Get expression data for this gene
        gene_idx = np.where(X.var.index == gene)[0][0]
        gene_data = exp[:, gene_idx]
        
        # Determine color and direction
        if gene in top_pos_genes:
            color = "red"
            direction = "Positive"
        else:  # Must be negative
            color = "blue" 
            direction = "Negative"
        
        # Plot histogram
        sns.histplot(
            gene_data,
            ax=ax_i,
            color=color,
            alpha=0.7,
            kde=False,
            bins=30,
            element="bars"
        )
        
        ax_i.set_title(f"{direction}: {gene}")
        ax_i.set_xlabel("Expression (CPM)")
        ax_i.set_ylabel("Count")
    
    f.suptitle("Expression Histograms for Top Eigen Genes", fontsize=16)
    return f

def genFig():
    """Generate the figure for the strongest eigen state and weighted genes."""

    # Get the data
    X, strongest_eigenstate = get_data()

    # Prepare the B matrix DataFrame
    B_df = prepare_B_matrix(X)

    # Prepare the weighted genes DataFrame
    weighted_genes_df = prepare_weighted_genes(X)

    # Number of genes to display
    n_genes = 10

    # Setup the figure
    layout = {"ncols": 3, "nrows": 3}
    fig_size = (18, 12)
    ax, f, gs = setupBase(fig_size, layout)
    
    # Delete row 2 and 3 axes, then make subplots spanning those rows
    for col in range(layout["ncols"]):
        f.delaxes(ax[3 + col])
        f.delaxes(ax[6 + col])
    ax_row2 = f.add_subplot(gs[1, :])
    ax_row3 = f.add_subplot(gs[2, :])

    # Plot the B matrix heatmap
    a = sns.heatmap(B_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax[0])
    a.set_title("B Matrix of PF2 Model")
    a.set_xlabel("Components")
    a.set_ylabel("Eigenstates")

    # Separate positive and negative genes for display
    pos_genes = weighted_genes_df[weighted_genes_df["direction"] == "positive"]
    neg_genes = weighted_genes_df[weighted_genes_df["direction"] == "negative"]

    # Calculate pos and neg % of total genes
    n_pos = pos_genes.shape[0]
    n_neg = neg_genes.shape[0]
    gene_space = X.var.index.size
    pos_cov = (n_pos / gene_space) * 100
    neg_cov = (n_neg / gene_space) * 100

    # Barplot of postive genes
    b = sns.barplot(
        data=pos_genes.head(n_genes),
        x="value",
        y="gene",
        ax=ax[1],
        color="red",
    )
    b.set_title(f"Top Positive Genes ({n_pos} genes)\nCoverage: {pos_cov:.2f}%")
    b.set_xlabel("Weighted Correlation")
    b.set_ylabel("Genes")

    # Barplot of negative genes
    c = sns.barplot(
        data=neg_genes.head(n_genes),
        x="value",
        y="gene",
        ax=ax[2],
        color="blue",
    )
    c.set_title(f"Top Negative Genes ({n_neg} genes)\nCoverage: {neg_cov:.2f}%")
    c.set_xlabel("Weighted Correlation")
    c.set_ylabel("Genes")

    f.suptitle(
        f"Strongest Eigenstate (Eigen-{strongest_eigenstate + 1}) and Associated Genes",
        fontsize=16,
    )

    # Prepare disease expression data for the top positive genes
    expr_long_pos = prepare_disease_expression(X, pos_genes["gene"].head(n_genes).to_list())

    # Boxplot of gene expression by disease
    d = sns.boxplot(data=expr_long_pos,
                    x="disease",
                    y="expression",
                    hue="gene",
                    ax=ax_row2,
                    flierprops=dict(marker="o", markersize=2)
    )
    d.set_title("Gene Expression by Disease (Positive Genes)")
    d.set_xlabel("Disease")
    d.set_ylabel("Expression (Log2 CPM)")

    # Prepare disease expression data for the top negative genes
    expr_long_neg = prepare_disease_expression(X, neg_genes["gene"].head(n_genes).to_list())

    # Boxplot of gene expression by disease
    e = sns.boxplot(
        data=expr_long_neg,
        x="disease",
        y="expression",
        hue="gene",
        ax=ax_row3,
        flierprops=dict(marker="o", markersize=2)
        )
    e.set_title("Gene Expression by Disease (Negative Genes)")
    e.set_xlabel("Disease")
    e.set_ylabel("Expression (Log2 CPM)")
    
    # Generate histogram figure for gene expression distributions
    hist_fig = plot_gene_expression_histograms(X, pos_genes, neg_genes, n_genes=n_genes)

    return f, hist_fig