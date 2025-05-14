"""This file explores sex-based determinants of MRSA data
through ROC curves and gene coefficient analysis across gene tiers."""

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import import_gene_tiers
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets


def setup_data():
    """Load the dataset."""
    datasets = ["mrsa"]
    disease = ["MRSA"]

    data: ad.AnnData = concat_datasets(
        datasets,
        disease,
    )

    return data


def create_model(data, gene_list):
    """Create model for a specific set of genes."""
    # Get genes that exist in the dataset
    genes = data.var.index.intersection(gene_list)

    # Print percent of genes found in the dataset and any missing genes
    percent_found = len(genes) / len(gene_list) * 100
    missing_genes = set(gene_list) - set(genes)
    print(
        f"Found {len(genes)} out of {len(gene_list)} genes in the dataset "
        f"({percent_found:.2f}%)"
    )
    if (len(missing_genes) > 0) & (len(missing_genes) < 100):
        print(f"Missing genes: {', '.join(missing_genes)}")

    # Extract features and target
    X = data[:, genes].to_df()
    target = data.obs["status"]

    # Create and train model
    score, proba, model = perform_LR(X, target, splits=5)

    return X, target, score, proba, model


def plot_roc_for_tier(ax, X, targets, score, proba, model):
    """Plot ROC curves for a specific gene tier."""
    classes = pd.Index(model.classes_)

    # Create a DataFrame to store all ROC curves
    all_roc_data = []

    # Calculate ROC curves for each class
    for i, cls in enumerate(classes.to_list()):
        # Convert targets to binary: 1 for current class, 0 for other classes
        binary_targets = (targets == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_true=binary_targets, y_score=proba[:, i])

        # Calculate AUC score
        auc = roc_auc_score(binary_targets, proba[:, i])

        # Create a DataFrame for this class's ROC curve
        roc_df = pd.DataFrame(
            {"FPR": fpr, "TPR": tpr, "Class": f"{cls} (AUC={auc:.2f})"}
        )

        all_roc_data.append(roc_df)

    # Combine all ROC data
    combined_roc_df = pd.concat(all_roc_data, ignore_index=True)

    # Plot all ROC curves
    sns.lineplot(data=combined_roc_df, x="FPR", y="TPR", hue="Class", ax=ax)

    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")

    # Set plot labels and title
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title(
    #     f"ROC Curves by Class ({len(X.columns)} genes)"
    #     f"\nBalanced Accuracy: {score:.2f}"
    # )

    # Adjust legend position
    ax.legend(loc="lower right", title="", fontsize="small")


def plot_coefficients_for_tier(ax, X, score, model, top_n=10):
    """Plot gene coefficients for a specific gene tier."""
    classes = pd.Index(model.classes_)

    # Get coefficients
    coeffs = pd.DataFrame(model.coef_, columns=X.columns, index=classes)
    coeffs = coeffs.T

    # Reset the index to convert gene names to a column
    coeffs_long = coeffs.reset_index(drop=False, names="Gene")

    # Reshape from wide to long format for seaborn
    coeffs_long = pd.melt(
        coeffs_long,
        id_vars=["Gene"],
        value_vars=classes.to_list(),
        var_name="Class",
        value_name="Coefficient",
    )

    # Sort genes by their average absolute coefficient
    # values to find most predictive ones
    mean_abs_coeffs = coeffs.abs().mean(axis=1)

    # Pyright always complains about taking the mean of a Dataframe
    if isinstance(mean_abs_coeffs, float | int):
        raise ValueError(
            "mean_abs_coeffs should be a Series, not a single value."
            f" Got {type(mean_abs_coeffs)} instead. Was the input data a single row?"
        )

    top_genes = mean_abs_coeffs.sort_values(ascending=False).head(top_n).index.tolist()

    # Filter to include only top genes
    plot_data = coeffs_long[coeffs_long["Gene"].isin(top_genes)]

    # Ensure plot_data is a DataFrame
    plot_data_df = pd.DataFrame(plot_data)

    # Create the bar plot
    sns.barplot(data=plot_data_df, x="Gene", y="Coefficient", hue="Class", ax=ax)

    # Set plot labels and title
    ax.set_xlabel("Genes")
    ax.set_ylabel("Coefficient Value")
    # ax.set_title(
    #     f"Top {top_n} Predictive Genes in ({len(X.columns)} genes)"
    #     f"\nBalanced Accuracy: {score:.2f}"
    # )

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Choose consistent legend position
    ax.legend(title="Class", loc="upper right", fontsize="small")

        # Add dividing lines between gene groups
    # Get the x-tick positions
    xticks = ax.get_xticks()
    
    # Add alternating background for each gene group
    for i in range(len(xticks)):
        # Calculate the boundaries for each gene group
        if i < len(xticks) - 1:
            left = (xticks[i] + xticks[i+1]) / 2
            ax.axvline(x=left, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)
    
    # Add light background shading for alternate gene groups
    for i in range(len(xticks)):
        if i % 2 == 0:  # For even-indexed genes
            if i < len(xticks) - 1:
                left = xticks[i] - 0.4
                right = xticks[i] + 0.4
                ax.axvspan(left, right, color='lightgray', alpha=0.2, zorder=0)


def genFig():
    """Generate both ROC curve and coefficient plots for each gene tier"""
    # Get tiers data
    tiers = import_gene_tiers()

    # Load data once
    data = setup_data()

    # Add 5th tier for all genes represented in the dataset
    all_genes = data.var.index.tolist()
    tiers["Tier 5"] = all_genes

    num_tiers = len(tiers)

    # Create first figure for ROC curves
    roc_fig_size = (4, num_tiers * 4)
    roc_layout = {"ncols": 1, "nrows": num_tiers}
    roc_ax, roc_f, _ = setupBase(roc_fig_size, roc_layout)

    # Create second figure for coefficients
    coef_fig_size = (14, num_tiers * 5)
    coef_layout = {"ncols": 1, "nrows": num_tiers}
    coef_ax, coef_f, _ = setupBase(coef_fig_size, coef_layout)

    # Set titles for both figures
    roc_f.suptitle("ROC Curves for Gene Tiers in MRSA Prediction", fontsize=16)
    coef_f.suptitle("Gene Coefficients by Class for MRSA Prediction", fontsize=16)

    # Process each tier, creating a single model for both plots
    for i, (tier_name, gene_list) in enumerate(tiers.items()):
        print(f"Modeling {tier_name} genes...")

        # Create model once
        X, targets, score, proba, model = create_model(data, gene_list)

        # Plot ROC curves
        plot_roc_for_tier(roc_ax[i], X, targets, score, proba, model)

        # Plot coefficients
        plot_coefficients_for_tier(coef_ax[i], X, score, model)

        # Add tier name to ROC title
        roc_ax[i].set_title(
            f"ROC Curves by Class - {tier_name} ({len(X.columns)} genes)"
            f"\nBalanced Accuracy: {score:.2f}"
        )

        # Add tier name to coefficients title
        coef_ax[i].set_title(
            f"Top 10 Predictive Genes - {tier_name} ({len(X.columns)} genes)"
            f"\nBalanced Accuracy: {score:.2f}"
        )

    # Return both figures as a tuple
    return (roc_f, coef_f)
