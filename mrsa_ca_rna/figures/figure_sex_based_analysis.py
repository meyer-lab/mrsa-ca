"""This file explores sex-based determinants of MRSA data
through ROC curves and gene coefficient analysis across gene tiers."""

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import import_gene_tiers
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets

from scipy import stats
from statsmodels.stats.multitest import multipletests


def setup_data():
    """Load the dataset."""

    # Load the MRSA dataset with minimal filtering
    data = concat_datasets(["mrsa"], filter_threshold=0)

    return data


def create_model(data: ad.AnnData, gene_list: list[str]) -> tuple:
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

    # Extract features and target. Pyright hates when I subset AnnData objects
    X = data[:, genes].to_df()  # type: ignore
    target = data.obs["gender"].astype(int)

    # Create and train model
    score, proba, model = perform_LR(X, target, splits=10)

    return X, target, score, proba, model


def plot_roc_for_tier(ax, targets, proba, model):
    """Plot ROC curves for a specific gene tier."""
    classes = pd.Index(model.classes_)

    # Handle binary classification differently
    if len(classes) == 2:
        # For binary classification, only plot ROC curve for the positive class
        positive_class_idx = np.where(classes == 1)[0][0]
        binary_targets = (targets == 1).astype(int)
        fpr, tpr, _ = roc_curve(
            y_true=binary_targets, y_score=proba[:, positive_class_idx]
        )

        # Calculate AUC score
        auc = roc_auc_score(binary_targets, proba[:, positive_class_idx])

        # Plot single ROC curve
        ax.plot(fpr, tpr, label=f"Gender (AUC={auc:.2f})")

    else:
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

    # Add diagonal line
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")

    # Set plot labels and title
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # Adjust legend position
    ax.legend(loc="lower right", title="", fontsize="small")


def plot_coefficients_for_tier(ax, X, model, top_n=10):
    """Plot gene coefficients for a specific gene tier."""
    classes = pd.Index(model.classes_)

    # Handle binary classification differently than multinomial
    if len(classes) == 2 and model.coef_.shape[0] == 1:
        # For binary classification, only show coefficients for the positive class
        coef = model.coef_[0]

        # Create a Series with gene names as index and coefficients as values
        coeffs_series = pd.Series(coef, index=X.columns)

        # Sort by absolute coefficient values to find most predictive genes
        top_genes = coeffs_series.abs().sort_values(ascending=False).head(top_n).index

        # Filter to only include top genes
        plot_data = pd.DataFrame(
            {"Gene": top_genes, "Coefficient": coeffs_series[top_genes]}
        )

        sns.barplot(data=plot_data, x="Gene", y="Coefficient", ax=ax, color="steelblue")

        # Add a horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    else:
        # For multinomial case, use all classes
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

        # Sort genes by their average absolute coefficient values
        mean_abs_coeffs = coeffs.abs().mean(axis=1)

        # Pyright always complains about taking the mean of a Dataframe
        if isinstance(mean_abs_coeffs, float | int):
            raise ValueError(
                "mean_abs_coeffs should be a Series, not a single value. "
                f"Got {type(mean_abs_coeffs)} instead. "
                "Was the input data a single row? "
            )

        top_genes = (
            mean_abs_coeffs.sort_values(ascending=False).head(top_n).index.tolist()
        )

        # Filter to include only top genes and ensure it's a DataFrame
        plot_data: pd.DataFrame = coeffs_long.loc[
            coeffs_long["Gene"].isin(top_genes)
        ].copy()

        # Create the bar plot with hue for multiple classes
        sns.barplot(data=plot_data, x="Gene", y="Coefficient", hue="Class", ax=ax)

        # Add legend for multinomial case
        ax.legend(title="Class", loc="upper right", fontsize="small")

    # Set plot labels and title
    ax.set_xlabel("Genes")
    ax.set_ylabel("Coefficient Value")

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add dividing lines between gene groups
    xticks = ax.get_xticks()

    # Add alternating background for each gene group
    for i in range(len(xticks)):
        if i < len(xticks) - 1:
            left = (xticks[i] + xticks[i + 1]) / 2
            ax.axvline(x=left, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)

    # Add light background shading for alternate gene groups
    for i in range(len(xticks)):
        if i % 2 == 0 and i < len(xticks) - 1:
            left = xticks[i] - 0.4
            right = xticks[i] + 0.4
            ax.axvspan(left, right, color="lightgray", alpha=0.2, zorder=0)


def plot_confusion_matrix(ax, targets, proba, model):
    """Plot confusion matrix for model predictions."""
    # Get predicted classes from probabilities
    y_pred = model.classes_[np.argmax(proba, axis=1)]

    # Create display labels with "Resolver" and "Persistent" instead of 0 and 1
    display_labels = np.array(
        [
            "Male" if cls == 0 else "Female" if cls == 1 else cls
            for cls in model.classes_
        ]
    )

    # Create and plot confusion matrix
    cm = confusion_matrix(targets, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    # Set title and labels
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Improve formatting
    for text in ax.texts:
        text.set_fontsize(10)


def plot_gender_confusion_matrix(ax, data, proba, model, gender_type="male"):
    """Plot a confusion matrix for a specific gender."""
    # Get predicted classes from probabilities
    y_pred = model.classes_[np.argmax(proba, axis=1)]

    # Get true outcomes and gender information
    true_outcome = data.obs["Persistent"].astype(int).values
    gender = data.obs["gender"].values

    # Filter for the specified gender
    indices = []
    for i, g in enumerate(gender):
        is_target_gender = False
        if isinstance(g, np.integer | int):
            # Numeric encoding
            is_target_gender = (g == 0 and gender_type == "male") or (
                g == 1 and gender_type == "female"
            )
        else:
            # String encoding
            g_str = str(g).lower()
            is_target_gender = g_str == gender_type

        if is_target_gender:
            indices.append(i)

    # Get data for the gender
    gender_true = true_outcome[indices]
    gender_pred = y_pred[indices]

    # Create labels
    labels = ["Resolver", "Persistent"]

    # Create and plot confusion matrix
    cm = confusion_matrix(gender_true, gender_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    # Set labels
    ax.set_title(f"{gender_type.capitalize()} Patients")
    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("True Outcome")

    return cm


def plot_gene_expression_by_gender(ax, data, gene_list):
    """
    Plot boxplots comparing gene expression between males and females
    for a given gene list and perform statistical testing with BH correction.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which to plot
    data : AnnData
        The AnnData object containing gene expression data
    gene_list : list
        List of genes to include in the analysis
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing statistical test results
    """
    # Get genes that exist in the dataset
    genes = data.var.index.intersection(gene_list)
    
    # Extract log-normalized expression data (before z-scoring) for meaningful comparisons
    # This uses CPM-normalized, log2-transformed data that preserves biological meaning
    raw_counts = data[:, genes].layers["raw"]
    total_counts = np.sum(raw_counts, axis=1, keepdims=True)
    cpm_data = raw_counts * 1e6 / total_counts
    log_expr_data = np.log2(cpm_data + 1)
    
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
    
    # Perform t-tests for each gene
    p_values = []
    t_stats = []
    genes_tested = []
    
    for gene in genes:
        male_expr = gene_expr_with_gender[
            gene_expr_with_gender["Gender"] == "Male"
        ][gene]
        female_expr = gene_expr_with_gender[
            gene_expr_with_gender["Gender"] == "Female"
        ][gene]
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(male_expr, female_expr, equal_var=False)
        
        p_values.append(p_val)
        t_stats.append(t_stat)
        genes_tested.append(gene)
    
    # Apply Benjamini-Hochberg correction
    reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Create results DataFrame
    stats_results = pd.DataFrame({
        'Gene': genes_tested,
        'T-statistic': t_stats,
        'P-value': p_values,
        'P-adjusted': p_corrected,
        'Significant': reject
    })
    
    # Convert to long format for seaborn
    gene_expr_long = pd.melt(
        gene_expr_with_gender,
        id_vars=["Gender"],
        value_vars=genes,
        var_name="Gene",
        value_name="Expression"
    )
    
    # Note: Removed variance-based filtering to use ALL genes in gene_list
    # This ensures we don't filter out any genes for an agnostic analysis
    
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
    ax.set_ylabel("Log2(CPM + 1)")  # Updated to reflect actual data scale
    
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
        
        # Add mean expression annotation using gender-specific normalized means
        # Get normalized expression data for this gene (matches y-axis scale)
        gene_data = gene_expr_long[gene_expr_long["Gene"] == gene]
        male_mean = gene_data[gene_data["Gender"] == "Male"]["Expression"].mean()
        female_mean = gene_data[gene_data["Gender"] == "Female"]["Expression"].mean()
        
        # Show gender-specific means in a compact format
        ax.text(i, y_min - y_range*0.15, 
                f"M:{male_mean:.1f}\nF:{female_mean:.1f}", 
                ha='center', fontsize=7, color='gray', va='top')
    
    # Add significance legend
    ax.text(0.5, 1.08, 
            "* p<0.05, ** p<0.01, *** p<0.001, ns: not significant (BH-corrected)",
            transform=ax.transAxes, ha='center', fontsize=9)
    
    # Add mean expression legend
    ax.text(0.5, -0.20, 
            "M/F = Male/Female mean log2(CPM+1) expression",
            transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    
    return stats_results


def genFig():
    """Generate ROC curves, confusion matrices, gene coefficients, 
    and gene expression boxplots."""
    # Get tiers data
    tiers = import_gene_tiers()

    # Load data once
    data = setup_data()

    # Add 5th tier for all genes represented in the dataset
    all_genes = data.var.index.tolist()
    tiers["Tier 5"] = all_genes

    # Optionally remove larger tiers for rapid testing
    tiers.pop("Tier 4", None)
    tiers.pop("Tier 5", None)

    num_tiers = len(tiers)

    # Create figure with 4 columns: ROC, standard CM, male CM, female CM
    fig_size = (16, num_tiers * 4)
    layout = {"ncols": 4, "nrows": num_tiers}
    ax, fig, _ = setupBase(fig_size, layout)

    # Set title for the figure
    fig.suptitle(
        "MRSA Prediction Analysis: ROC Curves, "
        "Standard and Gender-Based Confusion Matrices",
        fontsize=16,
    )

    # Create second figure for coefficients
    coef_fig_size = (14, num_tiers * 4)
    coef_layout = {"ncols": 1, "nrows": num_tiers}
    coef_ax, coef_f, _ = setupBase(coef_fig_size, coef_layout)
    coef_f.suptitle("Gene Coefficients by Class for Gender Prediction", fontsize=16)

    # Create an additional figure for gene expression boxplots
    expr_fig_size = (12, 12)
    expr_layout = {"ncols": 1, "nrows": 2}
    expr_ax, expr_f, _ = setupBase(expr_fig_size, expr_layout)
    expr_f.suptitle(
        "Gene Expression Comparison by Gender for Tiers 1 and 2", 
        fontsize=16
    )

    # Process each tier, creating a single model for all plots
    # Limit to only the first two tiers
    for i, (tier_name, gene_list) in enumerate(list(tiers.items())[:2]):
        print(f"Modeling {tier_name} genes...")

        # Create model once
        X, targets, score, proba, model = create_model(data, gene_list)

        # Calculate the indices for each visualization type
        # i tiers * 4 plots per tier + col_index for that plot
        roc_idx = i * 4
        cm_idx = i * 4 + 1
        male_cm_idx = i * 4 + 2
        female_cm_idx = i * 4 + 3

        # Plot ROC curves in first column
        plot_roc_for_tier(ax[roc_idx], targets, proba, model)

        # Plot standard confusion matrix in second column
        plot_confusion_matrix(ax[cm_idx], targets, proba, model)

        # Plot gender-specific confusion matrices
        plot_gender_confusion_matrix(
            ax[male_cm_idx], data, proba, model, gender_type="male"
        )
        plot_gender_confusion_matrix(
            ax[female_cm_idx], data, proba, model, gender_type="female"
        )

        # Plot coefficients in separate figure
        plot_coefficients_for_tier(coef_ax[i], X, model)

        # For tier 1, generate gene expression boxplot with statistical testing
        if tier_name == "Tier 1":
            gene_list = [
                "CCL2", "IL27", "CXCL8", "IL6", "GLS2", 
                "IL10", "SELE", "TIRAP", "CCL26", "CEBPB"
            ]
            tier1_stats = plot_gene_expression_by_gender(expr_ax[0], data, gene_list)
            print("Tier 1 statistical results:")
            print(tier1_stats)

        if tier_name == "Tier 2":
            gene_list = [
                "CXCL10", "GLS2", "CCL2", "TREM1", "SELE", 
                "NOXO1", "G6PD", "TIRAP", "HLA-DQA2", "HLA-DQB2"
            ]
            tier2_stats = plot_gene_expression_by_gender(expr_ax[1], data, gene_list)
            print("Tier 2 statistical results:")
            print(tier2_stats)

        # Add titles to each subplot
        ax[roc_idx].set_title(
            f"Gender Prediction - {tier_name} ({len(X.columns)} genes)"
            f"\nBalanced Accuracy: {score:.2f}"
        )

        ax[cm_idx].set_title(f"Standard Confusion Matrix - {tier_name}")

        ax[male_cm_idx].set_title(f"Male Confusion Matrix - {tier_name}")
        ax[female_cm_idx].set_title(f"Female Confusion Matrix - {tier_name}")

        # Add title to coefficients plot
        coef_ax[i].set_title(
            f"Top 10 Predictive Genes - {tier_name} ({len(X.columns)} genes)"
            f"\nBalanced Accuracy: {score:.2f}"
        )

    # Return all figures
    return fig, coef_f, expr_f
