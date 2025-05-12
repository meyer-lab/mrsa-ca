"""This file explores sex-based determinants of MRSA data
through ROC curves across gene tiers."""

import anndata as ad
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets


def get_gene_tiers():
    """Define gene tiers for the analysis.
    Tier 1 genes are the core genes, determined by Dr. Thaden's analysis.
    Tiers 2-4 are undecided and placeholders are used for now."""

    # Tier 1 genes (core genes)
    tier_1 = [
        "GLS2",
        "HLA-DRA",
        "CD74",
        "DNMT3A",
        "IL10",
        "STAT1",
        "CEBPB",
        "TIRAP",
        "IL1RN",
        "IL27",
        "CHI3L1",
        "CCL2",
        "ANGPT2",
        "CCL26",
        "CSCL8",
        "SELE",
        "IL2RA",
        "LCN2",
        "IL1B",
        "IL6",
        "CXCL1",
    ]

    # Tier 2 genes
    tier_2 = tier_1 + ["TNF", "IFNG", "IL4", "IL13", "CCR2", "TLR2", "TLR4"]

    # Tier 3 genes
    tier_3 = tier_2 + ["ICAM1", "VCAM1", "MMP9", "TIMP1", "FOXP3", "CD4", "CD8A"]

    # Tier 4 genes
    tier_4 = tier_3 + ["IL17A", "IL22", "IL23A", "RORC", "TGFB1", "SMAD3", "SMAD7"]

    return {"Tier 1": tier_1, "Tier 2": tier_2, "Tier 3": tier_3, "Tier 4": tier_4}


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

    # Extract features and target
    X = data[:, genes].to_df()
    target = data.obs["status"]

    # Create and train model
    score, proba, model = perform_LR(X, target, splits=5)

    return X, target, score, proba, model


def plot_roc_for_tier(ax, data, gene_list, tier_name):
    """Plot ROC curves for a specific gene tier."""
    # Create model
    X, targets, score, proba, model = create_model(data, gene_list)
    classes = pd.Index(model.classes_)

    # Create a DataFrame to store all ROC curves
    all_roc_data = []

    # Calculate ROC curves for each class
    for i, cls in enumerate(
        classes.to_list()
    ):  # Convert to list to ensure it's iterable
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
    ax.set_title(
        f"ROC Curves by Class - {tier_name} ({len(X.columns)} genes)"
        f"\nBalanced Accuracy: {score:.2f}"
    )

    # Adjust legend position
    ax.legend(loc="lower right", title="", fontsize="small")

    return score


def genFig():
    """Generate ROC curve figures for all gene tiers."""
    # Setup figure
    tiers = get_gene_tiers()
    num_tiers = len(tiers)

    fig_size = (4, num_tiers * 4)
    layout = {"ncols": 1, "nrows": num_tiers}
    ax, f, _ = setupBase(fig_size, layout)

    data = setup_data()

    # Plot ROC curves for each tier
    tier_scores = {}
    for i, (tier_name, gene_list) in enumerate(tiers.items()):
        # Replace Tier 4 with all genes for the last tier
        if tier_name == "Tier 4":
            gene_list = data.var.index.tolist()
            
        score = plot_roc_for_tier(ax[i], data, gene_list, tier_name)
        tier_scores[tier_name] = score

    f.suptitle("ROC Curves for Gene Tiers in MRSA Prediction", fontsize=16)

    return f
