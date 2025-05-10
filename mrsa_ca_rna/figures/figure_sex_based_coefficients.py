"""This file explores sex-based determinants of MRSA data
through gene coefficient analysis."""

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def plot_coefficients_for_tier(ax, data, gene_list, tier_name, top_n=10):
    """Plot gene coefficients for a specific gene tier."""
    # Create model
    X, _, score, _, model = create_model(data, gene_list)
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
    ax.set_title(
        f"Top {top_n} Predictive Genes - {tier_name} ({len(X.columns)} genes)"
        f"\nBalanced Accuracy: {score:.2f}"
    )

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Choose consistent legend position
    ax.legend(title="Class", loc="upper right", fontsize="small")

    return score


def genFig():
    """Generate coefficient figures for all gene tiers."""
    # Setup figure
    tiers = get_gene_tiers()
    num_tiers = len(tiers)

    fig_size = (14, num_tiers * 5)
    layout = {"ncols": 1, "nrows": num_tiers}
    ax, f, _ = setupBase(fig_size, layout)

    # Load data once
    data = setup_data()

    # Plot coefficients for each tier
    tier_scores = {}
    for i, (tier_name, gene_list) in enumerate(tiers.items()):
        score = plot_coefficients_for_tier(ax[i], data, gene_list, tier_name)
        tier_scores[tier_name] = score

    f.suptitle("Gene Coefficients by Class for MRSA Prediction", fontsize=16)

    return f
