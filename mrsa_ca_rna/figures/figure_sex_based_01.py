"""This file will explore sex based determinants of the MRSA data."""

import anndata as ad
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    datasets = ["mrsa"]
    disease = ["MRSA"]

    data: ad.AnnData = concat_datasets(
        datasets,
        disease,
    )

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

    genes = data.var.index.intersection(tier_1)

    X_tier_1 = data[:, genes].to_df()
    target = data.obs["status"]
    score, proba, model = perform_LR(X_tier_1, target, splits=5)

    metrics = (score, proba)

    return X_tier_1, target, metrics, model


def genFig():
    fig_size = (8, 8)
    layout = {"ncols": 1, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    data, targets, metrics, model = figure_setup()

    bal, proba = metrics

    classes = model.classes_

    # Create a DataFrame to store all ROC curves
    all_roc_data = []

    # Calculate ROC curves for each class
    for i, cls in enumerate(classes):
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
    a = sns.lineplot(data=combined_roc_df, x="FPR", y="TPR", hue="Class", ax=ax[0])

    # Add diagonal line (random classifier)
    a.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")

    # Set plot labels and title
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title("ROC Curves by Class")

    # Adjust legend position
    a.legend(loc="lower right", title="")

    # Plot the coefficients of each class
    coeffs = pd.DataFrame(model.coef_, columns=data.columns, index=classes)
    coeffs = coeffs.T
    
    # Reset the index to convert gene names to a column
    coeffs_long = coeffs.reset_index(drop=False, names="Gene")
    
    # Reshape from wide to long format for seaborn
    coeffs_long = pd.melt(
        coeffs_long, 
        id_vars=["Gene"], 
        value_vars=list(classes),
        var_name="Class", 
        value_name="Coefficient"
    )
    
    # Sort genes by their average absolute coefficient values to find most predictive ones
    top_genes = (coeffs.abs().mean(axis=1)
                 .sort_values(ascending=False)
                 .head(10)  # Show only top 10 most predictive genes
                 .index.tolist())
    
    # Filter to include only top genes
    plot_data = coeffs_long[coeffs_long['Gene'].isin(top_genes)]
    
    # Create the bar plot
    b = sns.barplot(
        data=plot_data,
        x='Gene',
        y='Coefficient',
        hue='Class',
        ax=ax[1]
    )
    
    # Set plot labels and title
    b.set_xlabel('Genes')
    b.set_ylabel('Coefficient Value')
    b.set_title('Top Predictive Genes by Class')
    
    # Adjust legend position
    b.legend(title='Class')
    
    
    return f

if __name__ == "__main__":
    f = genFig()