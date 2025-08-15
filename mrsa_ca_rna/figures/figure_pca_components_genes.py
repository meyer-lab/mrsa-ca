"""
This figure shows the ROC results of the combined logistic regression model,
the top 5 components ranked by predictive ability, and their associated comp_genes.
"""

# imports

import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.figures.helpers import plot_component_features
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import find_top_features, prepare_mrsa_ca


def make_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """Make the ROC curve"""

    # convert y's to integer
    y_true = y_true.astype(int)

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba[:, 1])
    data = {"FPR": fpr, "TPR": tpr}

    auc = roc_auc_score(y_true, y_proba[:, 1])

    return data, auc


def get_data():
    # Get the MRSA and CA data, grab persistance labels
    _, _, combined_ad = prepare_mrsa_ca()

    # Get the labels for MRSA samples
    y_true = combined_ad.obs.loc[combined_ad.obs["disease"] == "MRSA", "status"].astype(
        int
    )
    
    return y_true, combined_ad

def run_models(components, y_true, combined_ad: ad.AnnData):

    # Convert to DataFrame for PCA
    combined_df = combined_ad.to_df()

    patient_comps, gene_comps, _ = perform_pca(combined_df, components=components)

    # Can only use MRSA data for regression, so truncate the combined PC data
    mrsa_index = combined_ad.obs["disease"] == "MRSA"
    patient_comps = patient_comps.loc[mrsa_index].copy()

    # Perform logistic regression on the PCA components
    _, y_proba, model = perform_LR(patient_comps, y_true, splits=10)

    # get the beta coefficients from the model and associate them with the components
    weights: np.ndarray = model.coef_[0]
    weights_dict = {f"PC{i + 1}": weights[i] for i in range(len(weights))}

    return y_proba, gene_comps, weights_dict

def genFig():

    # Scope of our PCA analysis
    components = 5

    # ROC curve, beta coefficients, and top genes per component
    num_plots = 2 + components

    fig_size, layout = calculate_layout(num_plots)
    ax, f, gs = setupBase(fig_size, layout)

    gs.update(hspace=0.1)

    y_true, combined_ad = get_data()

    # Run the models and get the predictions
    y_proba, gene_comps, weights = run_models(components, y_true, combined_ad)

    # Get the top genes by component
    features_df = find_top_features(gene_comps.T, 0.75, feature_name="gene")

    data, auc = make_roc_curve(y_true, y_proba)

    a = sns.lineplot(data, x="FPR", y="TPR", ax=ax[0])
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title(
        f"Classification of MRSA outcomes using a {components} component\n"
        "PCA decomposition of Combined data\n"
        f"AUC: {auc:.3f}"
    )

    # Plot the beta coefficients of the PCA components
    a = sns.barplot(x=list(weights.keys()), y=list(weights.values()), ax=ax[1])
    a.set_xlabel("Component")
    a.set_ylabel("Weight")
    a.set_title("Beta Coefficients of the PCA Components")

    # Plot the top genes for each component
    for i in range(components):
        plot_component_features(
            ax=ax[2 + i],
            features_df=features_df,
            component=i + 1,
            feature_name="gene",
            n_features=10,
        )

    return f
