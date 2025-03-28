"""This file will plot some mrsa and ca results from pf2 factorization
plots will include: ROC curve for mrsa prediction, scatter plot of component 3 and 6
for mrsa, and a strip plot of component 3, 6, 9 and 7 for mrsa and ca
"""

import anndata as ad
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score

from mrsa_ca_rna.utils import concat_datasets
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.figures.base import setupBase


def figure_setup():

    disease_list = ["mrsa", "ca"]

    meta = concat_datasets(disease_list).obs
    Pf2_CA = pd.read_csv("output_gsea/Pf2_CA.csv", index_col=0)
    Pf2_MRSA = pd.read_csv("output_gsea/Pf2_MRSA.csv", index_col=0)
    data = pd.concat([Pf2_MRSA, Pf2_CA], axis=0)

    X = ad.AnnData(data, obs=meta)
    
    # perform logistic regression on MRSA data
    mrsa_data = X[X.obs["disease"] == "MRSA"].copy()
    y_true = mrsa_data.obs.loc[:, "status"].astype(int)
    _, y_proba, _ = perform_LR(mrsa_data.X, y_true)

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba[:, 1])
    LR_data = {"FPR": fpr, "TPR": tpr}
    LR_score = roc_auc_score(y_true, y_proba[:, 1])

    # organize the data for plotting components
    components = [x for x in range(1, len(Pf2_MRSA.columns) + 1)]
    df = pd.DataFrame(X.X, index=X.obs.index, columns=components)
    df["disease"] = X.obs["disease"]
    df["status"] = X.obs["status"]

    return df, LR_data, LR_score

def genFig():
    figure_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(figure_size, layout)

    df, LR_data, LR_score = figure_setup()

    n_cats = len(df.loc[:, "status"].unique())
    sns.set_palette("turbo", n_cats)

    # plot the ROC curve
    a = sns.lineplot(LR_data, x="FPR", y="TPR", ax=ax[0])
    a.set_title(f"ROC Curve\nAUC: {LR_score:.2f}")
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")

    # plot the scatter plot of component 3 and 6
    a = sns.scatterplot(data=df, x=3, y=6, hue="status", style="disease", ax=ax[1])
    a.set_title("Component 3 vs Component 6")
    a.set_xlabel("Component 3")
    a.set_ylabel("Component 6")

    # plot the strip plot of components 3, 6, 9, and 7
    df = df.loc[:, ["disease", "status", 3, 6, 9, 7]]
    df = df.melt(id_vars=["disease", "status"], var_name="Component", value_name="Value")
    a = sns.stripplot(data=df, x="Component", y="Value", hue="status", ax=ax[2], dodge=True)
    a.set_title("Component 3, 6, 9, 7")
    a.set_xlabel("Component")
    a.set_ylabel("Value")

    return f
