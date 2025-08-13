"""File plots an ROC curve for MRSA data under different conditions.
These conditions are as follows:
MRSA data, transformed using CA's PCA model, and the 'True' statuses
MRSA data, transformed using CA's PCA model, and 'Shuffled' statuses
Random data, made using random orthogonal matrix, and the 'True' statuses
Random data, made using random orthogonal matrix, and 'Shuffled' statuses"""

# main module imports
import numpy as np
import seaborn as sns

# secondary module imports
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# local imports
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.utils import concat_datasets, prepare_mrsa_ca

skf = StratifiedKFold(n_splits=10)


def make_roc_curve(X, y):
    """Function trains model on given data and returns the ROC curve"""
    # Import and scale mrsa and ca data together
    datasets = concat_datasets(filter_threshold=-1)
    # Prepare the MRSA and CA data
    mrsa_adata, _, _ = prepare_mrsa_ca(datasets)

    # Trim to mrsa data and extract y_true
    y_true = mrsa_adata.obs.loc[:, "status"].astype(int)

    _, y_proba, _ = perform_LR(X, y)  # type: ignore
    # for the life of me, I cannot figure out how to stop pyright from complaining here

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba[:, 1])

    data = {"FPR": fpr, "TPR": tpr}
    score = roc_auc_score(y_true, y_proba[:, 1])

    return data, score


def figure_setup():
    """Performs logistic regression on MRSA data, transformed using CA's PCA model,
    and using random data. The statuses are either shuffled or not for each case"""

    # Grab and split the data
    combined = concat_datasets(filter_threshold=-1)
    mrsa_data, ca_data, combined = prepare_mrsa_ca(combined)

    # Save the MRSA data index for later
    mrsa_index = mrsa_data.obs.index

    # Get regression targets
    y_true = mrsa_data.obs.loc[:, "status"].astype(int)

    # Perform PCA on combined MRSA and CA data
    components = 5
    combined_pc, _, _ = perform_pca(combined.to_df(), components=components)
    mrsa_pc, _, _ = perform_pca(mrsa_data.to_df(), components=components)
    _, _, ca_pca = perform_pca(ca_data.to_df(), components=components)

    # Generate random orthogonal matrix with the same number of components
    random_ortho = ortho_group.rvs(np.asarray(mrsa_data.X).shape[0])
    random_ortho = random_ortho[:, :components]

    # transform MRSA data using CA's PCA model
    mrsa_xform = ca_pca.transform(mrsa_data.to_df())

    # shuffle status
    shuffled_status = y_true.sample(frac=1)

    # Truncate the combined data to MRSA data for the combined_pc
    combined_pc = combined_pc.loc[mrsa_index, :].copy()

    mrsa_dict = {
        "combined": combined_pc,
        "mrsa": mrsa_pc,
        "mrsa_x": mrsa_xform,
        "random": random_ortho,
    }

    return mrsa_dict, shuffled_status, y_true


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    mrsa_dict, y_shuffled, y_true = figure_setup()

    # perform logistic regression for the following cases:
    # - Combined data, True status
    # - MRSA data, True status
    # - Transofmed MRSA data, True status
    # - Combined data, Shuffled status
    # - Random data, True status
    combined, combined_auc = make_roc_curve(mrsa_dict["combined"], y_true)
    mrsa, mrsa_auc = make_roc_curve(mrsa_dict["mrsa"], y_true)
    mrsa_x, mrsa_x_auc = make_roc_curve(mrsa_dict["mrsa_x"], y_true)
    shuffled, shuffled_auc = make_roc_curve(mrsa_dict["combined"], y_shuffled)
    random, random_auc = make_roc_curve(mrsa_dict["random"], y_true)

    a = sns.lineplot(
        combined,
        x="FPR",
        y="TPR",
        label=f"Combined (AUC: {combined_auc:.2f})",
        ax=ax[0],
    )
    sns.lineplot(
        mrsa,
        x="FPR",
        y="TPR",
        label=f"MRSA (AUC: {mrsa_auc:.2f})",
        ax=ax[0],
    )
    sns.lineplot(
        mrsa_x,
        x="FPR",
        y="TPR",
        label=f"Transformed MRSA (AUC: {mrsa_x_auc:.2f})",
        ax=ax[0],
    )
    sns.lineplot(
        shuffled,
        x="FPR",
        y="TPR",
        label=f"Shuffled (AUC: {shuffled_auc:.2f})",
        ax=ax[0],
    )
    sns.lineplot(
        random,
        x="FPR",
        y="TPR",
        label=f"Random (AUC: {random_auc:.2f})",
        ax=ax[0],
    )
    a.plot([0, 1], [0, 1], linestyle="--", color="grey", label="No descrimination")
    a.set_title("ROC Curves for MRSA Data under Different Conditions")
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.legend(title="Condition", loc="lower right")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)

    return f
