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
from mrsa_ca_rna.utils import concat_datasets

skf = StratifiedKFold(n_splits=10)


def make_roc_curve(X, y):
    """Function trains model on given data and returns the ROC curve"""
    # Import and scale mrsa and ca data together
    datasets = ["mrsa", "ca"]
    diseases = ["MRSA", "Candidemia"]
    mrsa_ca = concat_datasets(
        datasets,
        diseases,
        scale=True,
    )
    # Trim to mrsa data and extract y_true
    mrsa = mrsa_ca[mrsa_ca.obs["disease"] == "MRSA"].copy()
    y_true = mrsa.obs.loc[:, "status"].astype(int)

    _, y_proba, _ = perform_LR(X, y)  # type: ignore
    # for the life of me, I cannot figure out how to stop pyright from complaining here

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba[:, 1])

    data = {"FPR": fpr, "TPR": tpr}
    score = roc_auc_score(y_true, y_proba[:, 1])

    return data, score


def figure_setup():
    """Performs logistic regression on MRSA data, transformed using CA's PCA model,
    and using random data. The statuses are either shuffled or not for each case"""

    # Get MRSA data
    datasets = ["mrsa", "ca"]
    diseases = ["MRSA", "Candidemia"]
    orig_data = concat_datasets(
        datasets,
        diseases,
        scale=False,
    )
    mrsa_data = orig_data[orig_data.obs["disease"] == "MRSA"].copy()
    ca_data = orig_data[orig_data.obs["disease"] == "Candidemia"].copy()
    y_true = mrsa_data.obs.loc[:, "status"].astype(int)

    # Perform PCA on CA data to get CA components
    components = 5
    _, _, ca_pca = perform_pca(ca_data.to_df(), components=components)

    # scale MRSA data prior to use
    X = mrsa_data.X
    scaled_X = StandardScaler().fit_transform(X)
    mrsa_data.X = scaled_X

    # Explicitly convert X to np array to avoid calling shape on None
    X = np.asarray(mrsa_data.X)

    # Generate random orthogonal matrix
    random_ortho = ortho_group.rvs(X.shape[0])
    random_ortho = random_ortho[:, :components]

    # transform MRSA data using CA's PCA model
    mrsa_xform = ca_pca.transform(mrsa_data.to_df())

    # shuffle status
    shuffled_status = y_true.sample(frac=1)

    mrsa_dict = {"mrsa": mrsa_xform, "random": random_ortho}

    return mrsa_dict, shuffled_status, y_true


def genFig():
    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    mrsa_dict, y_shuffled, y_true = figure_setup()

    # perform logistic regression for the following cases:
    # - MRSA data, True status
    # - MRSA data, Shuffled status
    # - Random data, True status
    true_mrsa, true_score = make_roc_curve(mrsa_dict["mrsa"], y_true)
    shuffled_mrsa, shuffled_score = make_roc_curve(mrsa_dict["mrsa"], y_shuffled)
    random, random_score = make_roc_curve(mrsa_dict["random"], y_true)

    # plot ROC curves
    # - MRSA data, True status
    a = sns.lineplot(true_mrsa, x="FPR", y="TPR", ax=ax[0])
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title(
        "Classification of MRSA outcomes using\n"
        f"PCA decomposition of MRSA data with true labels\n"
        f"AUC: {true_score:.3f}"
    )

    # - MRSA data, Shuffled status
    a = sns.lineplot(shuffled_mrsa, x="FPR", y="TPR", ax=ax[1])
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title(
        "Classification of MRSA outcomes using\n"
        f"PCA decomposition of MRSA data with shuffled labels\n"
        f"AUC: {shuffled_score:.3f}"
    )

    # - Random data, True status
    a = sns.lineplot(random, x="FPR", y="TPR", ax=ax[2])
    a.set_xlabel("False Positive Rate")
    a.set_ylabel("True Positive Rate")
    a.set_title(
        "Classification of MRSA outcomes using\n"
        f"PCA decomposition of random data with true labels\n"
        f"AUC: {random_score:.3f}"
    )

    return f
