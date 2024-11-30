# main module imports
import numpy as np
import seaborn as sns

# secondary module imports
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# local imports
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_PC_LR

skf = StratifiedKFold(n_splits=10)


def setup_figure03b():
    """Perform similar analysis of figure03a, but with random pca"""

    # Get MRSA data
    orig_data = concat_datasets(scale=False, tpm=True)
    mrsa_data = orig_data[orig_data.obs["disease"] == "MRSA"].copy()
    ca_data = orig_data[orig_data.obs["disease"] == "Candidemia"].copy()

    # Perform PCA on CA data to get CA components
    _, _, ca_pca = perform_pca(ca_data.to_df())

    # scale MRSA data prior to use
    X = mrsa_data.X
    scaled_X = StandardScaler().fit_transform(X)
    mrsa_data.X = scaled_X

    # Explicitly convert X to np array to avoid calling shape on None
    X = np.asarray(mrsa_data.X)

    # Generate a random matrix orthogonal matrix the same size as X
    random_SVD = ortho_group.rvs(dim=X.shape[1])

    # select the first 70 components of the random SVD
    random_pca = random_SVD[:, :70]

    # transform MRSA data using random PCA model
    random_xform = np.dot(X, random_pca)

    # transform MRSA data using CA's PCA model
    mrsa_xform = ca_pca.transform(mrsa_data.to_df())

    # perform logistic regression on transformed MRSA data, with shuffling
    shuffled_status = mrsa_data.obs.loc[:, "status"].sample(frac=1)

    mrsa_dict = {"true": mrsa_xform, "random": random_xform}
    status_dict = {"true": mrsa_data.obs.loc[:, "status"], "random": shuffled_status}

    # perform logistic regression for the following cases:
    # - True MRSA data, True status
    # - True MRSA data, Shuffled status
    # - Random MRSA data, True status

    results = {}

    for data_key, data_value in mrsa_dict.items():
        for status_key, status_value in status_dict.items():
            key = f"{data_key}_{status_key}"
            _, model = perform_PC_LR(data_value, status_value, return_clf=True)
            results[key] = cross_val_predict(
                model, X=data_value, y=status_value, cv=skf, method="predict_proba"
            )

    return results, mrsa_data.obs["status"].values.astype(int)


def genFig():
    fig_size = (16, 4)
    layout = {"ncols": 4, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    # plot the ROC curve for random PCA
    proba_dict, y_true = setup_figure03b()

    for i, key in enumerate(proba_dict):
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=proba_dict[key][:, 1])
        data = {"FPR": fpr, "TPR": tpr}
        a = sns.lineplot(data, x="FPR", y="TPR", ax=ax[i])
        a.set_xlabel("False Positive Rate")
        a.set_ylabel("True Positive Rate")
        a.set_title(
            "Classification of MRSA outcomes using 70 component\n"
            f"PCA decomposition of {key} data\n"
            f"AUC: {roc_auc_score(y_true, proba_dict[key][:, 1]):.3f}"
        )

    return f
