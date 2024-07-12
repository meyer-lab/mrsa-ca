"""
Just for my own sanity, I'm going to perform all scaling, pca, and predictions
in a single spot to see if I can get some consisyency. This will just be
using the MRSA data to see what kind of accuracy I get with a single dataset

"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss

import pandas as pd
import numpy as np
import seaborn as sns

from mrsa_ca_rna.import_data import (
    import_mrsa_meta,
    import_mrsa_rna,
    import_mrsa_val_meta,
)
from mrsa_ca_rna.figures.base import setupBase

def aio_pipe(components=60):

    # bring in the data
    mrsa_rna= import_mrsa_rna()
    mrsa_meta = import_mrsa_meta()
    mrsa_val_meta = import_mrsa_val_meta()

    # combine the data together to whole dfs
    mrsa_train_rna = mrsa_meta.loc[~mrsa_meta["status"].str.contains("Unknown"), "status"]
    mrsa_train_rna = pd.concat([mrsa_train_rna, mrsa_rna], axis=1, join="inner") 

    mrsa_test_rna = mrsa_val_meta["status"]
    mrsa_test_rna = pd.concat([mrsa_test_rna, mrsa_rna], axis=1, join="inner")

    # data organizing block for application in the pipeline and validation
    mrsa_train_X = mrsa_train_rna.iloc[:, 1:]
    mrsa_train_y = mrsa_train_rna["status"].astype(int)
    mrsa_test_X = mrsa_test_rna.iloc[:, 1:]
    mrsa_test_y = mrsa_test_rna["status"].astype(int)

    rng = 42
    scaler = StandardScaler().set_output(transform="pandas")
    pca = PCA(n_components=components)
    skf = StratifiedKFold(n_splits=10)
    Cs = np.logspace(-5, 5, 20)

    clf = make_pipeline(scaler, pca, LogisticRegressionCV(Cs=Cs, cv=skf, max_iter=1000, random_state=rng))

    clf.fit(mrsa_train_X, mrsa_train_y)

    print(f"Optimal C for MRSA data: {clf[-1].C_[0]:.4f}")

    y_pred = clf.predict(mrsa_test_X)
    y_proba = clf.predict_proba(mrsa_test_X)

    print(f"Test accuracy of entire pipeline: {accuracy_score(mrsa_test_y, y_pred)}")
    print(f"Log-loss for the pipeline: {log_loss(mrsa_test_y, y_proba)}")

    return accuracy_score(mrsa_test_y, y_pred), log_loss(mrsa_test_y, y_proba)


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    components = np.arange(1, 61)
    scores = []

    for component in components:

        accuracy, _ = aio_pipe(components=component)
        scores.append(accuracy)

    data = pd.DataFrame(data=components, columns=["Components"])
    data["Accuracy"] = scores

    a = sns.lineplot(
        data=data, x="Components", y="Accuracy", ax=ax[0]
    )
    a.set_xlabel("# of components")
    a.set_ylabel("Accuracy")
    a.set_title("Pipeline Performance")

    return f
