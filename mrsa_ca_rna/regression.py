"""
This file will include logstical regression methods
and other relevant functions, starting with MRSA data,
to determine patterns of persistance vs. resolving infection
to eventually compare across disease types to see if
these observed patterns are preserved.

To-do:
    Remove the whole_scaled_clf.to_csv line and just return the weights
    from the perform_whole_LR() func.
"""

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    KFold,
    LeaveOneOut,
    cross_val_score,
    cross_val_predict,
)
from sklearn.linear_model import (
    LogisticRegressionCV,
    LogisticRegression,
    ElasticNet,
    ElasticNetCV,
    LinearRegression,
)
from sklearn.cross_decomposition import PLSRegression

import pandas as pd
import numpy as np
import random


skf = StratifiedKFold(n_splits=10)
kf = KFold(n_splits=10)
loocv = LeaveOneOut()


def perform_PC_LR(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_data: pd.DataFrame | None = None,
    y_data: pd.DataFrame | None = None,
):
    """
    Agnostically performs LogisticRegression with nested cross validation to passed data. Regularization
    can be determined with different data by passing additional data that will not be used for regularization.

    Parameters:
        X_train (pd.DataFrame): training X data for regularization and subsequent nested cross validation
        y_train (pd.DataFrame): training y data for regularization and subsequent nested cross validation
        X_data (pd.DataFrame): [Optional] final model fitted with this data after training is used for regularization
        y_data (pd.DataFrame): [Optional] final model fitted with this data after training is used for regularization

    Returns:
        nested_score (float): nested cross validation score of the final model
        convergence_failure (int): when not 0, indicates a convergence failure occurred, at indicated data size
        clf (object): final fitted model using X_data and y_data variables (X_train and y_train if not specified)
    """

    """
    I changed my mind. We are going to stick with making this function 'dumb' 
    by supplying it with pre-determined X and y data to regress. This makes it the most
    universal
    """

    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Passed X and y data must be the same length!"

    # check for additional X_data and y_data in case we are using different data for regaularization. Assert fails mypy check, removing for now
    if X_data is None:
        X_data = X_train
        y_data = y_train
    # else:
    #     assert (
    #         X_data.shape[0] == y_data.shape[0]
    #     ), "Passed X and y data must be the same length!"

    # make space for randomization. Keep things fixed for now.
    rng = random.seed(42)
    # rng = random.randint(0, 10)
    Cs = np.logspace(-5, 5, 20)

    # going with Jackon's settings instead of my original ones just to make sure this works. Continuing to use Cs though.
    pre_clf = LogisticRegressionCV(
        Cs=Cs,
        l1_ratios=[0.8],
        solver="saga",
        penalty="elasticnet",
        n_jobs=3,
        cv=skf,
        max_iter=100000,
        scoring="balanced_accuracy",
        random_state=rng,
    ).fit(X_train, y_train)

    # coef = pre_clf.coef_[0]
    # cv_scores = np.mean(list(pre_clf.scores_.values())[0], axis=0)

    clf = LogisticRegression(
        C=pre_clf.C_[0],
        l1_ratio=pre_clf.l1_ratio_[0],
        solver="saga",
        penalty="elasticnet",
        max_iter=100000,
        random_state=rng,
    ).fit(X_data, y_data)

    nested_score = cross_val_score(
        clf, X=X_data, y=y_data, cv=skf, scoring="balanced_accuracy"
    ).mean()

    return nested_score, clf


def perform_linear_regression(X_train: pd.DataFrame, y_train: pd.DataFrame):
    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Passed X and y data must be the same length!"

    lreg = LinearRegression()

    param_grid = {"positive": [True]}
    tuning = GridSearchCV(estimator=lreg, param_grid=param_grid, cv=kf).fit(
        X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float)
    )
    tuned_model = tuning.best_estimator_

    nested_score = cross_val_score(
        tuned_model,
        X_train.to_numpy(dtype=float),
        y_train.to_numpy(dtype=float),
        cv=kf,
    ).mean()

    return nested_score, tuned_model


def perform_elastic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame):
    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Passed X and y data must be the same length!"

    X_train = X_train.to_numpy(dtype=float)
    y_train = y_train.to_numpy(dtype=float)

    tuned_eNet = ElasticNetCV(
        l1_ratio=np.arange(0.1, 1, 0.1),
        n_alphas=1000,
        alphas=[0.01, 0.1, 1, 10, 100],
        max_iter=100000,
        cv=kf,
        n_jobs=3,
        selection="random",
    ).fit(X_train, y_train)

    eNet = ElasticNet(
        alpha=tuned_eNet.alpha_,
        l1_ratio=tuned_eNet.l1_ratio_,
        max_iter=100000,
        selection="random",
    ).fit(X_train, y_train)

    # nested cross val
    nested_score = cross_val_score(
        eNet, X_train, y_train, cv=kf, n_jobs=3, scoring="r2"
    ).mean()

    return nested_score, eNet


def perform_PLSR(
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    components: int = 10,
):
    """
    Performs PLS Regression for given data at given component or defaults to performing
    on transposed (genes x patients) mrsa and candidemia data with 10 components.

    Parameters:
        X_data: (pd.DataFrame) | X data for analysis. Default = MRSA data from concat_datasets()
        y_data: (pd.DataFrame) | y data for analysis. Default = Candidemia data from concat_datasets()
        components: (int) | number of components to use for decomposition.

    Returns:
        pls (fitted object) | The PLSR object fitted to X_data and y_data
    """

    # for each components added, we are going to calculate R2Y and Q2Y, then compare them

    print(f"Performing PLSR for {components} components")
    pls = PLSRegression(n_components=components)
    pls.fit(X_data, y_data)
    print(f"Finished for {components} components")

    pls_scores = {"X": pd.DataFrame(pls.x_scores_), "Y": pd.DataFrame(pls.y_scores_)}
    pls_loadings = {
        "X": pd.DataFrame(pls.x_loadings_),
        "Y": pd.DataFrame(pls.y_loadings_),
    }

    # set up DataFrames for scores and loadings
    component_labels = np.arange(1, components + 1)
    pls_scores["X"] = pd.DataFrame(
        pls_scores["X"].values, index=X_data.index, columns=component_labels
    )
    pls_scores["Y"] = pd.DataFrame(
        pls_scores["Y"].values, index=y_data.index, columns=component_labels
    )
    pls_loadings["X"] = pd.DataFrame(
        pls_loadings["X"].values, index=X_data.columns, columns=component_labels
    )
    pls_loadings["Y"] = pd.DataFrame(
        pls_loadings["Y"].values, index=y_data.columns, columns=component_labels
    )

    return pls_scores, pls_loadings, pls


def caluclate_R2Y_Q2Y(model: PLSRegression, X_data: pd.DataFrame, y_data: pd.DataFrame):
    assert isinstance(
        model, PLSRegression
    ), "Passed model was not a PLSRegression object!"

    # calculate R2Y using score()
    R2Y = model.score(X_data, y_data)

    # calculate Q2Y using kFold cross-validation
    leave = KFold(n_splits=10)
    y_pred = y_data.copy()
    y_press = 0
    y_tss = 0

    # calculate Q2Y using sklearn's cross_val_predict
    y_pred = cross_val_predict(model, X_data, y_data, cv=leave)
    y_press = np.average((y_data - y_pred) ** 2)
    y_tss = np.average((y_data - y_data.mean()) ** 2)

    # calculate Q2Y
    Q2Y = 1 - (y_press / y_tss)

    return R2Y, Q2Y
