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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    KFold,
    LeaveOneOut,
    cross_val_score,
)
from sklearn.linear_model import (
    LogisticRegressionCV,
    LogisticRegression,
    ElasticNet,
    ElasticNetCV,
    LinearRegression,
)
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import numpy as np
import random
import warnings

from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_PCA

skf = StratifiedKFold(n_splits=10)
kf = KFold(n_splits=10)
loocv = LeaveOneOut()


def perform_PC_LR(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_data: pd.DataFrame = None,
    y_data: pd.DataFrame = None,
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

    # check for additional X_data and y_data in case we are using different data for regaularization
    if X_data is None:
        X_data = X_train
        y_data = y_train
    else:
        assert (
            X_data.shape[0] == y_data.shape[0]
        ), "Passed X and y data must be the same length!"

    # make space for randomization. Keep things fixed for now.
    random.seed(42)
    rng = random.randint(0, 10)

    Cs = np.logspace(-5, 5, 20)

    # scaler = StandardScaler().set_output(
    #     transform="pandas"
    # )  # scaling again just in case?

    convergence_failure = 0

    # going with Jackon's settings instead of my original ones just to make sure this works. Continuing to use Cs though.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        pre_clf = LogisticRegressionCV(
            Cs=Cs,
            l1_ratios=[0.8],
            solver="saga",
            penalty="elasticnet",
            n_jobs=3,
            cv=skf,
            max_iter=100000,
            scoring="balanced_accuracy",
            multi_class="ovr",
            random_state=rng,
        ).fit(X_train, y_train)

        if any([issubclass(warning.category, ConvergenceWarning) for warning in w]):
            print(
                f"ConvergenceWarning detected using {len(X_train.columns)} components ({np.amax(pre_clf.n_iter_)} iterations)"
            )
            convergence_failure = len(X_train.columns)
        else:
            print(
                f"Convergence achieved within {np.amax(pre_clf.n_iter_)} iterations using {len(X_train.columns)} components."
            )

    coef = pre_clf.coef_[0]
    cv_scores = np.mean(list(pre_clf.scores_.values())[0], axis=0)

    clf = LogisticRegression(
        C=pre_clf.C_[0],
        l1_ratio=pre_clf.l1_ratio_[0],
        solver="saga",
        penalty="elasticnet",
        max_iter=100000,
    ).fit(X_data, y_data)

    nested_score = cross_val_score(
        clf, X=X_data, y=y_data, cv=skf, scoring="balanced_accuracy"
    ).mean()

    return (nested_score, convergence_failure, clf)


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

    # eNet = ElasticNet(max_iter=100000)
    # param_grid = {
    #     "alpha": [0.01, 0.1, 1, 10, 100],
    #     "l1_ratio": np.arange(0.1, 1.0, 0.1),
    #     "tol": [0.0001, 0.001],
    #     "selection": ["random", "cyclic"],
    # }

    # grid_search = GridSearchCV(
    #     eNet,
    #     param_grid,
    #     scoring="r2",
    #     cv = skf,
    #     return_train_score=True,
    #     n_jobs=3
    # ).fit(X_train, y_train)

    # print("Finished hyperparameter fitting.")
    # print(f"Best params: {grid_search.best_params_}\nBest score: {grid_search.best_score_}\nBest estimator: {grid_search.best_estimator_}")
    # results_dict = {"params": grid_search.best_params_, "score": grid_search.best_score_, "estimator": grid_search.best_estimator_}

    # tuned_eNet = grid_search.best_estimator_

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
