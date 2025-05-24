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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)


def perform_LR(
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    splits: int = 10,
) -> tuple[float, np.ndarray, LogisticRegressionCV]:
    """
    Agnostically performs LogisticRegression
    with nested cross validation to passed data.


    Parameters:
        X_data (pd.DataFrame): X data for regularization
                                and subsequent nested cross validation
        y_data (pd.DataFrame): y data for regularization
                                and subsequent nested cross validation
        splits (int): number of splits for nested cross validation.
                        Default = 10

    Returns:
        nested_score (float): nested cross validation score of the final model
        nested_proba (np.ndarray): nested cross validation predicted probabilities
                                    of the final model
        clf_cv (LogisticRegressionCV): classifier object fit to the data
    """

    assert X_data.shape[0] == y_data.shape[0], (
        "Passed X and y data must be the same length!"
    )

    # set up stratified kfold for nested cross validation
    skf = StratifiedKFold(n_splits=splits)

    # perform logistic regression with nested cross validation
    # eventually settle on a single l1_ratio?
    clf_cv = LogisticRegressionCV(
        l1_ratios=[0.2, 0.5, 0.8],
        solver="saga",
        penalty="elasticnet",
        n_jobs=10,
        cv=skf,
        max_iter=100000,
        scoring="balanced_accuracy",
    ).fit(X_data, y_data)

    nested_score = cross_val_score(
        clf_cv, X=X_data, y=y_data, cv=skf, scoring="balanced_accuracy", n_jobs=10
    ).mean()

    nested_proba = cross_val_predict(
        clf_cv, X=X_data, y=y_data, cv=skf, method="predict_proba", n_jobs=10
    )
    nested_proba = np.array(nested_proba)

    return nested_score, nested_proba, clf_cv
