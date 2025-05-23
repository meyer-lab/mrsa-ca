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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    KFold,
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


def perform_PLSR(
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    components: int = 10,
):
    """
    Performs PLS Regression for given data at given component or defaults to performing
    on transposed (genes x patients) mrsa and candidemia data with 10 components.

    Parameters:
        X_data: (pd.DataFrame) | X data for analysis.
                                Default = MRSA data from concat_datasets()
        y_data: (pd.DataFrame) | y data for analysis.
                                Default = Candidemia data from concat_datasets()
        components: (int) | number of components to use for decomposition.

    Returns:
        pls (fitted object) | The PLSR object fitted to X_data and y_data
    """

    # for each components added, we are going to calculate R2Y and Q2Y,
    # then compare them

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
    assert isinstance(model, PLSRegression), (
        "Passed model was not a PLSRegression object!"
    )

    # calculate R2Y using score()
    R2Y = model.score(X_data, y_data)

    # calculate Q2Y using kFold cross-validation
    y_pred = y_data.copy()
    y_press = 0.0
    y_tss = 0.0

    # calculate Q2Y using sklearn's cross_val_predict
    y_pred = cross_val_predict(model, X_data, y_data, cv=KFold(n_splits=10), n_jobs=10)
    y_press = np.average((y_data - y_pred) ** 2)
    y_tss = np.average((y_data - y_data.mean()) ** 2)

    # calculate Q2Y
    Q2Y = 1 - (y_press / y_tss)

    return R2Y, Q2Y
