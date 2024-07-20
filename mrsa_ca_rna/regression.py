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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import numpy as np
import random
import warnings

from mrsa_ca_rna.import_data import concat_datasets


def perform_PC_LR(annot_data: pd.DataFrame, components: int = 60):
    """
    Performs a logistic regression with cross validation on the passed data. Expects
    data to be annotated rna expression dataframes with columns[0,1] = "status", "disease"
    and all subsequent data rna expressions. Only performs regression on statused data,
    anything not labeled as "NA" or "Unknown."

    Parameters:
        annot_data (pandas.DataFrame): annotated dataframe containing rna expression
        components (int): first x components to run regression on, default=60

    Returns:
        clf.score(X_train, y_train): the score of the model
        clf (object): fitted model using specified components
    """

    # For now, we are expecting to be handling PCA'd data.
    assert (
        annot_data.columns[2] == "PC1"
    ), "Could not perform regression: concat PCA matrix shape has changed or scores matrix not passed."

    component_range = np.arange(2, components + 2)

    # we're grabbing the data relevant to our regression method, leaving out anything that cannot be regressed against.
    X_train = annot_data.loc[
        ~annot_data["status"].str.contains("Unknown|NA"),
        annot_data.columns[component_range],
    ]
    y_train = annot_data.loc[~annot_data["status"].str.contains("Unknown|NA"), "status"]

    # make space for randomization. Keep things fixed for now.
    random.seed(42)
    rng = random.randint(0, 10)

    Cs = np.logspace(-5, 5, 20)
    skf = StratifiedKFold(n_splits=10)
    # scaler = StandardScaler().set_output(
    #     transform="pandas"
    # )  # scaling again just in case?

    convergence_failure = 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegressionCV(Cs=Cs, cv=skf, max_iter=6000, random_state=rng).fit(
            X_train, y_train
        )

        if any([issubclass(warning.category, ConvergenceWarning) for warning in w]):
            print(
                f"ConvergenceWarning detected using {components} components ({np.amax(clf.n_iter_)} iterations)"
            )
            convergence_failure = components
        else:
            print(
                f"Convergence achieved within {np.amax(clf.n_iter_)} iterations using {components} components."
            )

    return (
        clf.score(X_train, y_train),
        convergence_failure,
        clf,
    )


def perform_whole_LR():
    """
    This function will perform a logistic regression on the entire dataset (unPCA'd) and regress
    against MRSA outcomes, like the perform_PC_LR. Eventually spend some time to make a generalized
    LR function to handle any supplied dataset.

    Returns:
        whole_scaled_clf.score (float): score from performing elasticnet logistic regression on MRSA data
        weights (pandas.DataFrame): coef_ of features after fitting
        whole_scaled_clf (object): object for further use
    """

    annot_data = concat_datasets()

    X_train = annot_data.loc[
        annot_data["disease"].str.contains("MRSA"),
        ~annot_data.columns.str.contains("status|disease"),
    ]
    y_train = annot_data.loc[~annot_data["status"].str.contains("Unknown|NA"), "status"]

    scaler = StandardScaler().set_output(transform="pandas")
    X_scaled_train = scaler.fit_transform(X_train)

    random.seed(42)
    rng = random.randint(0, 10)

    Cs = np.logspace(-5, 5, 20)
    skf = StratifiedKFold(n_splits=10)

    failure = 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        whole_scaled_clf = LogisticRegressionCV(
            verbose=True,
            Cs=Cs,
            cv=skf,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.75, 0.25],
            max_iter=6000,
            random_state=rng,
        ).fit(X_scaled_train, y_train)

        if any([issubclass(warning.category, ConvergenceWarning) for warning in w]):
            print(
                f"ConvergenceWarning detected. Max iterations hit: {np.amax(whole_scaled_clf.n_iter_)}"
            )
            failure = 1
        else:
            print(f"Convergence achieved within {np.amax(whole_scaled_clf.n_iter_)}")

    print(
        f"score: {whole_scaled_clf.score(X_scaled_train, y_train)}. Failures to converge: {bool(failure)}"
    )
    print(f"The coefficients of the fitting function are: {whole_scaled_clf.coef_}")

    coef_array = whole_scaled_clf.coef_

    weights = pd.DataFrame(
        coef_array, index=["Scaled Feature Coef"], columns=annot_data.columns[2:]
    )

    weights.to_csv("./output/weights_scaled.csv")

    return whole_scaled_clf.score(X_scaled_train, y_train), weights, whole_scaled_clf
