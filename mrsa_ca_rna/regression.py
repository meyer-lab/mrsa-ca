"""
This file will include logstical regression methods
and other relevant functions, starting with MRSA data,
to determine patterns of persistance vs. resolving infection
to eventually compare across disease types to see if
these observed patterns are preserved.

Main issue: current function setup involves performing pca
    on all the data at once, after which I need to pick out
    specific data. This can likely lead to really bad regression
    data.

To-do:
    Scale the data prior to running regression.
        Scale PCA output prior to regression?
            This removes Sigma?
        Follow 'importance of feature scaling' using pipeline
        just to try it out.

    All data PCA'd prior to use. Should I bring in a separate
        un-PCA'd validation set, PCA it alone, then predict
        using the current model?

    Using CA metadata, create a function marking persis/resolve on
        CA data using the probabilities plot present in the paper.
        "After 30 days patient is 30% more likely to be healthy"
        etc.



"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import numpy as np
import random
import warnings


def perform_LR(annot_data: pd.DataFrame, components: int = 60):
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
        clf = LogisticRegressionCV(Cs=Cs, cv=skf, max_iter=1000, random_state=rng).fit(
            X_train, y_train
        )

        if any([issubclass(warning.category, ConvergenceWarning) for warning in w]):
            print(f"ConvergenceWarning detected at {components} selected components")
            convergence_failure = components
        else:
            print(f"Convergence achieved without issue at {components} components.")

    return (
        clf.score(X_train, y_train),
        convergence_failure,
        clf,
    )
