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
    Scale the data prior to running regression
        Avoid data leakage?

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
import pandas as pd

from mrsa_ca_rna.pca import perform_PCA, perform_PCA_validation


def perform_mrsa_LR(
    training: pd.DataFrame, validation: pd.DataFrame, components: int = 100
):
    """
    Performs a logistic regression with cross validation on the
    PCA data of MRSA+CA+Healthy dataset.

    Parameters:
        training (pandas.DataFrame): scores matrix from PCA analysis
        validation (pandas.DataFrame): scores matrix of a PCA'd validation set
        components (int): first x components to run regression on, default=100

    Returns:
        clf (object): fitted model using specified components
    """

    assert (
        training.columns[2] == "PC1"
    ), "Could not perform regression: concat PCA matrix shape has changed or scores matrix not passed."

    mrsa_X = training.iloc[:, 2 : components + 2]
    mrsa_y = training.loc[:, "status"]

    mrsa_test_X = validation.iloc[:, 2:components+2]
    mrsa_test_y = validation.loc[:, "status"].astype(str) # mrsa_y is stored as strings '0' and '1' regression trained on strings fails on ints

    # create some randomization later
    rng = 42

    skf = StratifiedKFold(n_splits=10)
    clf = LogisticRegressionCV(cv=skf, random_state=rng).fit(mrsa_X, mrsa_y)

    return clf.score(mrsa_X, mrsa_y), clf.score(mrsa_test_X, mrsa_test_y), clf
