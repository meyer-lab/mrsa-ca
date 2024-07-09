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
    Change LogisticRegression to LogisticRegressionCV

"""

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd

from mrsa_ca_rna.pca import perform_PCA


def perform_mrsa_LR(dataset:pd.DataFrame, components: int = 100):
    """
    Performs a logistic regression with cross validation on the
    PCA data of MRSA+CA+Healthy dataset.
    
    Parameters:
        dataset (pandas.DataFrame): scores matrix from PCA analysis
        components (int): first x components to run regression on, default=100
    
    Returns:
        clf (object): fitted model using specified components
    """
    
    assert (
        dataset.columns[2] == "PC1"
    ), "Could not perform regression: concat PCA matrix shape has changed or scores matrix not passed."

    mrsa_X = dataset.iloc[:, 2:components+2]
    mrsa_y = dataset.loc[:, "status"]

    # create some randomization later
    rng = 42

    skf = StratifiedKFold(n_splits=10)
    clf = LogisticRegressionCV(cv=skf, random_state=rng).fit(mrsa_X, mrsa_y)
    print(f"Score with first {components} components: {clf.score(mrsa_X, mrsa_y)}")

    return clf.score(mrsa_X, mrsa_y), clf
