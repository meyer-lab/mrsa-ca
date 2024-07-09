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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.pca import concat_datasets

def perform_LR():

    whole_scores, whole_loadings, pca = perform_PCA()
    mrsa_scores = whole_scores.loc[whole_scores["disease"] == "mrsa"]
    mrsa_data = mrsa_scores.loc[~(mrsa_scores["status"]=="Unknown")]
    mrsa_X = mrsa_data.loc[:,"PC1":"PC100"]
    mrsa_y = mrsa_data.loc[:,"status"]

    # create some randomization later
    rng = 42

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(mrsa_X, mrsa_y, random_state=rng, )

    # scale separately to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    mrsa_Lreg = LogisticRegression(C=10).fit(X_train, y_train)
    
    score_train = mrsa_Lreg.score(X_train, y_train)
    score_test = mrsa_Lreg.score(X_test, y_test)
    confusion_train = confusion_matrix(y_train, mrsa_Lreg.predict(X_train))
    confusion_test = confusion_matrix(y_test, mrsa_Lreg.predict(X_test))

    print(f"Score and confusion matrix of training: {score_train} and {confusion_train}")
    print(f"Score and confusion matrix of test: {score_test} and {confusion_test}")

    return mrsa_Lreg

"""debug calls"""
perform_LR()