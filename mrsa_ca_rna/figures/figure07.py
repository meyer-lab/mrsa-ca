"""R2Y vs Q2Y of MRSA x CA data via PLSR"""

from mrsa_ca_rna.regression import perform_PLSR
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.figures.base import setupBase

from sklearn.model_selection import KFold

import pandas as pd
import seaborn as sns
import numpy as np

def figure07_setup():

    whole_data = concat_datasets()

    mrsa_whole = whole_data.loc["MRSA", :]
    ca_whole = whole_data.loc["Candidemia", :]

    X_data = mrsa_whole["rna"].T
    y_data = ca_whole["rna"].T

    components = 10

    R2Ys = []
    Q2Ys = []

    for i in range(components):

        fitted_pls = perform_PLSR(X_data, y_data, i+1)


        # calculate R2Y using score()
        R2Ys.append(fitted_pls.score(X_data, y_data))

        # calculate Q2Y using LeaveOneOut
        leave = KFold(n_splits=10)
        y_pred = y_data.copy()
        y_diff = 0
        y_true = 0
        for train_index, test_index in leave.split(X_data, y_data):
            X_train = X_data.iloc[train_index, :]
            y_train = y_data.iloc[train_index, :]

            trained_pls = perform_PLSR(X_train, y_train, i+1)
            y_pred.iloc[test_index, :] = trained_pls.predict(X_data.iloc[test_index, :])

            y_diff += np.average((y_pred.iloc[test_index, :] - y_data.iloc[test_index, :])**2)
            y_true += np.average((y_data.iloc[test_index, :])**2)

        Q2Ys.append(1-(y_diff/y_true))

    component_col = np.arange(1, components+1)
    matrix = np.array([component_col.astype(int), R2Ys, Q2Ys]).T
    data = pd.DataFrame(matrix, columns=["components", "R2Y", "Q2Y"])

    return data


def genFig():

    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)


    data = figure07_setup()
    melt = data.melt(id_vars=["components"], value_vars=["R2Y", "Q2Y"], var_name="Metric", value_name="Score")

    a = sns.barplot(melt, x="components", y="Score", hue="Metric", ax=ax[0])
    a.set_title("Performance of PLSR with MRSA (X) data Regressed against CA (Y) data\n10-folds")
    a.set_xlabel("# Components used in model")
    a.set_ylabel("Scores")

    return f