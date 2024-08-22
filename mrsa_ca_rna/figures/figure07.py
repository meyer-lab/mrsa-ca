"""R2Y vs Q2Y of MRSA x CA data via PLSR"""

from mrsa_ca_rna.regression import perform_PLSR, caluclate_R2Y_Q2Y
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.figures.base import setupBase

import pandas as pd
import seaborn as sns
import numpy as np


def figure07_setup():
    whole_data = concat_datasets()

    mrsa_df = whole_data[whole_data.obs["disease"] == "MRSA"].to_df()
    ca_df = whole_data[whole_data.obs["disease"] == "Candidemia"].to_df()
    # mrsa_whole = whole_data.loc["MRSA", :]
    # ca_whole = whole_data.loc["Candidemia", :]

    X_data = mrsa_df.T
    y_data = ca_df.T

    components = 10

    R2Ys = []
    Q2Ys = []

    # run PLSR with increasing components to find optimal number
    for i in range(components):
        _, _, fitted_pls = perform_PLSR(X_data, y_data, i + 1)

        r2y, q2y = caluclate_R2Y_Q2Y(fitted_pls, X_data, y_data)

        R2Ys.append(r2y)
        Q2Ys.append(q2y)

    # set up R2Y and Q2Y DataFrame to easily pass to the plotting in genFig()
    component_col = np.arange(1, components + 1)
    matrix = np.array([component_col.astype(int), R2Ys, Q2Ys]).T
    data = pd.DataFrame(matrix, columns=["components", "R2Y", "Q2Y"])

    return data


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data = figure07_setup()
    melt = data.melt(
        id_vars=["components"],
        value_vars=["R2Y", "Q2Y"],
        var_name="Metric",
        value_name="Score",
    )

    a = sns.barplot(melt, x="components", y="Score", hue="Metric", ax=ax[0])
    a.set_title(
        "Performance of PLSR with MRSA (X) data Regressed against CA (Y) data\n10-folds, new formula"
    )
    a.set_xlabel("# Components used in model")
    a.set_ylabel("Scores")

    return f
