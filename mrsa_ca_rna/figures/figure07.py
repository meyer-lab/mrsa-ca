"""R2Y vs Q2Y of MRSA x CA data via PLSR"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.regression import caluclate_R2Y_Q2Y, perform_PLSR


def figure07_setup():
    # do I want to not scale the data prior to splitting it and performing PLSR?
    whole_data = concat_datasets(scale=False, tpm=True)

    mrsa_df = whole_data[whole_data.obs["disease"] == "MRSA"].to_df()
    ca_df = whole_data[whole_data.obs["disease"] == "Candidemia"].to_df()

    # independently scale, using StandardScaler, the two datasets to avoid data leakage
    scaler = StandardScaler()
    mrsa_df.loc[:, :] = scaler.fit_transform(mrsa_df.values)
    ca_df.loc[:, :] = scaler.fit_transform(ca_df.values)

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
    data = pd.DataFrame(matrix, columns=pd.Index(["components", "R2Y", "Q2Y"]))

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
        "Performance of PLSR with MRSA (X) data Regressed against CA (Y) data\n"
        "10-folds, new formula"
    )
    a.set_xlabel("# Components used in model")
    a.set_ylabel("Scores")

    return f
